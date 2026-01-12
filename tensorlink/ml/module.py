from accelerate import init_empty_weights
from transformers import (
    PreTrainedModel,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForVision2Seq,
    AutoModelForSpeechSeq2Seq,
)
from typing import Generator, List, Optional, Type, Dict, Any, Union
from huggingface_hub import snapshot_download
from contextlib import contextmanager
from safetensors import safe_open
import torch.optim as optim
import torch.nn as nn
import threading
import logging
import pickle
import torch
import types
import queue
import glob
import time
import json
import gc
import io
import os

from tensorlink.ml.injector import generate_new_forward_method, get_loop_io_signature
from tensorlink.ml.optim import DistributedParameter, create_distributed_optimizer
from tensorlink.ml.graphing import ModelParser
from tensorlink.ml.utils import (
    get_gpu_memory,
    get_batch_size,
    chunk,
    handle_output,
    combine_micro_batches,
    split_into_micro_batches,
    replace_output_with_custom_grad,
    access_module,
    detach_tensor,
    bytes_to_tensor,
    tensor_to_bytes,
    enable_grad,
    get_nested_module,
    get_optimizer_from_spec,
    optimizer_to_spec,
)
from tensorlink.nodes.shared_memory import (
    get_from_shared_memory,
    store_in_shared_memory,
)


MAX_WAIT_TIME = 150


def contains_offloaded(module: nn.Module):
    if not list(module.named_children()):
        return False
    children = list(module.children())
    exists = False

    for child in children:
        # check if insntance offloadedsubmodule
        if isinstance(child, OffloadedModule):
            return True
        exists = exists or contains_offloaded(child)

    return exists


def _confirm_action():
    """
    Prompts the user with a confirmation message before proceeding.
    """
    while True:
        response = (
            input(
                "Trusted mode is enabled. Are you sure you want to proceed? (yes/no, y/n): "
            )
            .strip()
            .lower()
        )
        if response in {"yes", "y"}:
            print("Proceeding with trusted mode.")
            break
        elif response in {"no", "n"}:
            print("Aborting initialization in trusted mode.")
            exit(1)
        else:
            print("Invalid input. Please type 'yes'/'y' or 'no'/'n'.")


@contextmanager
def _set_micro(local: threading.local, micro: int):
    """Context manager to set thread-local micro id and remove it on exit."""
    setattr(local, "micro", micro)
    try:
        yield
    finally:
        # remove attribute so thread reuse won't leak it
        try:
            delattr(local, "micro")
        except AttributeError:
            pass


class CustomAutogradRouter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model, output_tensor):
        """
        Forward pass, save the model in the context (ctx) for use during the backward pass.
        """
        ctx.model = model
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass, call the model's custom backward method.
        """
        # Call the model's custom backward method
        grad_input = ctx.model.backward(grad_output)

        # Return gradients for model and output_tensor
        return None, grad_input


def _create_user_node():
    """Create a User instance if one wasn't provided."""
    from tensorlink.nodes import User, UserConfig

    node = User(
        config=UserConfig(
            upnp=True, off_chain_test=False, local_test=False, print_level=logging.INFO
        )
    )

    # Allow time for node to initialize
    time.sleep(1)
    return node


class DistributedModel(nn.Module):
    """
    A modular distributed model that supports offloading submodules
    while handling local operations. This model can be instantiated
    by either a Worker or a User, where the host is referred to as the 'master' node.

    Features:
    - Handles distributed training across multiple nodes.
    - Supports user-defined optimizers and learning rate schedulers.
    - Provides control over GPU memory allocation and computation pipelines.
    - Enables trusted execution mode with an explicit confirmation.
    """

    def __init__(
        self,
        model: Union[nn.Module, str],
        n_pipelines: int = 1,
        optimizer: Optional[Type[optim.Optimizer]] = None,
        scheduler_type: Optional[Type[optim.lr_scheduler._LRScheduler]] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        trusted: bool = False,
        node: Optional[Any] = None,
        training: bool = True,
        verbose: bool = False,
        tokenizer=None,
    ):
        """
        Args:
            model (nn.Module): The base model to distribute.
            n_pipelines (int): Number of parallel pipelines for computation.
            optimizer (Type[optim.Optimizer]): Optimizer class to use.
            scheduler_type (Optional[Type[optim.lr_scheduler._LRScheduler]]):
                Optional learning rate scheduler.
            device (Optional[str]): Device to run the model on (default: auto-detect).
            dtype (torch.dtype): Data precision for computations.
            trusted (bool): If True, requires user confirmation before execution.
            node (Optional[Any]): Pre-existing node instance for networking.
            verbose (bool): Enables debug messages if True.
            tokenizer: can be specified for inference
        """
        super().__init__()

        if isinstance(model, nn.Module):
            self.name = str(model).split("(")[0]
            self.model: nn.Module = model.to(dtype=dtype)
        else:
            self.model_name = model
            self.model = None

        self.tokenizer = tokenizer

        # Store model and training resources
        self.user_memory = get_gpu_memory()
        self.model_parser = ModelParser(self.user_memory)

        self.worker_info: Dict[str, Any] = {}

        # Parallelization settings
        self.n_pipelines = n_pipelines
        self.n_datalines = 1  # Default data pipeline setting

        # Distributed graph and parameters
        self.config = {}
        self.distributed_graph: Dict[str, Any] = {}
        self.parameters_storage: Dict[str, torch.Tensor] = {}
        self.my_modules = set()

        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Optimizer and scheduler placeholders
        self.optimizer = optimizer
        self.scheduler = scheduler_type
        self.training = training

        # Trusted execution mode
        self.trusted = trusted
        if trusted:
            _confirm_action()

        # Setup node if not provided
        self.node = node if node else _create_user_node()

        # Node communication attributes
        self.node_requests = self.node.node_requests
        self.node_responses = self.node.node_responses
        self.mpc_lock = self.node.mpc_lock
        self._thread_local = threading.local()

        self.hf_cache_dir = os.environ.get(
            'HF_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
        )

        self.job_id = None

        if verbose:
            print(f"DistributedModel '{self.name}' initialized on {self.device}")

        # Initialize model distribution
        if self.node.__class__.__name__ == "User":
            self._initialize_distribution()

    def forward(self, *args, **kwargs):
        """
        Performs the forward pass through the model.
        - Splits input into micro-batches and runs them in parallel.
        - Creates multiple parallel streams of workers for model parallel acceleration
        """
        if isinstance(args, tuple) and len(args) == 1:
            args = args[0]

        batch_size = get_batch_size(args)

        assert (
            batch_size % self.n_datalines == 0
        ), "Batch size must be divisible by n_datalines!"

        assert (
            batch_size // self.n_datalines % self.n_pipelines == 0
        ), "Minibatch size must be divisible by n_pipelines!"

        # Split the input tensor and kwargs into micro-batches
        micro_batch_args = chunk(args, self.n_pipelines)
        micro_batch_kwargs = chunk(kwargs, self.n_pipelines)

        # Create queues for forward and backward passes
        self.model.forward_queues = {m: queue.Queue() for m in range(self.n_pipelines)}
        self.model.backward_queues = {m: queue.Queue() for m in range(self.n_pipelines)}
        self.model.outputs = [None] * self.n_pipelines
        self.model.intermediates = {m: [] for m in range(self.n_pipelines)}

        # Spawn separate threads for each micro batch for pipeline parallelism
        threads = []
        for micro in range(self.n_pipelines):
            micro_args = micro_batch_args[micro]
            micro_kwargs = micro_batch_kwargs[micro]
            t = threading.Thread(
                target=self._perform_micro_forward,
                args=(micro, micro_args),
                kwargs=micro_kwargs,
            )
            t.start()
            threads.append(t)
            time.sleep(0.1)

        for t in threads:
            t.join()

        if not self.training:
            self.model.n_batch += 1

        combined_output = combine_micro_batches(self.model.outputs)
        output_tensor = handle_output(combined_output)
        custom_grad_output = CustomAutogradRouter.apply(self, output_tensor)
        return replace_output_with_custom_grad(combined_output, custom_grad_output)

    def _perform_micro_forward(self, micro, *args, **kwargs):
        with _set_micro(self._thread_local, micro):
            x = self.model(*args, **kwargs)
            self.model.outputs[micro] = x

    def backward(self, loss):
        """
        Performs the backward pass through the model.
        - Splits input into micro-batches and runs them in parallel.
        """
        if hasattr(loss, "micro_loss"):
            micro_losses = loss.micro_loss
        else:
            # Split the combined loss into micro-batch-specific losses
            micro_losses = split_into_micro_batches(loss, self.n_pipelines)

        threads = []
        for micro in range(self.n_pipelines):
            t = threading.Thread(
                target=self._perform_micro_backward, args=(micro, micro_losses[micro])
            )
            t.start()
            threads.append(t)
            time.sleep(0.1)

        for t in threads:
            t.join()

        self.model.n_batch += 1

    def _perform_micro_backward(self, micro, loss):
        """
        Process each backward stream
        """
        with _set_micro(self._thread_local, micro):
            # Get the oldest epoch intermediates from storage via dict key lookup (todo)
            while len(self.model.intermediates[micro]) > 0:
                vals = self.model.intermediates[micro].pop(-1)

                # Len vals 2 means backwards pass of first & last submodule
                if len(vals) == 2:
                    val1, val2 = vals

                    # Pass of the first submodule / section contains tensor in the first position
                    if isinstance(val1, torch.Tensor):
                        assoc_output, _ = vals
                        loss = assoc_output.backward(loss, retain_graph=True)

                        if loss is None:
                            # If loss is None, use the associated output for backward pass
                            loss = assoc_output

                    # Pass of the last section
                    else:
                        module_id_bytes, assoc_input = vals
                        tag = [self.model.n_batch, micro, module_id_bytes]

                        # If there are remaining computations between output and last submodule
                        if loss.grad_fn is not None:
                            loss.backward()
                            loss = assoc_input.grad

                        start_time = time.time()
                        worker_id = self.distributed_graph[module_id_bytes][
                            "assigned_workers"
                        ][-1]
                        key = tag[:2] + [module_id_bytes, module_id_bytes]

                        if self.trusted:
                            size, shm_name = store_in_shared_memory(
                                (detach_tensor(loss), None)
                            )
                        else:
                            loss_bytes = tensor_to_bytes(loss)
                            size, shm_name = store_in_shared_memory(
                                loss_bytes, encoded=True
                            )

                        self.send_request(
                            "send_backward", (worker_id, size, shm_name, tag)
                        )

                        # Wait for response, change to appending waiting thread to list in master
                        waiting = True
                        while waiting:
                            time.sleep(0.1)
                            args = self.send_request("check_backward", key)

                            if args is not None:
                                waiting = False
                                size, name = args

                                if self.trusted:
                                    loss = get_from_shared_memory(size, name)

                                else:
                                    loss_bytes = get_from_shared_memory(
                                        size, name, encoded=True
                                    )
                                    loss = bytes_to_tensor(loss_bytes)

                            if time.time() - start_time >= MAX_WAIT_TIME:
                                # Logic here to request another worker take his place
                                waiting = False

                        self.send_request(
                            "release_memory",
                            ("backward_queue", module_id_bytes, tuple(tag)),
                        )

                elif len(vals) == 1:
                    assoc_input = vals[0]
                    if isinstance(assoc_input, torch.Tensor):
                        assoc_input.backward(loss)
                else:
                    raise "Expect vals to be of length 1 or 2."

    def get_info_from_module_id(self, mod_id: list, micro: int = None):
        for info in self.distributed_graph.values():
            if mod_id == info["mod_id"]:
                if micro:
                    return info["id_hash"], info["assigned_workers"][micro]
                else:
                    return info["id_hash"]

    def train(self, mode: bool = True):
        """
        Sets the training mode for the model and sends requests to offloaded modules to update their training state.
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")

        self.training = mode
        threads = []

        def recurse_modules(module):
            if isinstance(module, OffloadedModule):
                self.send_request(
                    "update_train", (module.worker_id, mode, module.module_id)
                )
                t = threading.Thread(
                    target=self._wait_for_state_update,
                    args=(module.module_id, mode),
                )
                t.start()
                threads.append(t)

            elif contains_offloaded(module):
                for child in module.children():
                    recurse_modules(child)

        recurse_modules(self.model)

        for thread in threads:
            thread.join()

    def eval(self):
        self.train(False)

    def children(self):
        # If the model is an instance of OffloadedModule, return an iterator with only itself.
        if isinstance(self.model, OffloadedModule):
            yield self.model  # Just yield the OffloadedModule, don't dive into its children.
        else:
            # Otherwise, yield the model itself and then recursively yield its children.
            yield self.model
            yield from self.model.children()

    def parameters(
        self, recurse: bool = True, distributed: bool = True, load: bool = True
    ) -> Generator:
        """
        Collects parameters from all modules (including offloaded) asynchronously and returns them as a generator.
        """
        if distributed:
            parameters = []
            request_threads = []
            file_references = {}

            def collect_parameters(module):
                for child in module.children():
                    if isinstance(child, OffloadedModule):
                        # Asynchronously request parameters from offloaded modules
                        self.send_request(
                            "request_parameters", (child.worker_id, child.module_id)
                        )
                        # Start a thread to wait for each response
                        t = threading.Thread(
                            target=self._wait_for_parameters, args=(child.module_id,)
                        )
                        t.start()
                        request_threads.append(t)
                    elif contains_offloaded(child):
                        collect_parameters(child)
                    else:
                        parameters.extend(list(child.parameters()))

            # Recursively collect parameters from main and offloaded modules
            collect_parameters(self)

            # Ensure all parameter requests are completed
            for thread in request_threads:
                thread.join()

            # Handle each module's parameters, loading only if `load` is True
            for module_id, module_info in self.distributed_graph.items():
                if "offloaded" in module_info["type"]:
                    file_name = self.parameters_storage[module_id]
                    if load:
                        # Load parameters from file
                        with open(file_name, "rb") as f:
                            state_dict = torch.load(f)
                            parameters.extend(state_dict.values())
                    else:
                        # Store the reference to the file for lazy access
                        file_references[module_id] = file_name
                        os.makedirs("models", exist_ok=True)
                        os.makedirs(f"models/{self.name}", exist_ok=True)
                        os.rename(
                            file_name,
                            os.path.join("models", self.name, file_name.split("/")[1]),
                        )
                else:
                    # Directly collect parameters for in-memory modules
                    module, _ = access_module(self.model, module_info["mod_id"])
                    parameters.extend(list(module.parameters()))

            # Define a generator to yield either parameters or file references
            def parameter_generator():
                if load:
                    # Yield loaded or in-memory parameters directly
                    for param in parameters:
                        yield param
                else:
                    # Yield file references instead of loading parameters
                    for _module_id, file_name in file_references.items():
                        yield file_name  # Yield file paths if `load` is False

            return parameter_generator()

        # If not distributed, return parameters from the model directly
        return (param for param in self.model.parameters(recurse))

    def _wait_for_parameters(self, module_id: bytes):
        """
        Waits until parameters have been received from the specified offloaded module
        """
        # Wait for response, change to appending waiting thread to list in master
        start_time = time.time()
        waiting = True
        while waiting:
            time.sleep(0.25)
            args = self.send_request("check_parameters", module_id)

            if args is not None:
                waiting = False
                file_name = args

                self.parameters_storage[module_id] = file_name

            if time.time() - start_time >= MAX_WAIT_TIME:
                # Logic here to request another worker take his place
                waiting = False

    def _wait_for_state_update(self, module_id, state):
        """
        Waits until a state update has been confirmed from the specified offloaded module
        """
        # Wait for response, change to appending waiting thread to list in master
        start_time = time.time()
        waiting = True
        while waiting:
            time.sleep(0.1)
            args = self.send_request("check_train", module_id)

            if args is not None:
                if args == state:
                    waiting = False

            if time.time() - start_time >= MAX_WAIT_TIME:
                # Logic here to request another worker take his place
                waiting = False

    def get_node_most_memory(self):
        # gets roles with most mpc, returns key and mpc
        key = max(
            self.master_node.nodes, key=lambda x: self.master_node.nodes[x]["mpc"]
        )
        return key, self.master_node.nodes[key]["mpc"]

    def distribute_model(self, config=None, model_type: str = "chat"):
        # Retrieve model names and assign workers to offload. Contact candidate workers
        # and ensure they are ready to receive the model / train
        if config is None:
            config = self.parse_model(self.model, self.max_module_size)

        config = (
            config | self.config
        )  # Update config with any modules we host on our this device

        self.distributed_graph = config

        if self.model_name:
            self._load_model_skeleton(model_type)

        grouped_layers = {}
        host_modules = {}

        for module_id, module_info in config.items():
            if self.model_name:
                module_type = module_info.get('type', 'offloaded')

                if module_type == 'offloaded_group':
                    grouped_layers[module_id] = module_info
                elif module_type == 'loaded':
                    host_modules[module_id] = module_info
                else:
                    self._wrap_hf_module(module_id, module_info)
            else:
                # self.wrap_module(config)
                raise "Custom models are currently not supported."

        if host_modules:
            self._load_host_modules(host_modules)

        if grouped_layers:
            self._wrap_grouped_layers(grouped_layers)

        assert isinstance(self.model, nn.Module), "Model Distribution Failed!"

        self.model.n_batch = 0
        self.model.forward_queues = {}
        self.model.backward_queues = {}
        self.model.intermediates = [
            []
        ]  # Queue to hold intermediates, must be converted into dictionary
        # of queues if we wish to perform multiple epochs concurrently

        return 0, 0

    def generate(self, *args, **kwargs):
        """
        Generate method.

        Args:
            *args: Input tensors
            **kwargs: Additional generation parameters
        """
        with _set_micro(self._thread_local, 0):
            return self.model.generate(*args, **kwargs)

    def wrap_module(self, module_id: list, worker_id):
        # Access the module and parent
        if module_id == [-1]:  # Handling the case of the full model
            child_module = self.model
            parent_module = None  # No parent when the full model is offloaded
        else:
            child_module, module_name = access_module(self.model, module_id)
            parent_module = (
                access_module(self.model, module_id[:-1])[0]
                if len(module_id) > 1
                else self.model
            )

        module_hash = self.get_info_from_module_id(module_id)

        # Assign metadata to child_module
        child_module.id = module_hash
        child_module.n_batch = 0

        file_name = f"{module_hash}_{worker_id}.pt"
        # TODO Custom pickle function OR send state dict and recreate a model on other side
        if self.trusted:
            with open(file_name, "wb") as f:
                pickle.dump(child_module, f)

        elif isinstance(child_module, PreTrainedModel):
            metadata_bytes = json.dumps(
                {
                    "module_config": child_module.config.to_dict(),
                    "module_class": child_module.__class__.__name__,
                }
            ).encode("utf-8")

            state_dict = child_module.state_dict()
            sorted_state_dict = {k: state_dict[k] for k in sorted(state_dict.keys())}
            state_dict_buffer = io.BytesIO()
            torch.save(sorted_state_dict, state_dict_buffer)
            state_dict_bytes = state_dict_buffer.getvalue()

            buffer = io.BytesIO()
            buffer.write(len(metadata_bytes).to_bytes(4, "big"))
            buffer.write(metadata_bytes)
            buffer.write(state_dict_bytes)
            content = buffer.getvalue()

            with open(file_name, "wb") as f:
                f.write(content)
                os.fsync(f.fileno())

        else:
            try:
                scripted_module = torch.jit.script(child_module)
                scripted_module.save(file_name)
            except Exception as e:
                raise RuntimeError(
                    "The model cannot be processed for distributed training due to compatibility or security "
                    "constraints. Support for additional model types and improved security handling will be included "
                    "in future updates."
                )

        module_info = str(child_module)
        offloaded_module = OffloadedModule(self, module_info, worker_id, module_hash)

        # Detach and clear the parameters of the child_module to free memory
        # for name, param in child_module.named_parameters():
        #     # Create the DistributedParameter.
        #     module_hash = hash(child_module)
        #     distributed_param = DistributedParameter(child_module, module_hash, worker_id, name)
        #
        #     # Use a safe name for attribute registration.
        #     safe_name = name.replace('.', '_')
        #     setattr(child_module, safe_name, distributed_param)
        #
        #     # Replace the parameter in _parameters.
        #     child_module._parameters[safe_name] = distributed_param
        #     offloaded_module.add_distributed_parameter(safe_name, distributed_param)
        #
        #     # Remove the original parameter using its original name.
        #     if name in child_module._parameters:
        #         del child_module._parameters[name]

        for param in child_module.parameters():
            param.detach_()  # Detach from the computation graph
            param.requires_grad = False  # Ensure no gradients are computed
            param.data = torch.empty(0)  # Clear the data in the parameter

        # Optional: Clear any buffers (like batch norm running stats) in the child module
        for buffer_name, buffer in child_module.named_buffers(recurse=False):
            setattr(child_module, buffer_name, torch.empty(0))

        # Clear the module's parameters explicitly
        del child_module
        gc.collect()  # Force garbage collection

        # Spawn a worker thread for the offloaded module
        offloaded_module.spawn_worker(file_name)

        if parent_module is not None:
            # Update the parent module's child module with the offloaded module
            if isinstance(parent_module, nn.ModuleList):
                parent_module[module_id[-1]] = offloaded_module
            else:
                child_name = list(parent_module.named_children())[module_id[-1]][0]
                setattr(parent_module, child_name, offloaded_module)
        else:
            setattr(self, "model", offloaded_module)

    def _wrap_hf_module(self, module_id: str, module_info: dict):
        """Handle single module offloading"""
        # Get worker and their assigned modules
        worker_id = module_info["assigned_workers"][0]
        file_name = f"{module_id}_{worker_id}.pt"
        module_path = module_info.get("module_path")
        module_class = module_info.get("module")

        offloaded_module = OffloadedModule(self, module_class, worker_id, module_id)

        with open(file_name, "wb") as f:
            f.close()

        # Spawn a worker thread for the offloaded module
        offloaded_module.spawn_worker(file_name, module_info)

        if not module_path or module_path in ("model", ""):
            self.model = offloaded_module
            offloaded_module.entire_model = True
            return

        module_path_list = module_path.split(".")
        target = self.model

        if module_path_list[0] == "model" and not hasattr(self.model, "model"):
            module_path_list.pop(0)

        for attr in module_path_list[:-1]:
            target = getattr(target, attr)

        setattr(target, module_path_list[-1], offloaded_module)

    def _wrap_grouped_layers(self, grouped_layers: dict):
        """
        Replace a ModuleList loop with direct calls to offloaded worker(s).
        This handles the case where layers 0-11 might be on one worker,
        and we need to replace the parent's loop with a single call.
        """
        # Gather ordered calls to OffloadedModules to replace the existing loop in the forward pass
        offloaded_modules = []

        for module_id, module_info in sorted(
            grouped_layers.items(), key=lambda x: x[1]["layer_range"][0]
        ):
            worker_id = module_info["assigned_workers"][0]
            layer_range = module_info.get("layer_range", [])

            # Create offloaded module wrapper
            module_name = module_info["module"]
            offloaded_module = OffloadedModule(self, module_name, worker_id, module_id)
            offloaded_module.layer_range = layer_range
            offloaded_module.is_layer_group = True

            # Get expected input and output args and add to module_info
            io_signature = get_loop_io_signature(self.model)
            module_info["expected_inputs"] = list(io_signature["all_inputs"])
            module_info["expected_outputs"] = list(io_signature["all_outputs"])
            module_info["loop_body_source"] = io_signature["loop_body_source"]
            module_info["loop_iterator_name"] = io_signature["loop_iterator_name"]
            module_info["module_path"] = io_signature["module_path"]

            file_name = f"{module_id}_{worker_id}.pt"
            with open(file_name, "wb") as f:
                f.close()

            offloaded_module.spawn_worker(file_name, module_info)
            offloaded_modules.append(offloaded_module)

        self._inject_grouped_layer_forward(grouped_layers, offloaded_modules)

    def _inject_grouped_layer_forward(
        self,
        grouped_layers: Dict,
        offloaded_modules: List["OffloadedModule"],
    ):
        """
        Modify the parent module's forward method to call the offloaded
        layer group instead of looping through individual layers.
        """
        parent_path = list(grouped_layers.values())[0].get("parent_module_path", "")

        assert isinstance(self.model, nn.Module), "Invalid model type"
        parent_module = get_nested_module(self.model, parent_path)

        parent_module.offloaded_modules = offloaded_modules

        new_forward = generate_new_forward_method(parent_module, offloaded_modules)

        parent_module.forward = types.MethodType(new_forward, parent_module)

    def send_request(self, request_type, args):
        """
        Sends a request to the roles and waits for the response.
        """
        request = {"type": request_type, "args": args}
        try:
            self.mpc_lock.acquire()
            self.node_requests.put(request)
            response = self.node_responses.get()  # Blocking call, waits for response

        except Exception as e:
            print(f"Error sending request: {e}")
            response = {"error": str(e)}

        finally:
            self.mpc_lock.release()

        return response["return"]

    def _initialize_distribution(self):
        """Initialize the distributed model."""
        if self.optimizer is None:
            optimizer_cls = torch.optim.Adam
        else:
            optimizer_cls = self.optimizer

        optimizer_spec = optimizer_to_spec(optimizer_cls)

        # Create the distribution on our end if we have the model loaded
        distribution = {
            "model_name": self.model_name,
            "training": self.training,
            "optimizer": optimizer_spec,
        }

        # Request job from network
        distributed_config = self.node.send_request(
            "request_job",
            (self.n_pipelines, 1, distribution, self.training),
            timeout=200,
        )

        if not distributed_config:
            raise RuntimeError("Could not obtain job from network... Please try again.")

        # Distribute the model according to the configuration
        self.distribute_model(distributed_config)

        optimizer_type = get_optimizer_from_spec(optimizer_spec)

        # Create an optimizer creation function
        self.create_optimizer = lambda **kwargs: create_distributed_optimizer(
            self, optimizer_type, **kwargs
        )

    def _load_model_skeleton(self, model_type: str = "chat"):
        """Load the HF model structure with empty weights"""
        with init_empty_weights():
            if model_type in ("causal", "chat"):
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            elif model_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            elif model_type == "vision2text":
                self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)
            elif model_type == "audio2text":
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)
            else:
                model_config = AutoConfig.from_pretrained(self.model_name)
                self.model = AutoModel.from_config(model_config)

    def _load_host_modules(self, host_modules: Dict[str, Dict[str, Any]]):
        """
        Load weights for modules that will be hosted locally
        """
        logging.info(f"Loading {len(host_modules)} host modules")

        for module_id, module_info in host_modules.items():
            try:
                self._load_single_host_module(module_id, module_info)
            except Exception as e:
                logging.error(f"Failed to load host module {module_id}: {e}")
                raise

    def _load_single_host_module(self, module_id: str, module_info: Dict[str, Any]):
        """
        Load weights for a single host module
        """
        module_path = module_info.get("module_path")
        module_class = module_info.get("module")

        if not module_path:
            logging.warning(f"No module_path for host module {module_id}, skipping")
            return

        logging.info(f"Loading host module: {module_class} at {module_path}")

        # Navigate to the target module in the skeleton
        module_path_list = module_path.split(".")
        target = self.model

        # Handle 'model' prefix
        if module_path_list[0] == "model" and not hasattr(self.model, "model"):
            module_path_list.pop(0)

        # Navigate to parent
        for attr in module_path_list[:-1]:
            target = getattr(target, attr)

        # Get the actual module
        module_name = module_path_list[-1]
        host_module = getattr(target, module_name)

        # Load weights for this module
        logging.info(f"Loading weights for {module_path}")
        state_dict = self._load_module_weights(self.model_name, [module_path])

        # Convert module to empty weights on CPU first
        host_module = host_module.to_empty(device="cpu")

        # Load the state dict
        missing_keys, unexpected_keys = host_module.load_state_dict(
            state_dict, strict=False
        )

        if missing_keys:
            logging.warning(f"Host module {module_path} - Missing keys: {missing_keys}")
        if unexpected_keys:
            logging.warning(
                f"Host module {module_path} - Unexpected keys: {unexpected_keys}"
            )

        # Move to device
        host_module = host_module.to(self.device)

        # Update the module in place
        setattr(target, module_name, host_module)

        logging.info(f"Successfully loaded host module {module_class}")

    def _load_module_weights(
        self, model_name: str, module_paths: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Load weights for specific modules from HuggingFace.
        Similar to worker's _load_specific_layer_weights but adapted for user side.
        """
        state_dict = {}

        try:
            # Download model files
            logging.info(f"Downloading weights for {model_name}")
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=self.hf_cache_dir,
                allow_patterns=["*.safetensors", "*.bin"],
                local_files_only=False,
            )

            # Find safetensors files
            safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))

            if safetensor_files:
                logging.info(f"Found {len(safetensor_files)} safetensors files")

                for shard_path in safetensor_files:
                    with safe_open(shard_path, framework="pt", device="cpu") as f:
                        keys_loaded = 0
                        for key in f.keys():
                            # Check if key matches any module path
                            for module_path in module_paths:
                                prefix = module_path + "."
                                if key.startswith(prefix):
                                    # Remove the module path prefix
                                    new_key = key[len(prefix) :]
                                    state_dict[new_key] = f.get_tensor(key)
                                    keys_loaded += 1
                                    break

                        if keys_loaded > 0:
                            logging.info(
                                f"Loaded {keys_loaded} tensors from "
                                f"{os.path.basename(shard_path)}"
                            )
            else:
                # Fallback to .bin files
                logging.info("No safetensors found, trying .bin files")
                bin_files = glob.glob(os.path.join(model_path, "pytorch_model*.bin"))

                if bin_files:
                    for bin_path in bin_files:
                        shard_dict = torch.load(bin_path, map_location="cpu")

                        for key, value in shard_dict.items():
                            for module_path in module_paths:
                                prefix = module_path + "."
                                if key.startswith(prefix):
                                    new_key = key[len(prefix) :]
                                    state_dict[new_key] = value
                                    break
                else:
                    raise ValueError(f"No weight files found in {model_path}")

            logging.info(f"Loaded {len(state_dict)} weight tensors for host modules")

        except Exception as e:
            logging.error(f"Error loading module weights: {e}")
            raise

        return state_dict


class OffloadedModule(nn.Module):
    """
    OffloadedModule:
        A module wrapper that handles offloading on the master roles without disrupting normal
        pytorch model flow.
    """

    def __init__(
        self,
        parent_model: DistributedModel,
        module_name: str,
        worker_id,
        module_id: str,
    ):
        super(OffloadedModule, self).__init__()

        self.entire_model = False
        self.module_name = module_name.split("(")[0]

        self.parent_model = parent_model
        self.worker_id = worker_id
        self.module_id = module_id
        self.n_batch = 0

        self.is_layer_group = False
        self.layer_range = None

    def children(self):
        # Print the offloaded module and the original model name
        # print(f"OffloadedModule: {self.module_name}")
        # Return an empty iterator to hide deeper children
        return iter([])

    def spawn_worker(self, name: str, module_info: dict):
        # # Initialize a threading Timer to monitor the loading process
        # timer = threading.Tier(MAX_WAIT_TIME, self.handle_timeout)
        # timer.start()

        # try:
        # Send the module to the worker roles

        self.parent_model.send_request(
            "send_model", (name, self.worker_id, self.module_id, module_info)
        )

        # Wait for the module to be loaded on worker
        waiting = True
        start_time = time.time()
        while waiting:
            time.sleep(0.5)
            args = self.parent_model.send_request(
                "check_loaded", (self.worker_id, self.module_id)
            )

            if args is True:
                waiting = False

            if time.time() - start_time >= MAX_WAIT_TIME:
                # Logic here to request another worker take his place
                waiting = False

    def handle_timeout(self):
        # Timeout occurred, switch to another worker TODO
        new_worker_node = self.master_node.select_candidate_worker()
        if new_worker_node:
            self.worker_node = new_worker_node["connection"]
        else:
            # No available workers found, handle the situation accordingly
            pass

    def generate(self, *args, **kwargs):
        args_bytes = tensor_to_bytes(args)
        kwargs_bytes = tensor_to_bytes(kwargs)
        request_bytes = self.module_id.encode() + args_bytes + b"::" + kwargs_bytes
        size, shm_name = store_in_shared_memory(request_bytes, encoded=True)
        self.parent_model.send_request("generate", (self.worker_id, size, shm_name))

        # Wait for response, change to appending waiting thread to list in master
        waiting = True
        start_time = time.time()
        while waiting:
            time.sleep(0.1)
            args = self.parent_model.send_request("check_generate", self.module_id)

            if args is not None:
                waiting = False
                size, name = args
                output_bytes = get_from_shared_memory(size, name)

            if time.time() - start_time >= MAX_WAIT_TIME:
                # Logic here to request another worker take his place
                waiting = False

        output = enable_grad(bytes_to_tensor(output_bytes))
        return output

    def forward(self, *args, **kwargs):
        from tensorlink.ml.utils import handle_output

        start_time = time.time()
        n_batch = self.parent_model.model.n_batch
        n_micro = getattr(self.parent_model._thread_local, "micro", None)

        tag = [n_batch, n_micro, self.module_id]

        # Store the intermediate tensor for backwards pass
        if not self.entire_model:
            self.parent_model.model.intermediates[n_micro].append(
                [handle_output(args), self.module_id]
            )

        args = handle_output(args)
        detached_args = detach_tensor(args, clone=True)
        args_bytes = tensor_to_bytes(detached_args)
        kwargs_bytes = tensor_to_bytes(kwargs)
        forward_bytes = args_bytes + b"|" + kwargs_bytes

        size, shm_name = store_in_shared_memory(forward_bytes, encoded=True)

        # Relay forward pass to next roles
        self.parent_model.send_request(
            "send_forward", (self.worker_id, size, shm_name, tag)
        )

        # Wait for response, change to appending waiting thread to list in master
        waiting = True
        while waiting:
            time.sleep(0.1)
            key = (n_batch, n_micro, self.module_id)
            args = self.parent_model.send_request("check_forward", key)

            if args is not None:
                waiting = False
                size, name = args
                output_bytes = get_from_shared_memory(size, name)

            if time.time() - start_time >= MAX_WAIT_TIME:
                # Logic here to request another worker take his place
                waiting = False

        output = enable_grad(bytes_to_tensor(output_bytes))

        self.parent_model.send_request(
            "release_memory", ("forward_queue", self.module_id, key)
        )

        inter_storage = [
            self.module_id,
            handle_output(output),
        ]  # Store associated output

        # Store intermediates and connection for backwards pass
        self.parent_model.model.intermediates[n_micro].append(inter_storage)

        return output

    def add_distributed_parameter(self, name, distributed_param):
        """Register a DistributedParameter to the offloaded module."""
        self._parameters[name] = distributed_param

    def parameters(self, recurse=True):
        """Override parameters to return wrapped DistributedParameters."""
        return iter(self._parameters.values())

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Returns an empty dictionary for the OffloadedModule.
        This effectively hides the OffloadedModule from the model's state_dict.
        """
        return {}

    def __repr__(self):
        # Custom representation to prevent recursion when printing or debugging.
        return f"OffloadedModule(name={self.module_name}, worker_id={self.worker_id})"
