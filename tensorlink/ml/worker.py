from tensorlink.ml.utils import (
    bytes_to_tensor,
    tensor_to_bytes,
    detach_tensor,
    attach_tensor,
    handle_output,
    enable_grad,
    get_optimizer_from_spec,
)
from tensorlink.ml.utils.loading import (
    TiedLinear,
    load_full_model,
    load_model_skeleton,
    ModelCacheManager,
    load_module_weights,
    load_grouped_module_weights,
    get_nested_module,
)
from tensorlink.ml.utils.injector import LayerGroupModule
from tensorlink.nodes.shared_memory import (
    get_from_shared_memory,
    store_in_shared_memory,
)

import gc
import inspect
import json
import logging
import os
import pickle
import time
from threading import Thread
import torch
import torch.amp as amp
from typing import List, Dict, Any
from transformers.generation.streamers import BaseStreamer


def _create_layer_group_wrapper(
    base_model: torch.nn.Module,
    layer_paths: List[str],
    expected_inputs: List[str],
    expected_outputs: List[str],
    loop_body_source: str,
    loop_structure: Dict,
) -> torch.nn.Module:
    """
    Create a wrapper module that processes multiple layers sequentially.
    This allows the worker to process all layers in one forward pass.
    """
    # Extract the actual layer modules
    layers = [get_nested_module(base_model, path) for path in layer_paths]

    # Create and return the wrapper
    return LayerGroupModule(
        layers=layers,
        parent_module=base_model,
        input_vars=expected_inputs,
        output_vars=expected_outputs,
        loop_body_source=loop_body_source,
        loop_structure=loop_structure,
    )


class TensorlinkWorkerStreamer(BaseStreamer):
    def __init__(self, send_token, send_end):
        super().__init__()
        self.send_token = send_token
        self.send_end = send_end

    def put(self, value):
        if not isinstance(value, torch.Tensor):
            return

        # Normalize shape
        if value.dim() == 2:
            token_id = value[0, -1].item()
        elif value.dim() == 1:
            token_id = value[-1].item()
        else:
            token_id = value.item()

        self.send_token(token_id)

    def end(self):
        self.send_end()


class DistributedWorker:
    def __init__(self, node, trusted=False):
        self.node = node
        self.node_requests = node.node_requests
        self.node_responses = node.node_responses
        self.mpc_lock = node.mpc_lock
        self.cache_manager = ModelCacheManager(
            "./tmp/snapshots",
        )

        self.modules = {}
        self.optimizers = {}
        self.terminate = False
        self.trusted = trusted

        # CUDA optimization: Check device and optimize CUDA settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_cuda_environment()

        # Mixed precision setup vars (set on model load)
        self.scaler = None
        self.use_amp = False

        self.GC_CHECK_INTERVAL = 2_000
        self.CHECK_COUNTER = 1

        # Initialize CUDA streams for overlapping operations
        if self.device.type == "cuda":
            self.default_stream = torch.cuda.Stream()
            self.compute_stream = torch.cuda.Stream()
            self.memory_stream = torch.cuda.Stream()

    def cleanup_memory(self):
        """Aggressively clean up memory"""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def setup_cuda_environment(self):
        """Configure optimal CUDA settings for ML workloads"""
        if self.device.type == "cuda":
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

            # Set memory allocation strategy
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory

            # Log CUDA configuration
            self.send_request(
                "debug_print",
                (
                    "DistributedWorker -> "
                    f"CUDA device: {torch.cuda.get_device_name(0)}\n"
                    f"CUDA capability: {torch.cuda.get_device_capability(0)}\n"
                    f"Total CUDA memory: {total_memory / 1e9:.2f}GB\n",
                    "blue",
                    logging.DEBUG,
                ),
            )

    def send_request(self, request_type, args, timeout=None):
        """Send request to coordinator node with timeout handling"""
        request = {"type": request_type, "args": args}
        try:
            # print(f"MPC Locked: {self.node.__class__.__name__}: {request_type}")
            self.mpc_lock.acquire(timeout=timeout)
            self.node_requests.put(request)
            response = self.node_responses.get(
                timeout=timeout
            )  # Blocking call, waits for response
            if isinstance(response, dict):
                return response["return"]
            else:
                return response

        except TimeoutError:
            self.terminate = True
        except Exception as e:
            return {"return": str(e)}
        finally:
            self.mpc_lock.release()

    def _handle_backward(self, module_id, tag, loss_relay):
        """Handle backward pass with mixed precision support"""
        module = self.modules[module_id]
        n_batch = module.n_batch
        next_node = module.host

        # Only process if in training mode
        if module.training:
            # Get tensor from shared memory
            tensor_bytes = get_from_shared_memory(
                loss_relay[0], loss_relay[1], encoded=True
            )
            tensor = bytes_to_tensor(tensor_bytes)

            # Move tensors to device
            loss = attach_tensor(tensor, self.device)

            # Retrieve intermediate values from storage
            inter_tag = tuple(tag)
            record = module.intermediates.pop(inter_tag)
            assoc_input = record["inputs"]
            assoc_output = record["output"]

            if isinstance(assoc_input, (tuple, list)):
                assoc_input = tuple(x.to(self.device) for x in assoc_input)
            else:
                assoc_input = assoc_input.to(self.device)

            assoc_output = assoc_output.to(self.device)

            # Backward pass
            if self.use_amp:
                # Scale loss for mixed precision
                scaled_loss = self.scaler.scale(loss)
                assoc_output.backward(scaled_loss)
                # Unscale gradients for optimizer
                self.scaler.unscale_(self.optimizers.get(module_id, None))
            else:
                assoc_output.backward(loss)

            # Detach gradients and prepare for next node
            def extract_grad(x):
                if x.grad is None:
                    return torch.zeros_like(x, dtype=torch.float32)
                return x.grad

            if isinstance(assoc_input, (tuple, list)):
                grads = tuple(extract_grad(x) for x in assoc_input)
                dvalues = detach_tensor(grads)
            else:
                dvalues = detach_tensor(extract_grad(assoc_input))

            # Clean up to avoid memory leaks
            del assoc_input, assoc_output

            # Store pass in shared memory and send to next node
            dvalues_bytes = tensor_to_bytes(dvalues)
            size, name = store_in_shared_memory(dvalues_bytes, encoded=True)
            self.send_request("send_backward", (next_node, size, name, tag))

            # Strategic memory management - clear only when necessary
            if self.device.type == "cuda" and n_batch % 10 == 0:
                torch.cuda.empty_cache()

    def _handle_forward(self, module_id, key, size, name):
        """Handle forward pass with proper KV cache structure preservation"""
        module = self.modules[module_id]

        # Get data from shared memory
        data = get_from_shared_memory(size, name, encoded=True)
        args_len = int.from_bytes(data[:8], "big")
        args_bytes = data[8 : 8 + args_len]
        kwargs_bytes = data[8 + args_len :]

        args = bytes_to_tensor(args_bytes)
        kwargs = bytes_to_tensor(kwargs_bytes)

        inp = attach_tensor(args, self.device)
        kwargs = attach_tensor(kwargs, self.device)

        if module.training:
            inp = enable_grad(inp)
            kwargs = enable_grad(kwargs)

        if not isinstance(inp, (list, tuple)):
            inp = (inp,)

        # Forward pass
        if self.use_amp and module.training:
            with amp.autocast():
                out = module(*inp, **kwargs)
        else:
            with torch.set_grad_enabled(module.training):
                # we only use kwargs if this is a layer group module
                if hasattr(module, "num_layers"):
                    out = module(**kwargs)
                else:
                    out = module(*inp, **kwargs)

        # Store intermediate results if training
        if module.training:
            module.intermediates[key] = {
                "inputs": inp,
                "output": handle_output(out),
            }

        # Detach and store output
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        detached_out = detach_tensor(out)

        output_bytes = tensor_to_bytes(detached_out)
        size, name = store_in_shared_memory(output_bytes)

        self.send_request("send_forward", (module.host, module_id, size, name, key))

        # Incremental training counter
        if module.training:
            module.n_batch += 1

        # Strategic memory management
        if self.device.type == "cuda" and module.n_batch % 20 == 0:
            torch.cuda.empty_cache()

    def _handle_generate(self, module_id, size, name, stream):
        """
        Optimized text generation with optional stream generation. Called upon only
        if we have a full model loaded and not a submodule.
        """
        module = self.modules[module_id]
        payload = get_from_shared_memory(size, name, encoded=True)

        # Deserialize
        args_bytes, kwargs_bytes = payload.split(b"::")
        args = bytes_to_tensor(args_bytes)
        kwargs = bytes_to_tensor(kwargs_bytes)

        # Attach input_ids from args if missing
        if "input_ids" not in kwargs:
            if args is None:
                raise ValueError("generate() missing input_ids (no args, no kwargs)")
            kwargs["input_ids"] = args

        # Validate input_ids
        if not isinstance(kwargs["input_ids"], torch.Tensor):
            try:
                kwargs["input_ids"] = torch.Tensor(kwargs["input_ids"][0])
            except:
                raise ValueError("input_ids must be convertible to torch.Tensor")

        if kwargs["input_ids"].numel() == 0 or kwargs["input_ids"].shape[-1] == 0:
            raise ValueError("input_ids is empty; cannot generate")

        # Move everything to device
        kwargs = attach_tensor(kwargs, self.device)

        # CUDA defaults (non-invasive)
        if self.device.type == "cuda":
            kwargs.setdefault("use_cache", True)

        host_id = module.host

        try:
            if not stream:
                with torch.no_grad():
                    output = module.generate(**kwargs)

                output_bytes = tensor_to_bytes(detach_tensor(output))

                size, name = store_in_shared_memory(output_bytes)
                self.send_request(
                    "send_forward", (host_id, module_id, size, name, "generate")
                )

            else:
                streamer = TensorlinkWorkerStreamer(
                    send_token=lambda token: self._send_token(
                        module_id, token, host_id
                    ),
                    send_end=lambda: self._send_stream_end(module_id, host_id),
                )

                kwargs["streamer"] = streamer

                def _run_generate():
                    with torch.no_grad():
                        module.generate(**kwargs)

                gen_thread = Thread(target=_run_generate, daemon=True)
                gen_thread.start()
                gen_thread.join()
                return

        except Exception as e:
            output_bytes = json.dumps({"error": str(e)}).encode()
            if stream:
                self._send_stream_end(module_id, host_id)

        size, name = store_in_shared_memory(output_bytes)
        self.send_request("send_forward", (host_id, module_id, size, name, "generate"))

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _send_token(self, module_id, token, host_id):
        if isinstance(token, torch.Tensor):
            token = token.item()

        self.send_request("send_token", (module_id, token, host_id))

    def _send_stream_end(self, module_id, host_id):
        self.send_request("send_stream_end", (module_id, host_id))

    def load_module(self, module_info: dict):
        """
        Load and prepare model from file or directly from HuggingFace.

        For direct HuggingFace loading without a file, just provide module_name.
        Default parameters allow for simplified calling when loading generic models.
        """
        module_id = module_info.get("module_id")
        model_name = module_info.get("name")
        module_name = module_info.get("module_name")
        training = module_info.get("training", False)
        our_id = module_info.get("assigned_workers")[0]
        file_name = module_id + our_id

        if module_id is None:
            raise ValueError("For standard loading, module_id must be provided")

        # Clear memory before loading
        self.cleanup_memory()

        # Try to load the module based on trusted status
        if self.trusted:
            with open(file_name, "rb") as f:
                module = pickle.load(f)
                module = module.to(self.device)

        # Else try Hugging Face for model info
        else:
            skeleton_module = load_model_skeleton(model_name)
            module = self._initialize_module_from_config(
                skeleton_module, model_name, module_name, module_info
            )

            # Ensure skeleton cleanup
            del skeleton_module
            self.cleanup_memory()

        # Cleanup file
        try:
            os.remove(file_name)
        except:
            pass

        # Initialize storage structures
        module.intermediates = {}
        module.host = module_info.get('host')
        module.n_batch = 0

        self.modules[module_id] = module
        if training:
            optimizer_cls = get_optimizer_from_spec(module_info["optimizer_spec"])
            self.optimizers[module_id] = optimizer_cls

        self.send_request("module_loaded", module_id)

    def _initialize_module_from_config(
        self,
        skeleton_module: torch.nn.Module,
        model_name: str,
        module_name: str,
        module_info: Dict[str, Any],
    ) -> torch.nn.Module:
        """
        Load model or specific layers from HuggingFace.
        Handles both single modules and grouped layer ranges.
        """
        try:
            # Determine if this is a grouped layer load
            module_type = module_info.get('type', 'offloaded')
            module_id = module_info.get("module_id")

            if module_type == 'offloaded_group':
                # Load grouped layers
                return self._load_grouped_layers(
                    model_name, skeleton_module, module_id, module_info
                )
            else:
                # Load single module
                return self._load_single_module(
                    model_name, skeleton_module, module_info
                )

        except Exception as e:
            # Make sure skeleton is cleaned up on error
            err = f"Failed to load model from HuggingFace: {str(e)}"
            self.send_request(
                "debug_print",
                (
                    err,
                    "red",
                    logging.ERROR,
                ),
            )
            del skeleton_module
            self.cleanup_memory()

            logging.error(err)
            raise ValueError(err)

    def _load_grouped_layers(
        self,
        model_name: str,
        base_model: torch.nn.Module,
        module_id: str,
        module_info: Dict[str, Any],
    ) -> torch.nn.Module:
        """
        Load a group of layers as a single module. Uses empty weights initialization
        and only loads required layer weights.
        """
        layer_paths = module_info.get("layer_paths", [])
        expected_inputs = module_info.get("expected_inputs", [])
        expected_outputs = module_info.get("expected_outputs", [])
        loop_body_source = module_info.get("loop_body_source")
        loop_structure = module_info.get("loop_structure")

        if not layer_paths:
            raise ValueError("layer_paths must be provided for grouped layer loading")

        grouped_module = _create_layer_group_wrapper(
            base_model,
            layer_paths,
            expected_inputs,
            expected_outputs,
            loop_body_source,
            loop_structure,
        )

        del base_model
        self.cleanup_memory()

        model_path = self.cache_manager.load_model_path(model_name)

        # resolves each layer's prefix, remaps to layers.<idx>, applies
        # buffers, and moves to device (layer by layer to avoid OOM).
        load_grouped_module_weights(
            model_path=model_path,
            layer_paths=layer_paths,
            target_module=grouped_module,
            module_info=module_info,
            device=self.device,
        )

        return grouped_module

    def _debug_move_to_device(
        self, module: torch.nn.Module, device: torch.device
    ) -> torch.nn.Module:
        """
        Move module to device with granular tracking to identify OOM source.
        Moves each layer individually and tracks memory after each transfer.
        """
        self.send_request(
            "debug_print",
            (
                f"DistributedWorker -> Starting gradual move to {device}...",
                "yellow",
                logging.DEBUG,
            ),
        )

        # Move each layer individually
        for idx, layer in enumerate(module.layers):
            self.send_request(
                "debug_print",
                (
                    f"Moving layer {idx + 1}/{len(module.layers)} to {device}...",
                    "yellow",
                    logging.DEBUG,
                ),
            )

            # Track layer size
            layer_params = sum(p.numel() * p.element_size() for p in layer.parameters())
            self.send_request(
                "debug_print",
                (
                    f"  Layer {idx} size: {layer_params / 1024 ** 3:.3f}GB",
                    "cyan",
                    logging.DEBUG,
                ),
            )

            # Memory before
            self._debug_memory_state(f"Before layer {idx}")

            try:
                module.layers[idx] = layer.to(device)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()  # Ensure transfer completes

                # Memory after
                self._debug_memory_state(f"After layer {idx}")

            except RuntimeError as e:
                self.send_request(
                    "debug_print",
                    (
                        f"OOM at layer {idx}/{len(module.layers)}: {str(e)}",
                        "red",
                        logging.ERROR,
                    ),
                )
                raise

        # Move any remaining module components (non-layer parameters)
        self.send_request(
            "debug_print",
            (
                "Moving remaining module components...",
                "yellow",
                logging.DEBUG,
            ),
        )

        for name, param in module.named_parameters():
            if not name.startswith('layers.'):
                self.send_request(
                    "debug_print",
                    (
                        f"  Moving param: {name}, shape={list(param.shape)}, "
                        f"size={param.numel() * param.element_size() / 1024 ** 2:.2f}MB",
                        "cyan",
                        logging.DEBUG,
                    ),
                )

        return module

    def _debug_memory_state(self, label: str) -> None:
        """Print current GPU memory state."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3

            self.send_request(
                "debug_print",
                (
                    f"GPU Memory [{label}]: "
                    f"Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, "
                    f"Max={max_allocated:.2f}GB",
                    "cyan",
                    logging.DEBUG,
                ),
            )

    def _debug_module_state(self, module: torch.nn.Module, label: str) -> None:
        """Print module parameter statistics."""
        total_params = 0
        total_bytes = 0
        params_by_dtype = {}

        for name, param in module.named_parameters():
            num_params = param.numel()
            num_bytes = num_params * param.element_size()
            total_params += num_params
            total_bytes += num_bytes

            dtype_str = str(param.dtype)
            if dtype_str not in params_by_dtype:
                params_by_dtype[dtype_str] = {'params': 0, 'bytes': 0}
            params_by_dtype[dtype_str]['params'] += num_params
            params_by_dtype[dtype_str]['bytes'] += num_bytes

        self.send_request(
            "debug_print",
            (
                f"Module [{label}]: "
                f"Total params={total_params:,}, Size={total_bytes / 1024 ** 3:.2f}GB",
                "cyan",
                logging.DEBUG,
            ),
        )

        for dtype, stats in params_by_dtype.items():
            self.send_request(
                "debug_print",
                (
                    f"  {dtype}: {stats['params']:,} params, {stats['bytes'] / 1024 ** 3:.2f}GB",
                    "cyan",
                    logging.DEBUG,
                ),
            )

    def _load_single_module(
        self, model_name: str, base_model: torch.nn.Module, module_info: Dict[str, Any]
    ) -> torch.nn.Module:
        """
        Load a single module (e.g., just the RMSNorm layer).
        Uses empty weights initialization and only loads required module weights.
        """
        module_path = module_info.get("module_path", "")
        module_class_name = module_info.get("module", "")

        if not module_path or module_path == "model":
            del base_model
            self.cleanup_memory()
            return self._load_full_model(model_name, module_info)

        # Otherwise proceed with submodule extraction
        tied_to = module_info.get("tied_to", "")
        effective_path = tied_to or module_path
        target_module = get_nested_module(base_model, effective_path)

        del base_model
        self.cleanup_memory()

        model_path = self.cache_manager.load_model_path(model_name)

        load_module_weights(
            model_path=model_path,
            module_path=effective_path,
            target_module=target_module,
            module_info=module_info,
            device=self.device,
            log_fn=lambda x: self.send_request(
                "debug_print",
                (
                    x,
                    "cyan",
                    logging.DEBUG,
                ),
            ),
            warn_fn=lambda x: self.send_request(
                "debug_print",
                (
                    x,
                    "red",
                    logging.WARNING,
                ),
            ),
        )

        if tied_to and module_class_name == "Linear":
            return TiedLinear(target_module.weight)

        return target_module

    def _load_full_model(self, model_name: str, module_info: dict) -> torch.nn.Module:
        """
        Load a complete model from HuggingFace with optimal memory usage.
        Uses HF's native loading which is more memory-efficient than manual skeleton+weights.
        """
        model_type = module_info.get('model_type', 'chat')
        self.cleanup_memory()
        return load_full_model(model_name, model_type, self.device)

    def process_state_update(self, module_id, state_update):
        """Process optimizer state updates"""
        module = self.modules[module_id]

        if state_update[0] == "init":
            optimizer_kwargs = state_update[1]
            optimizer_name = self.optimizers[module_id].__name__

            # Configure optimizer with mixed precision support
            if self.use_amp and 'fused' not in optimizer_name.lower():
                # Use fused implementation when available for better performance
                if optimizer_name.lower() == 'adam':
                    try:
                        from torch.optim.adam import Adam

                        self.optimizers[module_id] = Adam(
                            module.parameters(), **optimizer_kwargs, fused=True
                        )
                    except:
                        self.optimizers[module_id] = self.optimizers[module_id](
                            module.parameters(), **optimizer_kwargs
                        )
                else:
                    self.optimizers[module_id] = self.optimizers[module_id](
                        module.parameters(), **optimizer_kwargs
                    )
            else:
                self.optimizers[module_id] = self.optimizers[module_id](
                    module.parameters(), **optimizer_kwargs
                )

            self.send_request(
                "debug_print",
                (
                    f"DistributedWorker -> Initialized optimizer: {optimizer_name} on {self.device.type}",
                    "bright_blue",
                    logging.DEBUG,
                ),
            )
            self.send_request("optimizer_response", (module_id, "loaded"))

        elif state_update[0] == "step":
            closure = state_update[1]
            # Step optimizer with mixed precision support if using CUDA
            if self.use_amp:
                # Update with scaler for mixed precision
                self.scaler.step(self.optimizers[module_id], closure)
                self.scaler.update()
            else:
                self.optimizers[module_id].step(closure)

            self.send_request(
                "debug_print",
                (
                    "DistributedWorker -> Optimizer stepped.",
                    "bright_blue",
                    logging.DEBUG,
                ),
            )
            self.send_request("optimizer_response", (module_id, "stepped"))

        elif state_update[0] == "zero_grad":
            # Zero gradients with optimized memory usage
            if self.device.type == "cuda":
                # More efficient for CUDA
                for param in module.parameters():
                    if param.grad is not None:
                        param.grad = None
            else:
                self.optimizers[module_id].zero_grad()

            self.send_request(
                "debug_print",
                (
                    "DistributedWorker -> Optimizer zeroed.",
                    "bright_blue",
                    logging.DEBUG,
                ),
            )
            self.send_request("optimizer_response", (module_id, "zeroed"))

    def main_loop(self):
        """Main execution loop. Sequentially executes the following tasks: check for new jobs, check for incoming data
        or model update requests, and then processes any outstanding forwards or backwards passes on the loaded modules
        """
        # Check for new modules to load
        args = self.send_request("check_module", None)

        # If we have received model info now load the model in this process
        if isinstance(args, dict):
            self.load_module(args)

        # For workers that have received model info, now load the model in this process
        elif isinstance(args, str):
            if args in self.modules:
                if self.modules[args].training:
                    del self.optimizers[args]
                del self.modules[args]

                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                self.send_request("debug_print", (f"Module {args} removed.",))

        if self.CHECK_COUNTER % self.GC_CHECK_INTERVAL == 0:
            # Check for termination request
            shutdown_signal = self.send_request("check_shutdown", None)
            if shutdown_signal:
                self.send_request(
                    "debug_print",
                    "Termination signal received. Shutting down DistributedWorker process...",
                )
                self.terminate = True

        # Process each module sequentially
        if self.modules:
            for module_id in list(self.modules.keys()):
                module = self.modules[module_id]

                # Check if module is in training mode
                is_training = self.send_request("check_train", module_id)
                if isinstance(is_training, bool):
                    module.training = is_training

                # Check for parameters requests
                params_req = self.send_request("check_parameters_request", module_id)
                if params_req:
                    self.send_request(
                        "debug_print", ("DistributedWorker -> Sending parameters.",)
                    )
                    # Save state dict to file
                    with open(f"parameters_{module_id}", "wb") as file:
                        # Optimize CPU transfer if needed
                        if self.device.type == "cuda":
                            # Temporarily move to CPU for saving
                            cpu_state_dict = {
                                k: v.detach().cpu()
                                for k, v in module.state_dict().items()
                            }
                            torch.save(cpu_state_dict, file)
                        else:
                            torch.save(module.state_dict(), file)

                    self.send_request("send_parameters", (module.host, module_id))

                # Handle state updates
                state_update = self.send_request("check_state_update", module_id)
                if state_update:
                    self.process_state_update(module_id, state_update)

                # Handle forward queue
                forward_task = self.send_request("check_forward", module_id)
                if forward_task:
                    key, args = forward_task
                    if len(args) == 3:
                        # len(arg) of 3 includes stream arg for generate requests
                        size, name, stream = args
                        self._handle_generate(module_id, size, name, stream)
                    else:
                        # len(arg) of 2 for basic forward requests
                        size, name = args
                        self._handle_forward(module_id, key, size, name)

                # Handle backward queue
                backward_task = self.send_request("check_backward", module_id)
                if backward_task:
                    tag, loss_relay = backward_task
                    self._handle_backward(module_id, tag, loss_relay)

        self.CHECK_COUNTER += 1

    def run(self):
        """Main execution thread"""
        while not self.terminate:
            self.main_loop()

            # Small sleep to prevent CPU hogging
            time.sleep(0.001)

        # Final cleanup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
