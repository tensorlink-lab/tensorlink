import gc
import inspect
import json
import logging
import os
import pickle
import time
import glob

from threading import Thread
import torch
import torch.amp as amp
from accelerate import init_empty_weights
from typing import Optional, List, Dict, Any
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    Cache,
)
from transformers.generation.streamers import BaseStreamer
from safetensors import safe_open
from huggingface_hub import snapshot_download

from tensorlink.ml.utils import (
    bytes_to_tensor,
    tensor_to_bytes,
    detach_tensor,
    attach_tensor,
    handle_output,
    enable_grad,
    get_nested_module,
    get_optimizer_from_spec,
)
from tensorlink.ml.injector import LayerGroupModule
from tensorlink.nodes.shared_memory import (
    get_from_shared_memory,
    store_in_shared_memory,
)


def _find_module_path_by_class(
    model: torch.nn.Module, class_name: str
) -> Optional[str]:
    """
    Search the model for the first submodule whose class name matches class_name.
    Returns the module path as returned by named_modules (empty string for root).
    """
    if not class_name:
        return None

    for name, mod in model.named_children():
        # skip the root empty name if it is the same class as requested
        if name == "":
            # if root has the requested class, return empty string
            if mod.__class__.__name__ == class_name:
                return name
            continue

        if name == "model":
            if hasattr(model, "model"):
                return "model." + _find_module_path_by_class(model.model, class_name)
            else:
                return _find_module_path_by_class(model.model, class_name)

        if mod.__class__.__name__ == class_name:
            return name

    return None


def _create_layer_group_wrapper(
    base_model: torch.nn.Module,
    layer_paths: List[str],
    expected_inputs: List[str],
    expected_outputs: List[str],
    loop_body_source: str,
    loop_iterator_name: str,
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
        input_vars=expected_inputs,
        output_vars=expected_outputs,
        loop_body_source=loop_body_source,
        loop_iterator_name=loop_iterator_name,
    )


def normalize_past_key_values(pkv):
    """
    Ensures past_key_values has shape:
    Tuple[Tuple[Tensor, Tensor], ...]
    """
    if pkv is None:
        return None

    if isinstance(pkv, Cache):
        return pkv

    # unwrap accidental singleton nesting
    while (
        isinstance(pkv, (list, tuple))
        and len(pkv) == 1
        and isinstance(pkv[0], (list, tuple))
    ):
        pkv = pkv[0]

    # enforce tuple-of-tuples
    return tuple(tuple(layer) for layer in pkv)


def _load_model_skeleton(model_name: str, module_id: str, model_type: str = "chat"):
    """Load the HF model structure with empty weights"""
    with init_empty_weights():
        if model_type in ("causal", "chat"):
            skeleton_model = AutoModelForCausalLM.from_pretrained(model_name)
        elif model_type == "seq2seq":
            skeleton_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif model_type == "vision2text":
            skeleton_model = AutoModelForVision2Seq.from_pretrained(model_name)
        elif model_type == "audio2text":
            skeleton_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        else:
            model_config = AutoConfig.from_pretrained(model_name)
            skeleton_model = AutoModel.from_config(model_config)

    skeleton_model.eval()  # Set to eval mode initially

    # Ensure no cached gradients or cached computations
    for param in skeleton_model.parameters():
        param.requires_grad = False

    return skeleton_model


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
        self.storage_path = "./tmp/snapshots"

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

        self.hf_cache_dir = os.environ.get(
            'HF_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
        )

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
        tensor_bytes = get_from_shared_memory(size, name, encoded=True)
        args, kwargs = tensor_bytes.split(b"|")
        args = bytes_to_tensor(args)
        kwargs = bytes_to_tensor(kwargs)

        # Move tensors to device
        inp = enable_grad(attach_tensor(args, self.device))
        kwargs = enable_grad(attach_tensor(kwargs, self.device))

        if not isinstance(inp, (list, tuple)):
            inp = (inp,)

        if "past_key_values" in kwargs:
            kwargs["past_key_values"] = normalize_past_key_values(
                kwargs.get("past_key_values")
            )

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
            if (
                not stream
                or "streamer" not in inspect.signature(module.generate).parameters
            ):
                with torch.no_grad():
                    output = module.generate(**kwargs)
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

            output_bytes = tensor_to_bytes(detach_tensor(output))

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
            skeleton_module = _load_model_skeleton(model_name, module_id)
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

        # # Apply model optimizations
        # if self.device.type == "cuda":
        #     # Try converting to faster kernel implementations when possible
        #     if hasattr(module, 'to_bettertransformer'):
        #         try:
        #             module = module.to_bettertransformer()
        #         except:
        #             pass

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
            del skeleton_module
            self.cleanup_memory()

            logging.error(f"Failed to load model from HuggingFace: {str(e)}")
            raise ValueError(f"Failed to load model from HuggingFace: {str(e)}")

    def _load_grouped_layer_weights(
        self,
        model_name: str,
        layer_paths: list[str],
        target_module: torch.nn.Module,
    ) -> None:
        """
        Load weights for a grouped LayerGroupModule directly into the target
        grouped layer module.
        """
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=self.hf_cache_dir,
            allow_patterns=["*.safetensors", "*.bin"],
            local_files_only=False,
        )

        # Map full HF layer prefix → local index
        layer_prefix_to_local_idx = {
            layer_path: i for i, layer_path in enumerate(layer_paths)
        }

        safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))

        if not safetensor_files:
            raise RuntimeError(
                "No safetensors found; .bin fallback not implemented here"
            )

        loaded_keys = []
        missing_keys = set(target_module.state_dict().keys())

        for shard_idx, shard_path in enumerate(safetensor_files):
            self.send_request(
                "debug_print",
                (
                    f"DistributedWorker -> Loading shard {shard_idx + 1}/{len(safetensor_files)}",
                    "blue",
                    logging.DEBUG,
                ),
            )

            shard_state_dict = {}

            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    for layer_prefix, local_idx in layer_prefix_to_local_idx.items():
                        full_prefix = layer_prefix + "."
                        if not key.startswith(full_prefix):
                            continue

                        # Strip "model.layers.XX."
                        subkey = key[len(full_prefix) :]

                        # Remap to local ModuleList index
                        new_key = f"layers.{local_idx}.{subkey}"

                        shard_state_dict[new_key] = f.get_tensor(key)
                        loaded_keys.append(new_key)
                        missing_keys.discard(new_key)
                        break

            # Load this shard into the model
            if shard_state_dict:
                target_module.load_state_dict(shard_state_dict, strict=False)
                del shard_state_dict
                self.cleanup_memory()

        if missing_keys:
            self.send_request(
                "debug_print",
                (
                    f"DistributedWorker -> Missing keys after loading: {missing_keys}",
                    "yellow",
                    logging.ERROR,
                ),
            )

        self.send_request(
            "debug_print",
            (
                "DistributedWorker -> "
                f"Loaded {len(loaded_keys)} weight tensors across {len(safetensor_files)} shards",
                "blue",
                logging.DEBUG,
            ),
        )

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
        layer_paths = module_info.get('layer_paths', [])
        layer_range = module_info.get('layer_range', [])
        expected_inputs = module_info.get('expected_inputs', [])
        expected_outputs = module_info.get('expected_outputs', [])
        loop_body_source = module_info.get('loop_body_source')
        loop_iterator_name = module_info.get('loop_iterator_name')

        if not layer_paths:
            raise ValueError("layer_paths must be provided for grouped layer loading")

        self.send_request(
            "debug_print",
            (
                f"DistributedWorker -> Loading grouped layers {layer_range[0]}-{layer_range[1]} from {model_name}",
                "blue",
                logging.DEBUG,
            ),
        )

        # Create the layer group wrapper with the skeleton's layers
        # Extract references quickly before cleanup
        grouped_module = _create_layer_group_wrapper(
            base_model,
            layer_paths,
            expected_inputs,
            expected_outputs,
            loop_body_source,
            loop_iterator_name,
        )

        # Aggressively cleanup skeleton immediately after extraction
        del base_model
        self.cleanup_memory()

        # Convert grouped module to empty tensors on CPU to clear any weight references
        grouped_module = grouped_module.to_empty(device="cpu")

        # Now load only the weights for the assigned layers
        self.send_request(
            "debug_print",
            (
                f"DistributedWorker -> Loading weights for layers {layer_range[0]}-{layer_range[1]}",
                "blue",
                logging.DEBUG,
            ),
        )

        self._load_grouped_layer_weights(model_name, layer_paths, grouped_module)

        self.cleanup_memory()

        # Move to device incrementally to avoid memory peaks
        if self.device.type == "cuda":
            # Enable expandable segments to reduce fragmentation
            os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
            torch.cuda.empty_cache()

            # Move parameters one at a time to minimize peak memory
            with torch.no_grad():
                for param_name, param in grouped_module.named_parameters():
                    param.data = param.data.to(self.device)
                    torch.cuda.empty_cache()

                for buffer_name, buffer in grouped_module.named_buffers():
                    buffer.data = buffer.data.to(self.device)
                    torch.cuda.empty_cache()
        else:
            grouped_module = grouped_module.to(self.device)

        self.send_request(
            "debug_print",
            (
                f"DistributedWorker -> Successfully loaded {len(layer_paths)} layers with weights",
                "blue",
                logging.DEBUG,
            ),
        )

        return grouped_module

    def _load_specific_layer_weights(
        self,
        model_name: str,
        layer_paths: List[str],
        single: bool = False,
        base_model_prefix: str = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Load only the weights for specific layers from HuggingFace.
        Uses safetensors for efficient weight loading without loading entire model.

        If layer_paths contains 'model' or is empty, loads all weights.
        """
        state_dict = {}

        try:
            # Use snapshot_download for efficient caching
            self.send_request(
                "debug_print",
                (
                    f"DistributedWorker -> Checking cache for {model_name}",
                    "blue",
                    logging.DEBUG,
                ),
            )
            model_path = snapshot_download(
                repo_id=model_name,
                cache_dir=self.hf_cache_dir,
                allow_patterns=[
                    "*.safetensors",
                    "*.bin",
                    "*.json",
                ],
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
                local_files_only=False,
            )
            self.send_request(
                "debug_print",
                (
                    f"DistributedWorker -> Model located at: {model_path}",
                    "blue",
                    logging.DEBUG,
                ),
            )

            # Find all safetensors files
            safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))

            if safetensor_files:
                self.send_request(
                    "debug_print",
                    (
                        f"DistributedWorker -> Found {len(safetensor_files)} safetensors files",
                        "blue",
                        logging.DEBUG,
                    ),
                )

                # Load only specific layers
                layer_path_to_idx = {path: idx for idx, path in enumerate(layer_paths)}

                for shard_path in safetensor_files:
                    self.send_request(
                        "debug_print",
                        (
                            f"DistributedWorker -> Reading weights from {os.path.basename(shard_path)}",
                            "blue",
                            logging.DEBUG,
                        ),
                    )

                    with safe_open(shard_path, framework="pt", device="cpu") as f:
                        keys_loaded = 0
                        for key in f.keys():
                            # Check if key starts with any of our layer paths
                            for layer_path, layer_idx in layer_path_to_idx.items():
                                layer_prefix = layer_path + '.'
                                if key.startswith(layer_prefix):
                                    # Extract the part after the layer path
                                    new_key = key[len(layer_prefix) :]

                                    if single and '.' in new_key:
                                        # For single modules, remove one more level
                                        new_key = new_key.split('.', 1)[1]
                                    elif len(new_key.split(".")) > 1:
                                        new_key = key.split('.', 1)[1]

                                    state_dict[new_key] = f.get_tensor(key)
                                    keys_loaded += 1
                                    break

                        if keys_loaded > 0:
                            self.send_request(
                                "debug_print",
                                (
                                    f"DistributedWorker -> Loaded {keys_loaded} tensors from this shard",
                                    "blue",
                                    logging.DEBUG,
                                ),
                            )

                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()

                self.send_request(
                    "debug_print",
                    (
                        f"DistributedWorker -> Total: Loaded {len(state_dict)} weight tensors for {len(layer_paths)} layers",
                        "blue",
                        logging.DEBUG,
                    ),
                )

            else:
                # Fallback: use pytorch_model.bin files
                self.send_request(
                    "debug_print",
                    (
                        f"DistributedWorker -> No safetensors found, trying pytorch_model.bin",
                        "yellow",
                        logging.ERROR,
                    ),
                )
                raise ValueError(f"No weight files found in {model_path}")

        except Exception as e:
            logging.error(f"Error loading weights: {e}")
            raise ValueError(f"Failed to load layer weights: {str(e)}")

        return state_dict

    def _load_single_module(
        self, model_name: str, base_model: torch.nn.Module, module_info: Dict[str, Any]
    ) -> torch.nn.Module:
        """
        Load a single module (e.g., just the RMSNorm layer).
        Uses empty weights initialization and only loads required module weights.
        """
        parent_module_path = module_info.get('parent_module_path', '')
        module_class_name = module_info.get('module', '')
        module_id = module_info.get("module_id")

        self.send_request(
            "debug_print",
            (
                f"DistributedWorker -> Loading single module {module_class_name} from {model_name}",
                "blue",
                logging.DEBUG,
            ),
        )

        if parent_module_path == "":
            self.send_request(
                "debug_print",
                (
                    f"DistributedWorker -> Parent module is entire model — loading full model.",
                    "blue",
                    logging.DEBUG,
                ),
            )

            # aggressive cleanup before full model load
            if module_id in self.modules:
                del self.modules[module_id]

            del base_model
            self.cleanup_memory()

            final_model = self._load_full_model(model_name, module_info)
            return final_model

        # Extract the specific module with empty weights
        if parent_module_path and parent_module_path != "model":
            target_module = get_nested_module(base_model, parent_module_path)
            effective_layer_path = parent_module_path
        else:
            # parent_module_path is 'model' or empty -> try to find by class name
            effective_layer_path = _find_module_path_by_class(
                base_model, module_class_name
            )
            if effective_layer_path is None:
                # if not found, as a safe fallback return the root module but warn the caller
                target_module = base_model
                effective_layer_path = parent_module_path or "model"
            else:
                target_module = get_nested_module(base_model, effective_layer_path)

        # Get name of model for loading weights
        base_model_prefix = getattr(base_model, "base_model_prefix", None)

        # Load only the weights for this specific module
        self.send_request(
            "debug_print",
            (
                f"DistributedWorker -> Loading weights for {parent_module_path}",
                "blue",
                logging.DEBUG,
            ),
        )

        state_dict = self._load_specific_layer_weights(
            model_name,
            [effective_layer_path],
            single=True,
            base_model_prefix=base_model_prefix,
        )

        del base_model
        self.cleanup_memory()

        target_module = target_module.to_empty(device="cpu")

        # Load the state dict
        missing_keys, unexpected_keys = target_module.load_state_dict(
            state_dict, strict=False
        )

        del state_dict
        self.cleanup_memory()

        if missing_keys:
            self.send_request(
                "debug_print",
                (
                    f"DistributedWorker -> Error loading single module weights on model: {model_name}"
                    f"\n Module: {effective_layer_path}\n Missing keys: {missing_keys}\n Unexpected keys: {unexpected_keys}",
                    "bright_red",
                    logging.CRITICAL,
                ),
            )

        # Move to device
        target_module = target_module.to(self.device)

        self.send_request(
            "debug_print",
            (
                f"DistributedWorker -> Successfully loaded single module {module_class_name}",
                "blue",
                logging.DEBUG,
            ),
        )
        return target_module

    def _load_full_model(self, model_name: str, module_info: dict) -> torch.nn.Module:
        """
        Load a complete model from HuggingFace with optimal memory usage.
        Uses HF's native loading which is more memory-efficient than manual skeleton+weights.
        """
        model_type = module_info.get('model_type', 'chat')
        num_gpus = torch.cuda.device_count()

        # Force garbage collection before loading
        self.cleanup_memory()
        
        load_kwargs = {
            "low_cpu_mem_usage": True,
            "torch_dtype": torch.float16,  # TODO route quantization params through job requests, should also be done for module loading
        }

        # Only use device_map for multi-GPU
        if num_gpus > 1:
            load_kwargs["device_map"] = "auto"
        else:
            # For single GPU, load to CPU first then move
            load_kwargs["device_map"] = "cpu"

        self.send_request(
            "debug_print",
            (
                f"DistributedWorker -> Loading full model {model_name} with type {model_type}",
                "blue",
                logging.DEBUG,
            ),
        )

        # Load model based on type
        if model_type in ("causal", "chat"):
            final_model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )
        elif model_type == "seq2seq":
            final_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, **load_kwargs
            )
        elif model_type == "vision2text":
            final_model = AutoModelForVision2Seq.from_pretrained(
                model_name, **load_kwargs
            )
        elif model_type == "audio2text":
            final_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name, **load_kwargs
            )
        else:
            model_config = AutoConfig.from_pretrained(model_name)
            final_model = AutoModel.from_pretrained(
                model_name, config=model_config, **load_kwargs
            )

        # Move to GPU only after fully loaded (for single GPU)
        if num_gpus == 1 and self.device.type == "cuda":
            self.cleanup_memory()
            final_model = final_model.to(self.device)

        self.send_request(
            "debug_print",
            (
                f"DistributedWorker -> Successfully loaded full model {model_name}",
                "blue",
                logging.DEBUG,
            ),
        )
        return final_model

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
                ("DistributedWorker -> Optimizer zeroed.", "bright_blue", logging.DEBUG),
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
                        size, name, stream = args
                        self._handle_generate(module_id, size, name, stream)
                    else:
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
