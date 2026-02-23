import importlib
import json
from collections import defaultdict
from typing import Dict
import time
import os
import inspect
import psutil
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers.utils import ModelOutput
from transformers.cache_utils import DynamicCache
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
)


MODELS_CACHE_PATH = "logs/models.json"


def format_memory_size(number: int) -> str:
    """Format memory size in readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if number < 1024:
            return f"{number:.2f} {unit}"
        number /= 1024
    return f"{number:.2f} TB"


def estimate_memory(
    module: nn.Module,
    training: bool = True,
    batch_size: int = 256,
    seq_length: int = 2048,
    dtype: torch.dtype = torch.float16,
    optimizer_type: str = "adam",
    include_kv_cache: bool = True,
    recursive: bool = True,
    count_activations: bool = True,
) -> tuple[float, dict]:
    """Estimate GPU memory required for a model."""

    dtype_size = torch.tensor([], dtype=dtype).element_size()

    breakdown = {
        "parameters": 0,
        "gradients": 0,
        "optimizer": 0,
        "activations": 0,
        "kv_cache": 0,
    }

    # ---- parameters ----
    if recursive:
        param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
        param_bytes += sum(b.numel() * b.element_size() for b in module.buffers())
    else:
        param_bytes = sum(
            p.numel() * p.element_size() for p in module.parameters(recurse=False)
        )
        param_bytes += sum(
            b.numel() * b.element_size() for b in module.buffers(recurse=False)
        )

    breakdown["parameters"] = param_bytes

    # ---- training extras ----
    if training:
        breakdown["gradients"] = param_bytes
        if optimizer_type.lower() in {"adam", "adamw"}:
            breakdown["optimizer"] = 2 * param_bytes * (4 / dtype_size)
        else:
            breakdown["optimizer"] = param_bytes

    # ---- activations ----
    if count_activations:
        if hasattr(module, "config"):
            hidden_size = module.config.hidden_size
        elif hasattr(module, "hidden_size"):
            hidden_size = module.hidden_size
        elif hasattr(module, "embed_dim"):
            hidden_size = module.embed_dim
        elif hasattr(module, "d_model"):
            hidden_size = module.d_model
        else:
            total_params = sum(p.numel() for p in module.parameters())
            hidden_size = max(256, min(int((total_params / 12) ** 0.5), 8192))

        activation_multiplier = 4 if not training else 7

        breakdown["activations"] = (
            batch_size * seq_length * hidden_size * dtype_size * activation_multiplier
        )

        if include_kv_cache and hasattr(module, "config") and not training:
            num_layers = module.config.num_hidden_layers
            num_heads = getattr(
                module.config,
                "num_key_value_heads",
                module.config.num_attention_heads,
            )
            head_dim = hidden_size // module.config.num_attention_heads

            breakdown["kv_cache"] = (
                batch_size
                * seq_length
                * num_layers
                * num_heads
                * head_dim
                * 2
                * dtype_size
            )

    # ---- overhead ----
    OVERHEAD = 1.20
    total = sum(breakdown.values()) * OVERHEAD

    return total, breakdown


def get_gpu_memory(max_vram_gb: float | None = None):
    """
    Returns available memory in bytes.
    - Uses total free CUDA VRAM if available.
    - Falls back to available system RAM if CUDA is not available.
    - If max_vram_gb is provided, caps the returned memory.
    """

    # Determine max memory cap
    max_memory_bytes = None
    if max_vram_gb is not None and max_vram_gb > 0:
        max_memory_bytes = int(max_vram_gb * 1e9)

    # Case 1: CUDA available
    if torch.cuda.is_available():
        memory = 0
        for device in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(device)
            memory += free

    # Case 2: Fallback to system RAM
    else:
        memory = psutil.virtual_memory().available

    # Apply cap if specified
    if max_memory_bytes is not None:
        memory = min(memory, max_memory_bytes)

    return int(memory)


def find_module(module: nn.Module, target_name: str, ids: list = []):
    if not list(module.named_children()):
        return
    children = list(module.named_children())
    for i in range(len(children)):
        name, values = children[i]
        new_ids = ids + [i]
        if name == target_name:
            return values, new_ids
        res = find_module(values, target_name, new_ids)
        if res:
            return res


def access_module(module: nn.Module, indices: list):
    """Access a module from a model based on its integer ID (depth) and return the module class name."""
    if indices == [-1]:
        # If -1 is passed, return the root module and its class name
        return module, type(module).__name__

    current_module = module
    module_name = type(module).__name__  # Set the root module's class name

    for index in indices:
        children = list(
            current_module.named_children()
        )  # Get all child modules with their names
        if index >= len(children):
            raise IndexError("Index out of range for current module's children.")

        # Access the child module at the specified index
        module_name = type(
            children[index][1]
        ).__name__  # Update to the class name of the child module
        current_module = children[index][1]  # Get the actual child module

    return current_module, module_name


def detach_tensor(obj, clone: bool = False):
    """
    Recursively detach tensors (and optionally clone them) from GPU to CPU.
    Supports Tensor, DynamicCache, ModelOutput, list, tuple, and dict.
    """
    # Case 1: torch.Tensor
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        if clone:
            t = t.clone()
        return t

    # Case 2: DynamicCache
    elif isinstance(obj, DynamicCache):
        new_cache = DynamicCache()
        new_cache.key_cache = [
            detach_tensor(t, clone=clone) if isinstance(t, torch.Tensor) else t
            for t in obj.key_cache
        ]
        new_cache.value_cache = [
            detach_tensor(t, clone=clone) if isinstance(t, torch.Tensor) else t
            for t in obj.value_cache
        ]
        new_cache._seen_tokens = obj._seen_tokens
        return new_cache

    # Case 3: ModelOutput (transformers container)
    elif isinstance(obj, ModelOutput):
        new_out = obj.__class__()
        for key, value in obj.items():
            if isinstance(
                value, (torch.Tensor, DynamicCache, ModelOutput, list, tuple, dict)
            ):
                new_out[key] = detach_tensor(value, clone=clone)
            else:
                new_out[key] = value
        return new_out

    # Case 4: list or tuple
    elif isinstance(obj, (list, tuple)):
        new_seq = [
            (
                detach_tensor(v, clone=clone)
                if isinstance(
                    v, (torch.Tensor, DynamicCache, ModelOutput, list, tuple, dict)
                )
                else v
            )
            for v in obj
        ]
        return type(obj)(new_seq)

    # Case 5: dictionary
    elif isinstance(obj, dict):
        return {
            k: (
                detach_tensor(v, clone=clone)
                if isinstance(
                    v, (torch.Tensor, DynamicCache, ModelOutput, list, tuple, dict)
                )
                else v
            )
            for k, v in obj.items()
        }

    else:
        raise TypeError(f"Unsupported input type: {type(obj)}")


def attach_tensor(tensor, device):
    # Case 1: DynamicCache
    if isinstance(tensor, DynamicCache):
        tensor.key_cache = [
            attach_tensor(t, device) if isinstance(t, torch.Tensor) else t
            for t in tensor.key_cache
        ]
        tensor.value_cache = [
            attach_tensor(t, device) if isinstance(t, torch.Tensor) else t
            for t in tensor.value_cache
        ]
        return tensor

    # Case 2: torch.Tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)

    # Case 3: ModelOutput
    elif isinstance(tensor, ModelOutput):
        for key, value in tensor.items():
            if isinstance(
                value, (torch.Tensor, DynamicCache, ModelOutput, list, tuple, dict)
            ):
                tensor[key] = attach_tensor(value, device)
        return tensor

    # Case 4: list or tuple
    elif isinstance(tensor, (list, tuple)):
        return type(tensor)(
            (
                attach_tensor(t, device)
                if isinstance(
                    t, (torch.Tensor, DynamicCache, ModelOutput, list, tuple, dict)
                )
                else t
            )
            for t in tensor
        )

    # Case 5: dict
    elif isinstance(tensor, dict):
        return {
            key: (
                attach_tensor(value, device)
                if isinstance(
                    value, (torch.Tensor, DynamicCache, ModelOutput, list, tuple, dict)
                )
                else value
            )
            for key, value in tensor.items()
        }

    elif hasattr(tensor, "to"):
        return tensor.to(device)

    else:
        raise TypeError(f"Unsupported input type: {type(tensor)}")


def enable_grad(tensor):
    """
    Enables gradient computation on floating-point Tensors within nested structures.
    """
    if isinstance(tensor, torch.Tensor):
        # Enable grad if the tensor is a floating-point type
        if tensor.is_floating_point():
            return tensor.detach().clone().requires_grad_(True)
        return tensor

    elif isinstance(tensor, ModelOutput):
        # Iterate through ModelOutput fields, enabling grad for any Tensors
        for key in tensor.__dataclass_fields__:
            value = getattr(tensor, key)
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                tensor[key] = value.detach().clone().requires_grad_(True)
        return tensor

    elif isinstance(tensor, (list, tuple)):
        # Recursively apply to each element in lists or tuples
        return type(tensor)(enable_grad(t) for t in tensor)

    elif isinstance(tensor, dict):
        # Recursively apply to each item in dictionaries
        return {key: enable_grad(value) for key, value in tensor.items()}

    else:
        return tensor


def handle_output(tensor):
    """
    Handle various output types from models, convert to their raw tensor form:
    - Check for specific attributes like `logits` and `last_hidden_state`.
    - If output is a tuple, return the first element (assumed to be the main output tensor).
    - If output is a dictionary, check common keys or return the first tensor found.
    - If it's already a tensor, return as-is.
    """
    if hasattr(tensor, "logits"):
        return tensor.logits
    elif hasattr(tensor, "last_hidden_state"):
        return tensor.last_hidden_state
    elif isinstance(tensor, (tuple, list)):
        if len(tensor) == 1:
            return tensor[0] if isinstance(tensor[0], torch.Tensor) else tensor
        elif len(tensor) > 1:
            return type(tensor)(t for t in tensor if isinstance(t, torch.Tensor))
        return tensor
    elif isinstance(tensor, dict):
        # Look for common keys like 'logits' or 'last_hidden_state'
        for key in ["logits", "last_hidden_state"]:
            if key in tensor and isinstance(tensor[key], torch.Tensor):
                return tensor[key]
        # Fallback to first tensor found in dict
        for value in tensor.values():
            if isinstance(value, torch.Tensor):
                return value
    elif isinstance(tensor, torch.Tensor):
        return tensor
    raise ValueError("Unsupported output format: could not find a tensor.")


def replace_output_with_custom_grad(combined_output, custom_grad_output):
    """
    Replace the main output tensor (logits, last_hidden_state, etc.) in the combined_output
    with the custom_grad_output, preserving structure and returning a ModelOutput when possible.
    """

    # If the combined output is already a tensor
    if isinstance(combined_output, torch.Tensor):
        return custom_grad_output

    # Handle ModelOutput subclasses
    if isinstance(combined_output, ModelOutput):
        data = {}

        # Extract fields safely (HF-version independent)
        for key in combined_output.__dataclass_fields__:
            data[key] = getattr(combined_output, key)

        # Replace primary tensor
        if "logits" in data and isinstance(data["logits"], torch.Tensor):
            data["logits"] = custom_grad_output

        elif "last_hidden_state" in data and isinstance(
            data["last_hidden_state"], torch.Tensor
        ):
            data["last_hidden_state"] = custom_grad_output

        else:
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = custom_grad_output
                    break

        return combined_output.__class__(**data)

    # Handle dict outputs
    if isinstance(combined_output, dict):
        new_output = dict(combined_output)

        if "logits" in new_output and isinstance(new_output["logits"], torch.Tensor):
            new_output["logits"] = custom_grad_output

        elif "last_hidden_state" in new_output and isinstance(
            new_output["last_hidden_state"], torch.Tensor
        ):
            new_output["last_hidden_state"] = custom_grad_output

        else:
            for k, v in new_output.items():
                if isinstance(v, torch.Tensor):
                    new_output[k] = custom_grad_output
                    break

        return ModelOutput(**new_output)

    raise TypeError(f"Unsupported output type: {type(combined_output)}")


def combine_micro_batches(micro_batches):
    """
    Combines the micro-batch outputs into a single output.
    """
    if isinstance(micro_batches[0], torch.Tensor):
        # If outputs are tensors, concatenate them along the batch dimension
        return torch.cat(micro_batches, dim=0)

    elif isinstance(micro_batches[0], ModelOutput):
        combined_output = defaultdict(list)

        for output in micro_batches:
            for key in output.__dataclass_fields__:
                value = getattr(output, key)
                combined_output[key].append(value)

        # Concatenate fields that are tensors
        final_output = {}
        micro_loss = None

        for key, value in combined_output.items():
            if isinstance(value[0], torch.Tensor):
                # Handle zero-dimensional tensors
                if key == "loss":
                    # Average the loss and store individual losses for backward pass
                    averaged_loss = torch.mean(torch.stack(value))
                    setattr(averaged_loss, "micro_loss", value)
                    final_output[key] = averaged_loss

                elif value[0].dim() == 0:
                    final_output[key] = torch.stack(value)
                else:
                    final_output[key] = torch.cat(value, dim=0)
            else:
                final_output[key] = value  # Leave as is if not a tensor

        return type(micro_batches[0])(**final_output)

    elif isinstance(micro_batches[0], dict):
        combined_output = defaultdict(list)

        for output in micro_batches:
            for key, value in output.items():
                combined_output[key].append(value)

        final_output = {}
        for key, value in combined_output.items():
            if isinstance(value[0], torch.Tensor):
                # Handle scalar tensors separately
                if value[0].dim() == 0:
                    final_output[key] = torch.stack(value)
                else:
                    final_output[key] = torch.cat(value, dim=0)
            else:
                final_output[key] = value  # Leave non-tensor values as-is

        return final_output

    else:
        raise TypeError("Unsupported output type")


def split_micro_batches(inputs, n_chunks):
    """
    Safely chunks inputs into n_chunks.
    - Tensors and ModelOutput are split along batch dim.
    - Dicts are split recursively.
    - Scalars (bool, None, int, float, str) are passed through to each chunk.
    """
    if isinstance(inputs, torch.Tensor):
        return torch.chunk(inputs, n_chunks)

    elif isinstance(inputs, ModelOutput):
        chunked_outputs = [defaultdict(list) for _ in range(n_chunks)]
        for key in inputs.__dataclass_fields__:
            value = getattr(inputs, key)
            if isinstance(value, torch.Tensor):
                value_chunks = torch.chunk(value, n_chunks)
                for i in range(n_chunks):
                    chunked_outputs[i][key] = value_chunks[i]
            else:
                # Non-tensor fields are replicated across chunks
                for i in range(n_chunks):
                    chunked_outputs[i][key] = value
        return [type(inputs)(**dict(chunk)) for chunk in chunked_outputs]

    elif isinstance(inputs, dict):
        chunked_dicts = [{} for _ in range(n_chunks)]
        for key, value in inputs.items():
            # Recursively chunk if possible, otherwise replicate
            try:
                value_chunks = split_micro_batches(value, n_chunks)
                # If value_chunks is not iterable (scalar), replicate
                if not hasattr(value_chunks, "__getitem__"):
                    value_chunks = [value] * n_chunks
            except Exception:
                value_chunks = [value] * n_chunks

            for i in range(n_chunks):
                chunked_dicts[i][key] = value_chunks[i]

        return chunked_dicts

    else:
        # Scalars, None, bools, strings, etc. just get replicated
        return [inputs] * n_chunks


def get_batch_size(inputs):
    """
    Returns the batch size from the inputs to the forward pass.
    Handles both tensor and ModelOutput types.
    """
    if isinstance(inputs, torch.Tensor):
        return inputs.size(0)
    elif isinstance(inputs, ModelOutput):
        for key in inputs.__dataclass_fields__:
            value = getattr(inputs, key)
            if isinstance(value, torch.Tensor):
                return value.size(0)
    else:
        raise ValueError("Unsupported input type")


def tensor_to_bytes(tensor):
    """Serialize tensor or tensor-like structures into bytes, including dtype."""

    def _serialize(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj

        elif isinstance(obj, torch.Tensor):
            return {
                "__serialized__": True,
                "module": obj.__class__.__module__,
                "class": obj.__class__.__name__,
                "dtype": str(obj.dtype),  # Save dtype as string
                "data": obj.tolist(),
            }

        elif hasattr(obj, "__class__") and obj.__class__.__name__ == "BatchEncoding":
            return {
                "__serialized__": True,
                "module": obj.__class__.__module__,
                "class": obj.__class__.__name__,
                "data": {k: _serialize(v) for k, v in obj.items()},
            }

        # Handle tokenizers (like Qwen2TokenizerFast)
        elif (
            hasattr(obj, "vocab_size")
            and hasattr(obj, "special_tokens")
            and hasattr(obj, "get_vocab")
        ):
            # Extract essential tokenizer metadata
            tokenizer_data = {
                "name_or_path": getattr(obj, "name_or_path", None),
                "vocab_size": getattr(obj, "vocab_size", None),
                "model_max_length": getattr(obj, "model_max_length", None),
                "is_fast": getattr(obj, "is_fast", None),
                "padding_side": getattr(obj, "padding_side", None),
                "truncation_side": getattr(obj, "truncation_side", None),
                "special_tokens": {
                    k: _serialize(v)
                    for k, v in getattr(obj, "special_tokens", {}).items()
                },
                "added_tokens": [],  # We'll store essential info about added tokens
                "class_name": obj.__class__.__name__,
                "module_name": obj.__class__.__module__,
            }

            # Capture essential added token info
            if hasattr(obj, "added_tokens_decoder") and getattr(
                obj, "added_tokens_decoder", None
            ):
                for token_id, token in obj.added_tokens_decoder.items():
                    if hasattr(token, "content"):
                        tokenizer_data["added_tokens"].append(
                            {
                                "id": token_id,
                                "content": token.content,
                                "special": getattr(token, "special", False),
                            }
                        )

            return {
                "__serialized__": True,
                "module": "tokenizers",
                "class": "Tokenizer",
                "is_tokenizer": True,
                "data": tokenizer_data,
            }

        # Handle stopping criteria objects that may contain tokenizers
        elif hasattr(obj, "__class__") and "StoppingCriteria" in obj.__class__.__name__:
            serialized_data = {
                "__serialized__": True,
                "module": obj.__class__.__module__,
                "class": obj.__class__.__name__,
                "is_stopping_criteria": True,
                "data": {},
            }

            # Special handling for StringStoppingCriteria which requires tokenizer
            if obj.__class__.__name__ == "StringStoppingCriteria" and hasattr(
                obj, "tokenizer"
            ):
                serialized_data["data"]["tokenizer"] = _serialize(obj.tokenizer)

            # Serialize all attributes that can be serialized
            for attr_name in dir(obj):
                # Skip private attributes and methods
                if attr_name.startswith("_") or callable(getattr(obj, attr_name)):
                    continue

                try:
                    attr_value = getattr(obj, attr_name)
                    serialized_data["data"][attr_name] = _serialize(attr_value)
                except (ValueError, TypeError):
                    # If we can't serialize an attribute, log it but continue
                    serialized_data["data"][
                        attr_name
                    ] = f"UNSERIALIZABLE_OBJECT_{type(attr_value).__name__}"

            return serialized_data

        elif hasattr(obj, "__dict__"):
            try:
                return {
                    "__serialized__": True,
                    "module": obj.__class__.__module__,
                    "class": obj.__class__.__name__,
                    "is_object": True,
                    "data": _serialize(obj.__dict__),
                }
            except (ValueError, TypeError):
                pass

        elif isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [_serialize(v) for v in obj]

        elif isinstance(obj, tuple):
            return {
                "__serialized__": True,
                "type": "tuple",
                "data": [_serialize(v) for v in obj],
            }

        # Return a placeholder for unsupported types
        return f"UNSERIALIZABLE_OBJECT_{type(obj).__name__}"

    return json.dumps(_serialize(tensor)).encode("utf-8")


def bytes_to_tensor(tensor_data):
    """Deserialize bytes or JSON-like dicts/lists into tensors, tokenizers, and other objects."""
    if isinstance(tensor_data, bytes):
        tensor_data = json.loads(tensor_data)

    def _deserialize(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj

        # Handle strings that represent unserializable objects
        if isinstance(obj, str) and obj.startswith("UNSERIALIZABLE_OBJECT_"):
            # Return None or a placeholder for unserializable objects
            return None

        if isinstance(obj, list):
            return [_deserialize(v) for v in obj]

        if (
            isinstance(obj, dict)
            and obj.get("__serialized__")
            and obj.get("type") == "tuple"
        ):
            return tuple(_deserialize(v) for v in obj["data"])

        if isinstance(obj, dict):
            # Special handling for tokenizer metadata
            if obj.get("__tokenizer_metadata__"):
                try:
                    # Import the tokenizer class from transformers
                    from transformers import AutoTokenizer

                    # Get the name or path to load from
                    name_or_path = obj.get("name_or_path")
                    if not name_or_path:
                        # If no name_or_path, return the metadata as is
                        return obj

                    # Safely load the tokenizer using AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
                    return tokenizer
                except (ImportError, Exception) as e:
                    # Fall back to just returning the metadata
                    return obj

            # Handle serialized objects
            elif obj.get("__serialized__"):
                module_name = obj.get("module")
                cls_name = obj.get("class")
                data = obj.get("data")

                # Handle tensors
                if cls_name == "Tensor" or cls_name.endswith("Tensor"):
                    dtype_str = obj["dtype"]
                    if "." in dtype_str:
                        dtype_str = dtype_str.split(".")[
                            -1
                        ]  # Remove 'torch.' if present
                    return torch.tensor(data, dtype=getattr(torch, dtype_str))

                # Handle BatchEncoding objects
                elif cls_name == "BatchEncoding":
                    from transformers import BatchEncoding

                    return BatchEncoding({k: _deserialize(v) for k, v in data.items()})

                # Handle stopping criteria objects
                elif obj.get("is_stopping_criteria"):
                    try:
                        # First, deserialize all the data to handle nested objects like tokenizers
                        deserialized_data = {
                            k: _deserialize(v) for k, v in data.items()
                        }

                        # Try to import the stopping criteria class
                        module_parts = module_name.split(".")
                        criteria_module = __import__(module_parts[0])
                        for part in module_parts[1:]:
                            try:
                                criteria_module = getattr(criteria_module, part)
                            except AttributeError:
                                # If module path is invalid, just return the deserialized data
                                return deserialized_data

                        try:
                            criteria_class = getattr(criteria_module, cls_name)
                        except AttributeError:
                            # If class doesn't exist, return the data
                            return deserialized_data

                        # Special handling for StringStoppingCriteria which needs tokenizer
                        if cls_name == "StringStoppingCriteria":
                            # Extract tokenizer and stop_string
                            tokenizer = deserialized_data.get("tokenizer")
                            stop_string = deserialized_data.get("stop_string", "")

                            if tokenizer is None:
                                # Fall back to dictionary since we can't instantiate without tokenizer
                                return deserialized_data

                            try:
                                # Create the criteria with tokenizer and stop_string
                                criteria_instance = criteria_class(
                                    tokenizer, stop_string
                                )

                                # Set any additional attributes
                                for attr_name, attr_value in deserialized_data.items():
                                    if attr_name not in ["tokenizer", "stop_string"]:
                                        try:
                                            setattr(
                                                criteria_instance, attr_name, attr_value
                                            )
                                        except (AttributeError, TypeError):
                                            pass

                                return criteria_instance
                            except Exception:
                                # If instantiation fails for any reason, return the data
                                return deserialized_data
                        else:
                            # For other stopping criteria, try constructor with deserialized data
                            try:
                                sig = inspect.signature(criteria_class.__init__)
                                param_names = list(sig.parameters.keys())[
                                    1:
                                ]  # Skip 'self'

                                # Prepare constructor arguments
                                kwargs = {}
                                for param in param_names:
                                    if param in deserialized_data:
                                        kwargs[param] = deserialized_data[param]

                                # Create the instance
                                criteria_instance = criteria_class(**kwargs)

                                # Set any remaining attributes that weren't in the constructor
                                for attr_name, attr_value in deserialized_data.items():
                                    if (
                                        attr_name not in param_names
                                        and not attr_name.startswith("_")
                                    ):
                                        try:
                                            setattr(
                                                criteria_instance, attr_name, attr_value
                                            )
                                        except (AttributeError, TypeError):
                                            pass

                                return criteria_instance
                            except Exception:
                                # If instantiation fails, return the data
                                return deserialized_data
                    except Exception:
                        # Fall back to a dictionary if reconstruction fails
                        return {k: _deserialize(v) for k, v in data.items()}

                elif obj.get("is_object"):
                    try:
                        import importlib

                        # Dynamically import the module and get the class
                        module = importlib.import_module(module_name)
                        cls = getattr(module, cls_name)

                        # Try to create an instance
                        try:
                            # First try with no arguments
                            instance = cls()
                        except TypeError:
                            # If that fails, try with deserialized data as kwargs
                            try:
                                sig = inspect.signature(cls.__init__)
                                param_names = list(sig.parameters.keys())[
                                    1:
                                ]  # Skip 'self'

                                # Prepare constructor arguments from data
                                deserialized_data = _deserialize(data)
                                kwargs = {}
                                for param in param_names:
                                    if param in deserialized_data:
                                        kwargs[param] = deserialized_data[param]

                                instance = cls(**kwargs)

                            except Exception:
                                # Last resort: return deserialized data as dict
                                return _deserialize(data)

                        # Set all attributes from the data
                        deserialized_data = _deserialize(data)
                        if isinstance(deserialized_data, dict):
                            for attr_name, attr_value in deserialized_data.items():
                                try:
                                    setattr(instance, attr_name, attr_value)
                                except (AttributeError, TypeError):
                                    # Skip attributes that can't be set
                                    pass

                        return instance

                    except (ImportError, AttributeError) as e:
                        # If we can't import the class, return deserialized data as dict
                        return _deserialize(data)
                    except Exception as e:
                        # For any other error, return deserialized data
                        return _deserialize(data)

                # For other serialized objects, return dictionary representation
                return {
                    k: _deserialize(v) for k, v in data.items() if k != "__serialized__"
                }

            # Handle regular dictionaries
            return {k: _deserialize(v) for k, v in obj.items()}

        return obj

    return _deserialize(tensor_data)


def load_models_cache():
    try:
        with open(MODELS_CACHE_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_models_cache(models):
    os.makedirs(os.path.dirname(MODELS_CACHE_PATH), exist_ok=True)
    with open(MODELS_CACHE_PATH, "w") as f:
        json.dump(models, f, indent=4)


def get_popular_model_stats(
    days: int = 7, min_requests: int = 1, limit: int = None
) -> Dict:
    """
    Get popular model demand statistics for API responses

    Args:
        days: Number of days to look back for request counts (default: 7)
        min_requests: Minimum requests to include in results (default: 1)
        limit: Maximum number of models to return (default: None - all models)

    Returns:
        Dictionary containing popular model statistics
    """
    cache = load_models_cache()

    if not cache:
        return {
            "status": "success",
            "data": {
                "popular_models": [],
                "total_models_tracked": 0,
                "time_period_days": days,
                "generated_at": time.time(),
            },
        }

    cutoff_time = time.time() - (days * 24 * 3600)
    popular_models = []

    for model_name, model_data in cache.items():
        demand_metrics = model_data.get("demand_metrics", {})
        timestamps = demand_metrics.get("request_timestamps", [])

        # Count recent requests
        recent_requests = sum(1 for ts in timestamps if ts >= cutoff_time)

        if recent_requests >= min_requests:
            model_stats = {
                "model_name": model_name,
                "recent_requests": recent_requests,
                "total_requests": demand_metrics.get("total_requests", 0),
                "last_accessed": demand_metrics.get("last_accessed"),
                "has_distribution": model_data.get("distribution") is not None,
                "requests_per_day": round(recent_requests / days, 2) if days > 0 else 0,
            }

            # Add human-readable last accessed time
            if model_stats["last_accessed"]:
                time_ago = time.time() - model_stats["last_accessed"]
                if time_ago < 3600:  # Less than 1 hour
                    model_stats["last_accessed_human"] = (
                        f"{int(time_ago // 60)} minutes ago"
                    )
                elif time_ago < 86400:  # Less than 1 day
                    model_stats["last_accessed_human"] = (
                        f"{int(time_ago // 3600)} hours ago"
                    )
                else:
                    model_stats["last_accessed_human"] = (
                        f"{int(time_ago // 86400)} days ago"
                    )
            else:
                model_stats["last_accessed_human"] = "Never"

            popular_models.append(model_stats)

    # Sort by recent requests (descending)
    popular_models.sort(key=lambda x: x["recent_requests"], reverse=True)

    # Apply limit if specified
    if limit and limit > 0:
        popular_models = popular_models[:limit]

    return {
        "status": "success",
        "data": {
            "popular_models": popular_models,
            "total_models_tracked": len(cache),
            "models_with_recent_activity": len(popular_models),
            "time_period_days": days,
            "min_requests_threshold": min_requests,
            "generated_at": time.time(),
        },
    }


def get_model_detailed_stats(model_name: str) -> Dict:
    """
    Get detailed statistics for a specific model

    Args:
        model_name: Name of the model to get stats for

    Returns:
        Dictionary containing detailed model statistics
    """
    cache = load_models_cache()

    if model_name not in cache:
        return {
            "status": "error",
            "message": f"Model '{model_name}' not found in cache",
            "data": None,
        }

    model_data = cache[model_name]
    demand_metrics = model_data.get("demand_metrics", {})
    timestamps = demand_metrics.get("request_timestamps", [])

    current_time = time.time()

    # Calculate request counts for different time periods
    time_periods = {
        "1_hour": 3600,
        "1_day": 86400,
        "7_days": 7 * 86400,
        "30_days": 30 * 86400,
    }

    request_counts = {}
    for period_name, seconds in time_periods.items():
        cutoff = current_time - seconds
        count = sum(1 for ts in timestamps if ts >= cutoff)
        request_counts[period_name] = count

    # Distribution info
    distribution = model_data.get("distribution")
    distribution_info = {
        "has_distribution": distribution is not None,
        "distribution_keys": list(distribution.keys()) if distribution else [],
    }

    return {
        "status": "success",
        "data": {
            "model_name": model_name,
            "demand_metrics": {
                "total_requests": demand_metrics.get("total_requests", 0),
                "last_accessed": demand_metrics.get("last_accessed"),
                "request_counts_by_period": request_counts,
                "recent_request_timestamps": (
                    timestamps[-10:] if len(timestamps) > 10 else timestamps
                ),
            },
            "distribution_info": distribution_info,
            "generated_at": current_time,
        },
    }


def get_nested_module(
    model: torch.nn.Module, path: str, target_class_name: str = None
) -> torch.nn.Module:
    """
    Navigate to a nested module using dot notation path.
    Example: 'model.layers.0' -> returns model.layers[0]
    """
    parts = path.split('.')
    current = model

    for i in range(len(parts)):
        part = parts[i]

        if part == "model":
            if not hasattr(current, "model") or (
                len(parts) > i + 1 and not hasattr(current.model, parts[i + 1])
            ):
                continue

        if part.isdigit():
            # Handle list/ModuleList indexing
            current = current[int(part)]
        else:
            # Handle attribute access
            current = getattr(current, part)

    return current


def resolve_module_from_path(model: nn.Module, path: str):
    """Return (parent_module, child_module, child_name)."""
    parts = path.split(".")
    parent = model
    for p in parts[:-1]:
        if p == "model":
            continue
        parent = getattr(parent, p)
    child_name = parts[-1]
    child = getattr(parent, child_name)
    return parent, child, child_name


def get_optimizer_from_spec(optimizer_spec: dict):
    module_path, class_name = optimizer_spec["class_path"].rsplit(".", 1)
    import_module = importlib.import_module(module_path)
    optimizer_cls = getattr(import_module, class_name)
    return optimizer_cls


def optimizer_to_spec(optimizer_cls):
    if not isinstance(optimizer_cls, type):
        raise TypeError("optimizer must be an optimizer class")

    optimizer_spec = {
        "framework": "torch",
        "class_path": f"{optimizer_cls.__module__}.{optimizer_cls.__name__}",
    }

    optimizer_spec["type"] = optimizer_spec["class_path"].rsplit(".", 1)[-1]
    return optimizer_spec


def load_model_skeleton(model_name: str, model_type: str = "chat"):
    """
    Load the HF model structure with empty weights.
    """
    # First, load the config
    model_config = AutoConfig.from_pretrained(model_name)

    # Then create model from config with init_empty_weights
    with init_empty_weights():
        if model_type in ("causal", "chat"):
            skeleton_model = AutoModelForCausalLM.from_config(model_config)
        elif model_type == "seq2seq":
            skeleton_model = AutoModelForSeq2SeqLM.from_config(model_config)
        elif model_type == "vision2text":
            skeleton_model = AutoModelForVision2Seq.from_config(model_config)
        elif model_type == "audio2text":
            skeleton_model = AutoModelForSpeechSeq2Seq.from_config(model_config)
        else:
            skeleton_model = AutoModel.from_config(model_config)

    skeleton_model.eval()  # Set to eval mode initially

    # Ensure no cached gradients or cached computations
    for param in skeleton_model.parameters():
        param.requires_grad = False

    return skeleton_model
