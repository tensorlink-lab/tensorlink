import importlib
import json
from collections import defaultdict
from typing import Dict
import time
import os
from safetensors.torch import save as st_save_bytes, load as st_load_bytes
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


def get_gpu_memory(max_vram_gb: float | None = None) -> int:
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


def tensor_to_bytes(obj):
    """
    Safe serialization: safetensors for tensors, JSON for structure.
    No pickle, no arbitrary code execution.
    """
    # Flatten the object into a tensor map + structure skeleton
    tensor_map = {}
    counter = [0]

    def _extract_tensors(o):
        """Replace tensors with placeholder keys, collect into tensor_map"""
        if isinstance(o, torch.Tensor):
            key = f"__tensor_{counter[0]}__"
            counter[0] += 1
            # safetensors requires contiguous float tensors on CPU
            tensor_map[key] = o.detach().cpu().contiguous().clone()
            return {
                "__tensor_ref__": key,
                "dtype": str(o.dtype),
                "shape": list(o.shape),
            }
        elif isinstance(o, dict):
            return {k: _extract_tensors(v) for k, v in o.items()}
        elif isinstance(o, (list, tuple)):
            result = [_extract_tensors(v) for v in o]
            return (
                {"__tuple__": True, "data": result} if isinstance(o, tuple) else result
            )
        elif isinstance(o, (int, float, bool, str, type(None))):
            return o
        elif hasattr(o, '__class__') and o.__class__.__name__ == 'DynamicCache':
            # Serialize DynamicCache as its key/value tensor lists
            return {
                "__dynamic_cache__": True,
                "key_cache": _extract_tensors(o.key_cache),
                "value_cache": _extract_tensors(o.value_cache),
                "_seen_tokens": o._seen_tokens,
            }
        else:
            return None  # drop unserializable objects safely

    structure = _extract_tensors(obj)
    structure_bytes = json.dumps(structure).encode("utf-8")

    # Pack: 4-byte length prefix for structure, then safetensors blob
    if tensor_map:
        tensor_bytes = st_save_bytes(tensor_map)
    else:
        tensor_bytes = b""

    structure_len = len(structure_bytes).to_bytes(4, "big")
    return structure_len + structure_bytes + tensor_bytes


def bytes_to_tensor(data: bytes):
    """
    Safe deserialization matching tensor_to_bytes.
    """
    structure_len = int.from_bytes(data[:4], "big")
    structure_bytes = data[4 : 4 + structure_len]
    tensor_bytes = data[4 + structure_len :]

    structure = json.loads(structure_bytes.decode("utf-8"))

    # Load tensors from safetensors blob
    tensor_map = {}
    if tensor_bytes:
        tensor_map = st_load_bytes(tensor_bytes)

    def _restore(o):
        if isinstance(o, dict):
            if "__tensor_ref__" in o:
                t = tensor_map[o["__tensor_ref__"]]
                # Restore original dtype
                dtype = getattr(torch, o["dtype"].replace("torch.", ""))
                return t.to(dtype=dtype)
            elif o.get("__dynamic_cache__"):
                cache = DynamicCache()
                cache.key_cache = _restore(o["key_cache"])
                cache.value_cache = _restore(o["value_cache"])
                cache._seen_tokens = o["_seen_tokens"]
                return cache
            elif o.get("__tuple__"):
                return tuple(_restore(v) for v in o["data"])
            else:
                return {k: _restore(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [_restore(v) for v in o]
        else:
            return o

    return _restore(structure)


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


from collections.abc import Mapping, Sequence


def print_output(x, name=None, indent=0, max_elements=5):
    def _tensor_preview(t, max_el=5):
        shape = tuple(t.shape)
        flat = t.flatten()

        if flat.numel() == 0:
            return f"shape={shape} EMPTY"

        vals = flat[:max_el].detach().cpu().tolist()

        if flat.numel() <= max_el:
            return f"shape={shape} vals={vals}"
        else:
            tail = flat[-max_el:].detach().cpu().tolist()
            return f"shape={shape} head={vals} tail={tail}"

    p = " " * indent
    prefix = f"{p}{name}: " if name else p

    # ---- Dict ----
    if isinstance(x, Mapping):
        print(f"{prefix}dict[{len(x)}]")
        for k, v in x.items():
            print_output(v, name=str(k), indent=indent + 2)

    # ---- Sequence ----
    elif isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        if len(x) == 0:
            print(f"{prefix}list[0]")
            return

        print(f"{prefix}list[{len(x)}]")

        # only show first element
        print_output(x[0], name="[0]", indent=indent + 2)
        if len(x) > 1:
            print(f"{p}  ...")

    # ---- Tensor ----
    elif isinstance(x, torch.Tensor):
        preview = _tensor_preview(x, max_elements)
        print(f"{prefix}Tensor({preview}, dtype={x.dtype}, device={x.device})")

    # ---- Cache-like ----
    elif hasattr(x, "__class__") and "Cache" in x.__class__.__name__:
        cls = x.__class__.__name__

        summary = []
        for attr in ["seen_tokens", "is_compileable"]:
            if hasattr(x, attr):
                summary.append(f"{attr}={getattr(x, attr)}")

        print(f"{prefix}{cls}({', '.join(summary)})")

        # selectively inspect key/value cache
        for attr in ["key_cache", "value_cache"]:
            if hasattr(x, attr):
                val = getattr(x, attr)
                if isinstance(val, list) and len(val) > 0:
                    print(f"{p}  {attr}: list[{len(val)}]")
                    print_output(val[0], name=f"{attr}[0]", indent=indent + 4)

    # ---- Fallback ----
    else:
        print(f"{prefix}{type(x).__name__}({x})")
