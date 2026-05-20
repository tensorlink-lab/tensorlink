import importlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Dict
import time
import os
from safetensors.torch import save as st_save_bytes, load as st_load_bytes
import psutil
import torch
import torch.nn as nn
from dataclasses import is_dataclass, asdict
from transformers.utils import ModelOutput
from transformers.cache_utils import DynamicCache


MODELS_CACHE_PATH = "logs/models.json"
DTYPE_STR_MAP = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
DTYPE_OBJ_MAP = {v: k for k, v in DTYPE_STR_MAP.items()}


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


def handle_output(tensor):
    """
    Extract the primary tensor from a model output for backward pass storage
    and intermediate tracking ONLY. Do NOT use this to preprocess forward inputs.

    Returns the logits/last_hidden_state tensor, or the input unchanged if it's
    already a plain tensor or tuple of tensors.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor

    # ModelOutput: return logits or last_hidden_state for grad tracking
    if isinstance(tensor, ModelOutput):
        if hasattr(tensor, "logits") and isinstance(tensor.logits, torch.Tensor):
            return tensor.logits
        if hasattr(tensor, "last_hidden_state") and isinstance(
            tensor.last_hidden_state, torch.Tensor
        ):
            return tensor.last_hidden_state
        # Fallback: first tensor field
        for v in tensor.values():
            if isinstance(v, torch.Tensor):
                return v

    # Tuple/list: only unwrap if single-element containing a tensor
    # Do NOT flatten multi-element tuples — they may be (input_ids, attention_mask, ...)
    if isinstance(tensor, (tuple, list)):
        tensors = [t for t in tensor if isinstance(t, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
        return type(tensor)(tensors) if tensors else tensor

    if isinstance(tensor, dict):
        for key in ["logits", "last_hidden_state"]:
            if key in tensor and isinstance(tensor[key], torch.Tensor):
                return tensor[key]
        for v in tensor.values():
            if isinstance(v, torch.Tensor):
                return v

    raise ValueError(f"handle_output: unsupported type {type(tensor)}")


def detach_tensor(obj, clone: bool = False):
    """
    Recursively detach (and optionally clone) all tensors to CPU.
    Preserves structure: ModelOutput, DynamicCache, tuple, list, dict.
    Passes unsupported types through unchanged.
    """
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        return t.clone() if clone else t

    if isinstance(obj, DynamicCache):
        keys, vals = _get_cache_kv(obj)
        new_cache = DynamicCache()
        for i, (k, v) in enumerate(zip(keys, vals)):
            if k is not None and v is not None:
                new_cache.update(
                    detach_tensor(k, clone=clone),
                    detach_tensor(v, clone=clone),
                    i,
                )
        return new_cache

    if isinstance(obj, ModelOutput):
        return obj.__class__(
            **{
                k: detach_tensor(v, clone=clone) if v is not None else v
                for k, v in obj.items()
            }
        )

    if isinstance(obj, (list, tuple)):
        result = [detach_tensor(v, clone=clone) for v in obj]
        return type(obj)(result)

    if isinstance(obj, dict):
        return {k: detach_tensor(v, clone=clone) for k, v in obj.items()}

    # Primitives, None, etc. are passed through
    return obj


def attach_tensor(obj, device):
    """
    Recursively move all tensors to device.
    Preserves structure: ModelOutput, DynamicCache, tuple, list, dict.
    Passes unsupported types through unchanged.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    if isinstance(obj, DynamicCache):
        keys, vals = _get_cache_kv(obj)
        new_cache = DynamicCache()
        for i, (k, v) in enumerate(zip(keys, vals)):
            if k is not None and v is not None:
                new_cache.update(k.to(device), v.to(device), i)
        return new_cache

    if isinstance(obj, ModelOutput):
        return obj.__class__(
            **{
                k: attach_tensor(v, device) if v is not None else v
                for k, v in obj.items()
            }
        )

    if isinstance(obj, (list, tuple)):
        result = [attach_tensor(v, device) for v in obj]
        return type(obj)(result)

    if isinstance(obj, dict):
        return {k: attach_tensor(v, device) for k, v in obj.items()}

    # Primitives, None, etc. are passed through
    return obj


def enable_grad(obj):
    """
    Enable gradients on floating-point tensors within any supported structure.
    Integer/bool tensors are passed through unchanged (they can't have grads).
    """
    if isinstance(obj, torch.Tensor):
        if obj.is_floating_point():
            return obj.detach().clone().requires_grad_(True)
        return obj  # int/bool tensors: no grad, no error

    if isinstance(obj, ModelOutput):
        return obj.__class__(
            **{k: enable_grad(v) if v is not None else v for k, v in obj.items()}
        )

    if isinstance(obj, DynamicCache):
        keys, vals = _get_cache_kv(obj)
        new_cache = DynamicCache()
        for i, (k, v) in enumerate(zip(keys, vals)):
            if k is not None and v is not None:
                new_cache.update(enable_grad(k), enable_grad(v), i)
        return new_cache

    if isinstance(obj, (list, tuple)):
        result = [enable_grad(v) for v in obj]
        return type(obj)(result)

    if isinstance(obj, dict):
        return {k: enable_grad(v) for k, v in obj.items()}

    return obj


def tensor_to_bytes(obj):
    """
    Safe serialization: safetensors for tensors, JSON for structure.
    No pickle, no arbitrary code execution.

    Supports:
    - torch.Tensor
    - HF ModelOutput
    - dataclasses
    - dict/list/tuple
    - DynamicCache
    - custom classes with __dict__
    """

    tensor_map = {}
    counter = [0]

    def _extract_tensors(o):
        # Tensors
        if isinstance(o, torch.Tensor):
            key = f"__tensor_{counter[0]}__"
            counter[0] += 1
            tensor_map[key] = o.detach().cpu().contiguous().clone()
            return {
                "__tensor_ref__": key,
                "dtype": DTYPE_STR_MAP[o.dtype],
                "shape": list(o.shape),
                "requires_grad": o.requires_grad,
            }

        # HF model outputs
        elif isinstance(o, ModelOutput):
            return {
                "__hf_model_output__": True,
                "module": o.__class__.__module__,
                "class": o.__class__.__name__,
                "data": _extract_tensors(dict(o)),
            }

        # HF DynamicCache
        elif isinstance(o, DynamicCache):
            keys, vals = _get_cache_kv(o)
            populated = [
                (i, k, v)
                for i, (k, v) in enumerate(zip(keys, vals))
                if k is not None and v is not None
            ]
            seen = o.get_seq_length() if hasattr(o, "get_seq_length") else 0

            return {
                "__dynamic_cache__": True,
                "key_cache": _extract_tensors([k for _, k, _ in populated]),
                "value_cache": _extract_tensors([v for _, _, v in populated]),
                "layer_indices": [i for i, _, _ in populated],
                "_seen_tokens": seen,
            }

        # Dataclasses
        elif is_dataclass(o):
            return {
                "__dataclass__": True,
                "module": o.__class__.__module__,
                "class": o.__class__.__name__,
                "data": _extract_tensors(asdict(o)),
            }

        # Custom classes
        elif hasattr(o, "__dict__"):
            return {
                "__custom_object__": True,
                "module": o.__class__.__module__,
                "class": o.__class__.__name__,
                "data": _extract_tensors(o.__dict__),
            }

        # dicts
        elif isinstance(o, dict):
            return {k: _extract_tensors(v) for k, v in o.items()}

        # List/tuple
        elif isinstance(o, (list, tuple)):
            result = [_extract_tensors(v) for v in o]
            if isinstance(o, tuple):
                return {
                    "__tuple__": True,
                    "data": result,
                }
            return result

        # Primitives
        elif isinstance(o, (int, float, bool, str, type(None))):
            return o

        raise TypeError(f"Unsupported serialization type: {type(o)}")

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

    tensor_map = {}

    if tensor_bytes:
        tensor_map = st_load_bytes(tensor_bytes)

    def _restore(o):
        # Lists
        if isinstance(o, list):
            return [_restore(v) for v in o]

        # Dicts & other objects
        elif isinstance(o, dict):
            # Tensors
            if "__tensor_ref__" in o:
                t = tensor_map[o["__tensor_ref__"]]
                dtype_str = o["dtype"]
                if dtype_str not in DTYPE_OBJ_MAP:
                    raise ValueError(f"Unsupported dtype: {dtype_str}")
                t = t.to(dtype=DTYPE_OBJ_MAP[dtype_str])
                t.requires_grad_(o.get("requires_grad", False))
                return t

            # Tuple
            elif o.get("__tuple__"):

                return tuple(_restore(v) for v in o["data"])

            # HF model output
            elif o.get("__hf_model_output__"):
                module = importlib.import_module(o["module"])
                cls = getattr(module, o["class"])
                restored_data = _restore(o["data"])
                return cls(**restored_data)

            # Dyncamic Cache
            elif o.get("__dynamic_cache__"):
                cache = DynamicCache()
                key_layers = _restore(o["key_cache"])
                val_layers = _restore(o["value_cache"])
                layer_indices = o.get("layer_indices", list(range(len(key_layers))))

                for layer_idx, k, v in zip(
                    layer_indices,
                    key_layers,
                    val_layers,
                ):
                    if k is not None and v is not None:
                        cache.update(
                            k,
                            v,
                            layer_idx,
                        )

                return cache

            # Dataclass
            elif o.get("__dataclass__"):
                module = importlib.import_module(o["module"])
                cls = getattr(module, o["class"])
                restored_data = _restore(o["data"])

                return cls(**restored_data)

            # Custom classes
            elif o.get("__custom_object__"):
                module = importlib.import_module(o["module"])

                cls = getattr(module, o["class"])

                restored_data = _restore(o["data"])
                obj = cls.__new__(cls)
                obj.__dict__.update(restored_data)

                return obj

            return {k: _restore(v) for k, v in o.items()}

        return o

    return _restore(structure)


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


def resolve_module_from_path(model: nn.Module, path: str):
    """Return (parent_module, child_module, child_name)."""
    parts = path.split(".")
    parent = model
    for i in range(len(parts[:-1])):
        p = parts[i]
        if i == 0 and p == "model":
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


def _get_cache_kv(cache):
    """
    Extract (key_list, value_list) from a DynamicCache regardless of
    transformers version. New versions use cache.layers[i].keys/.values,
    old versions expose cache.key_cache / cache.value_cache directly.
    """
    if hasattr(cache, 'key_cache'):
        return cache.key_cache, cache.value_cache

    elif hasattr(cache, 'layers'):
        if not cache.layers:
            return [], []

        # Probe attribute names on first layer
        layer = cache.layers[0]
        key_attr = 'keys' if hasattr(layer, 'keys') else 'key'
        val_attr = 'values' if hasattr(layer, 'values') else 'value'

        keys, vals = [], []
        for l in cache.layers:
            k = getattr(l, key_attr, None)
            v = getattr(l, val_attr, None)
            keys.append(k)
            vals.append(v)

        return keys, vals
    else:
        raise TypeError(f"Unrecognised DynamicCache layout: {list(vars(cache).keys())}")


def debug_structure(obj, name="root", indent=0, max_depth=6, visited=None):
    """
    Recursively prints structure/types/shapes of nested objects.
    Useful for debugging transformer/model outputs.
    """

    if visited is None:
        visited = set()

    prefix = "  " * indent

    # Prevent recursive loops
    obj_id = id(obj)
    if obj_id in visited:
        print(f"{prefix}{name}: <recursive reference>")
        return

    visited.add(obj_id)

    # Depth limit
    if indent > max_depth:
        print(f"{prefix}{name}: <max depth reached>")
        return

    # Tensors
    if isinstance(obj, torch.Tensor):
        print(
            f"{prefix}{name}: "
            f"Tensor(shape={tuple(obj.shape)}, "
            f"dtype={obj.dtype}, "
            f"device={obj.device}, "
            f"requires_grad={obj.requires_grad})"
        )

    # Dict-like
    elif isinstance(obj, Mapping):
        print(f"{prefix}{name}: dict[{len(obj)}]")
        for k, v in obj.items():
            debug_structure(
                v,
                name=f"[{repr(k)}]",
                indent=indent + 1,
                max_depth=max_depth,
                visited=visited,
            )

    # List/Tuple
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        print(f"{prefix}{name}: {type(obj).__name__}[{len(obj)}]")

        for i, v in enumerate(obj):
            debug_structure(
                v,
                name=f"[{i}]",
                indent=indent + 1,
                max_depth=max_depth,
                visited=visited,
            )

    # HF model outputs / dataclasses / custom objects
    elif hasattr(obj, "__dict__"):
        print(f"{prefix}{name}: {type(obj).__name__}")

        for k, v in vars(obj).items():
            debug_structure(
                v, name=f".{k}", indent=indent + 1, max_depth=max_depth, visited=visited
            )

    # Primitive
    else:
        value = repr(obj)

        # Truncate huge reprs
        if len(value) > 120:
            value = value[:120] + "..."

        print(f"{prefix}{name}: {type(obj).__name__} = {value}")
