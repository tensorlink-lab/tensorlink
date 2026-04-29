from tensorlink.ml.utils import bytes_to_tensor, get_nested_module

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoConfig,
)
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download, model_info
from safetensors import safe_open
from typing import Any, Callable, Dict, List, Optional
import torch.nn as nn
import torch
import logging
import shutil
import glob
import os
import gc


def has_space(required_bytes, path) -> bool:
    total, used, free = shutil.disk_usage(path)
    return free > required_bytes * 1.2


def get_hf_cache_dir() -> str:
    return os.environ.get(
        "HF_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    )


class ModelCacheManager:
    def __init__(self, primary_dir, max_cache_size_gb=50):
        self.primary_dir = primary_dir
        self.hf_cache_dir = get_hf_cache_dir()
        self.max_cache_size = max_cache_size_gb * 1024**3

    def get_local_snapshot(self, model_name):
        path = os.path.join(self.primary_dir, model_name.replace("/", "_"))
        return path if os.path.exists(path) else None

    def get_hf_cached(self, model_name):
        try:
            return snapshot_download(
                repo_id=model_name,
                cache_dir=self.hf_cache_dir,
                local_files_only=True,  # 🔑 key
            )

        except Exception:
            return None

    def load_model_path(self, model_name):
        """
        Get model path, first check local cache, then global, then download
        if not found. If download size is too large, remove the oldest cache.
        """
        local = self.get_local_snapshot(model_name)
        if local:
            return local

        cached = self.get_hf_cached(model_name)
        if cached:
            return cached

        size = self._estimate_model_size(model_name)
        if not has_space(size, self.hf_cache_dir):
            self.cleanup_hf_cache(size)

        return snapshot_download(
            repo_id=model_name,
            cache_dir=self.hf_cache_dir,
            allow_patterns=["*.safetensors", "*.bin", "*.json"],
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
            local_files_only=False,
        )

    @staticmethod
    def _estimate_model_size(model_name: str) -> int:
        try:
            info = model_info(model_name)
            return sum(s.size for s in info.siblings if s.size is not None)
        except Exception:
            return 0

    def cleanup_hf_cache(self, required_bytes: int) -> None:
        """Evict least-recently-used cache entries until *required_bytes* are freed."""
        entries = []
        for root, dirs, _ in os.walk(self.hf_cache_dir):
            for d in dirs:
                full_path = os.path.join(root, d)
                try:
                    stat = os.stat(full_path)
                    size = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, _, filenames in os.walk(full_path)
                        for f in filenames
                    )
                    entries.append((full_path, stat.st_atime, size))

                except Exception:
                    continue

        entries.sort(key=lambda x: x[1])  # oldest first

        freed = 0
        for path, _, size in entries:
            shutil.rmtree(path, ignore_errors=True)
            freed += size
            if freed >= required_bytes:
                break


def apply_required_buffers(
    module: nn.Module,
    module_info: Dict[str, Any],
    log_fn: Callable[[str], None] = logging.debug,
    warn_fn: Callable[[str], None] = logging.warning,
    error_fn: Callable[[str], None] = logging.error,
) -> None:
    """
    Apply buffers that cannot be recovered from safetensors weights alone.

    Handles three cases:
      1. Buffer already exists on module (overwrite in-place or re-register on
         shape mismatch)
      2. Buffer missing (register_buffer)
      3. Nested buffer path (navigate to the correct submodule first)
    """
    required_buffers = module_info.get("required_buffers", {})
    if not required_buffers or required_buffers == b"{}":
        return

    for key, buf_spec in required_buffers.items():
        try:
            # buf_spec may arrive as raw bytes or as an encoded string
            if isinstance(buf_spec, str):
                tensor = bytes_to_tensor(buf_spec.encode())
            else:
                tensor = bytes_to_tensor(buf_spec)

            # Navigate to the correct submodule if key is dotted
            parts = key.rsplit(".", 1)
            if len(parts) == 2:
                submodule_path, buf_name = parts
                try:
                    target = get_nested_module(module, submodule_path)

                except Exception:
                    warn_fn(
                        f"Buffer submodule not found: {submodule_path}, skipping {key}"
                    )
                    continue
            else:
                target = module
                buf_name = key

            existing_buffers = dict(target.named_buffers(recurse=False))
            if hasattr(target, buf_name) and buf_name in existing_buffers:
                existing = getattr(target, buf_name)
                if existing.shape == tensor.shape:
                    existing.copy_(tensor)
                else:
                    # Shape mismatch (e.g. seq_len changed) — re-register
                    target.register_buffer(buf_name, tensor)
            else:
                target.register_buffer(buf_name, tensor)

            log_fn(f"Applied buffer {key}")

        except Exception as e:
            error_fn(f"Failed to apply buffer {key}: {e}")


def load_weights_for_paths(
    model_path: str,
    layer_paths: List[str],
    log_fn: Callable[[str], None] = logging.debug,
    warn_fn: Callable[[str], None] = logging.warning,
    single: bool = False,
    base_model_prefix: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load safetensors weight shards from *model_path*, returning only the tensors
    that belong to any of the module paths in *layer_paths*.

    Key stripping rules:
      - The matched layer prefix (and a leading "model." variant) is removed.
      - If *single* is True and the remaining key still contains a ".", the
        first segment is also stripped (for single-module loads where the weight
        key encodes the class path rather than just the parameter name).

    Falls back to pytorch_model*.bin only when no safetensors shards exist.
    """
    state_dict: Dict[str, torch.Tensor] = {}

    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    if safetensor_files:
        log_fn(f"Found {len(safetensor_files)} safetensors shards in {model_path}")
        layer_path_to_idx = {p: i for i, p in enumerate(layer_paths)}

        for shard_path in safetensor_files:
            log_fn(f"Reading {os.path.basename(shard_path)}")
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys_loaded = 0
                for key in f.keys():
                    for layer_path in layer_path_to_idx:
                        layer_prefix = layer_path + "."

                        # Try progressively shorter prefixes (strip leading "model.")
                        matched_prefix = None
                        trimmed = layer_prefix
                        for _ in range(2):
                            if key.startswith(trimmed):
                                matched_prefix = trimmed
                                break
                            trimmed = trimmed.split("model.", 1)[-1]

                        if matched_prefix is None:
                            continue

                        new_key = key[len(matched_prefix) :]

                        if single and "." in new_key:
                            new_key = new_key.split(".", 1)[1]
                        elif len(new_key.split(".")) > 1:
                            new_key = key.split(".", 1)[1]

                        state_dict[new_key] = f.get_tensor(key)
                        keys_loaded += 1
                        break

                if keys_loaded:
                    log_fn(f"  Loaded {keys_loaded} tensors")

            gc.collect()

    else:
        # Fallback: .bin shards
        bin_files = glob.glob(os.path.join(model_path, "pytorch_model*.bin"))
        if not bin_files:
            raise ValueError(f"No weight files found in {model_path}")

        warn_fn("No safetensors found — falling back to .bin files")

        for bin_path in bin_files:
            shard_dict = torch.load(bin_path, map_location="cpu")

            # Full-model load: no filtering
            if layer_paths == ["model"]:
                state_dict.update(shard_dict)
                continue

            for key, value in shard_dict.items():
                for layer_path in layer_paths:
                    prefix = layer_path + "."
                    if key.startswith(prefix):
                        state_dict[key[len(prefix) :]] = value
                        break

    log_fn(f"Loaded {len(state_dict)} weight tensors total")
    return state_dict


def load_grouped_layer_weights(
    model_path: str,
    layer_paths: List[str],
    target_module: nn.Module,
    log_fn: Callable[[str], None] = logging.debug,
    warn_fn: Callable[[str], None] = logging.warning,
) -> None:
    """
    Load weights for a grouped LayerGroupModule directly into *target_module*,
    remapping HF layer prefixes to local ``layers.<idx>`` indices in-place.

    Mutates *target_module* via ``load_state_dict(strict=False)`` per shard.
    """
    layer_prefix_to_local_idx = {p: i for i, p in enumerate(layer_paths)}
    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    if not safetensor_files:
        raise RuntimeError("No safetensors found for grouped layer loading")

    loaded_keys: List[str] = []
    missing_keys = set(target_module.state_dict().keys())

    for shard_idx, shard_path in enumerate(safetensor_files):
        log_fn(f"Loading shard {shard_idx + 1}/{len(safetensor_files)}")
        shard_state_dict: Dict[str, torch.Tensor] = {}

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                for layer_prefix, local_idx in layer_prefix_to_local_idx.items():
                    full_prefix = layer_prefix + "."
                    # Try progressively shorter prefixes
                    matched = None
                    for _ in range(3):
                        if key.startswith(full_prefix):
                            matched = full_prefix
                            break
                        full_prefix = full_prefix.split(".", 1)[-1]

                    if matched is None:
                        continue

                    subkey = key[len(matched) :]
                    new_key = f"layers.{local_idx}.{subkey}"
                    shard_state_dict[new_key] = f.get_tensor(key)
                    loaded_keys.append(new_key)
                    missing_keys.discard(new_key)
                    break

        if shard_state_dict:
            target_module.load_state_dict(shard_state_dict, strict=False)
            del shard_state_dict
            gc.collect()

    if missing_keys:
        warn_fn(f"Missing keys after grouped load: {missing_keys}")

    log_fn(f"Loaded {len(loaded_keys)} tensors across {len(safetensor_files)} shards")


def load_full_model(
    model_name: str,
    model_type: str,
    device: torch.device,
    log_fn: Callable[[str], None] = logging.debug,
    torch_dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """
    Load a complete HuggingFace model with optimal memory usage.

    For single-GPU setups the model is loaded to CPU first and moved to *device*
    afterward. Multi-GPU setups use ``device_map="auto"``.
    """
    num_gpus = torch.cuda.device_count()
    load_kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch_dtype,
        "device_map": "auto" if num_gpus > 1 else "cpu",
    }

    log_fn(f"Loading full model {model_name} (type={model_type})")

    if model_type in ("causal", "chat"):
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    elif model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
    # elif model_type == "vision2text":
    #     model = AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)
    elif model_type == "audio2text":
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **load_kwargs)
    else:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, config=config, **load_kwargs)

    if num_gpus == 1 and device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        model = model.to(device)

    log_fn(f"Successfully loaded full model {model_name}")
    return model


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
        # elif model_type == "vision2text":
        #     skeleton_model = AutoModelForVision2Seq.from_config(model_config)
        elif model_type == "audio2text":
            skeleton_model = AutoModelForSpeechSeq2Seq.from_config(model_config)
        else:
            skeleton_model = AutoModel.from_config(model_config)

    skeleton_model.eval()  # Set to eval mode initially

    # Ensure no cached gradients or cached computations
    for param in skeleton_model.parameters():
        param.requires_grad = False

    return skeleton_model
