from tensorlink.ml.utils import bytes_to_tensor
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
from typing import Any, Callable, Dict, List, Optional, Tuple
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
                local_files_only=True,
            )
        except Exception:
            return None

    def load_model_path(self, model_name):
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

        entries.sort(key=lambda x: x[1])

        freed = 0
        for path, _, size in entries:
            shutil.rmtree(path, ignore_errors=True)
            freed += size
            if freed >= required_bytes:
                break


# ---------------------------------------------------------------------------
# Universal prefix resolution
# ---------------------------------------------------------------------------


class TiedLinear(nn.Module):
    """Linear projection using a weight tensor tied to an Embedding."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=weight.requires_grad)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


def _iter_safetensor_keys(model_path: str):
    """Yield all weight keys from safetensors shards (or .bin as fallback)."""
    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if safetensor_files:
        for shard_path in safetensor_files:
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                yield from f.keys()
            return  # one shard is enough to discover prefixes
    else:
        bin_files = glob.glob(os.path.join(model_path, "pytorch_model*.bin"))
        if bin_files:
            shard = torch.load(bin_files[0], map_location="cpu")
            yield from shard.keys()


def resolve_weight_prefix(model_path: str, module_path: str) -> str:
    """
    Discover the actual key prefix used in the weight files for *module_path*.

    Works by progressively stripping leading path components until a prefix
    is found that actually exists in the weight files. This is entirely
    architecture-agnostic — no hardcoded model family names.

    Examples (Qwen3 weight keys: "model.embed_tokens.weight", "lm_head.weight"):
      module_path="model.model"              -> "model"
      module_path="model.model.embed_tokens" -> "model.embed_tokens"
      module_path="model.lm_head"            -> "lm_head"
      module_path="model"                    -> "model"

    Returns the matched prefix string, or "" if nothing matched (caller
    should treat "" as a root/full load).
    """
    sampled: List[str] = []
    try:
        for key in _iter_safetensor_keys(model_path):
            sampled.append(key)
            if len(sampled) >= 256:
                break
    except Exception:
        return ""

    if not sampled:
        return ""

    # Build candidates by dropping 0, 1, 2, … leading components.
    # e.g. "model.model.embed_tokens" ->
    #   ["model.model.embed_tokens", "model.embed_tokens", "embed_tokens"]
    parts = module_path.split(".")
    for i in range(len(parts)):
        candidate = ".".join(parts[i:])
        if any(key.startswith(candidate + ".") for key in sampled):
            return candidate

    # Nothing matched — signal a root/full load
    return ""


def _strip_to_local_key(key: str, matched_prefix: str) -> str:
    """
    Strip *matched_prefix* (and a trailing dot) from *key*, returning the
    local parameter name suitable for load_state_dict on the target module.

    Examples:
        key="model.layers.3.mlp.gate_proj.weight", prefix="model.layers.3"
          -> "mlp.gate_proj.weight"
        key="layers.3.mlp.gate_proj.weight",        prefix="layers.3"
          -> "mlp.gate_proj.weight"
    """
    return key[len(matched_prefix) + 1 :]  # +1 for the "."


# ---------------------------------------------------------------------------
# Core shared loading primitives
# ---------------------------------------------------------------------------


def apply_required_buffers(
    module: nn.Module,
    module_info: Dict[str, Any],
    log_fn: Callable[[str], None] = logging.debug,
    warn_fn: Callable[[str], None] = logging.warning,
    error_fn: Callable[[str], None] = logging.error,
) -> None:
    """
    Apply buffers that cannot be recovered from safetensors weights alone.

    Buffer keys in module_info may be stored relative to the model root
    (e.g. ``"model.layers.0.rotary_emb.inv_freq"``) or relative to the
    target module (e.g. ``"rotary_emb.inv_freq"``).  We normalise to the
    latter before navigating into *module*.

    Handles three cases:
      1. Buffer already exists on module (overwrite in-place or re-register on
         shape mismatch)
      2. Buffer missing (register_buffer)
      3. Nested buffer path (navigate to the correct submodule first)
    """
    required_buffers = module_info.get("required_buffers", {})
    if not required_buffers or required_buffers == b"{}":
        return

    # The module_path tells us what prefix to strip from absolute buffer keys
    module_path = module_info.get("module_path", "")
    # Build candidate prefixes to strip (same logic as weight prefix resolution)
    strip_candidates = set()
    if module_path:
        strip_candidates.add(module_path + ".")
        # Also try without leading "model." in case keys were stored differently
        bare = module_path
        for segment in ("model", "transformer", "encoder", "decoder"):
            if bare.startswith(segment + "."):
                bare = bare[len(segment) + 1 :]
                strip_candidates.add(bare + ".")

    for key, buf_spec in required_buffers.items():
        try:
            if isinstance(buf_spec, str):
                tensor = bytes_to_tensor(buf_spec.encode())
            else:
                tensor = bytes_to_tensor(buf_spec)

            rel_key = key
            for strip in strip_candidates:
                if rel_key.startswith(strip):
                    rel_key = rel_key[len(strip) :]
                    break

            parts = rel_key.rsplit(".", 1)
            if len(parts) == 2:
                submodule_path, buf_name = parts
                try:
                    target = get_nested_module(module, submodule_path)
                except Exception:
                    target = None
                    for mod_name, mod in module.named_modules():
                        if mod_name == submodule_path:
                            target = mod
                            break
                if target is None:
                    warn_fn(
                        f"Buffer submodule not found: '{submodule_path}' (key '{key}'), skipping"
                    )
                    continue
            else:
                target = module
                buf_name = rel_key

            existing_buffers = dict(target.named_buffers(recurse=False))
            if hasattr(target, buf_name) and buf_name in existing_buffers:
                existing = getattr(target, buf_name)
                if existing.shape == tensor.shape:
                    existing.copy_(tensor)
                else:
                    target.register_buffer(buf_name, tensor)
            else:
                target.register_buffer(buf_name, tensor)

            log_fn(f"Applied buffer '{rel_key}'")

        except Exception as e:
            error_fn(f"Failed to apply buffer '{key}': {e}")


def _load_tensors_from_shards(
    model_path: str,
    prefix: str,
    log_fn: Callable[[str], None] = logging.debug,
) -> Dict[str, torch.Tensor]:
    """
    Low-level: scan all safetensors (or .bin) shards in *model_path* and
    return every tensor whose key starts with ``prefix + "."``, stripped to
    its local name.

    When *prefix* is ``""`` (root / whole-model load) every key is returned
    as-is with no stripping.
    """
    state_dict: Dict[str, torch.Tensor] = {}
    root_load = prefix == ""
    full_prefix = "" if root_load else (prefix + ".")

    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    if safetensor_files:
        for shard_path in safetensor_files:
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                keys_loaded = 0
                for key in f.keys():
                    if root_load or key.startswith(full_prefix):
                        local_key = (
                            key if root_load else _strip_to_local_key(key, prefix)
                        )
                        state_dict[local_key] = f.get_tensor(key)
                        keys_loaded += 1
                if keys_loaded:
                    log_fn(f"  {os.path.basename(shard_path)}: {keys_loaded} tensors")
            gc.collect()

    else:
        bin_files = glob.glob(os.path.join(model_path, "pytorch_model*.bin"))
        if not bin_files:
            raise ValueError(f"No weight files found in {model_path}")
        for bin_path in bin_files:
            shard = torch.load(bin_path, map_location="cpu")
            for key, value in shard.items():
                if root_load or key.startswith(full_prefix):
                    local_key = key if root_load else _strip_to_local_key(key, prefix)
                    state_dict[local_key] = value

    return state_dict


# ---------------------------------------------------------------------------
# Universal single-module weight loader
# ---------------------------------------------------------------------------


def load_module_weights(
    model_path: str,
    module_path: str,
    target_module: nn.Module,
    module_info: Optional[Dict[str, Any]] = None,
    device: torch.device = torch.device("cpu"),
    log_fn: Callable[[str], None] = logging.debug,
    warn_fn: Callable[[str], None] = logging.warning,
) -> Tuple[List[str], List[str]]:
    """
    Universal weight loader for a single module.  Used by both module.py
    (host loading) and worker.py (offloaded loading).

    Resolves the actual weight-file prefix for *module_path* automatically,
    so no hard-coded "model." trimming or ``single`` flag is needed.

    Steps:
      1. Resolve the real prefix used in the weight files.
      2. Extract the matching tensors (stripped to local names).
      3. Materialise the module on CPU with empty weights.
      4. Load the state dict (strict=False so tied / missing weights don't crash).
      5. Apply required buffers from *module_info* if provided.
      6. Move to *device*.

    Returns (missing_keys, unexpected_keys) from load_state_dict.
    """
    log_fn(f"Resolving weight prefix for '{module_path}'")
    prefix = resolve_weight_prefix(model_path, module_path)

    if prefix is None:
        warn_fn(
            f"Could not resolve weight prefix for '{module_path}' — "
            f"falling back to full scan. Module may load with missing keys."
        )
        # Last resort: load everything and let load_state_dict filter
        state_dict = _fallback_full_scan(model_path, module_path, log_fn, warn_fn)
    else:
        log_fn(f"Resolved prefix '{prefix}' for module '{module_path}'")
        state_dict = _load_tensors_from_shards(model_path, prefix, log_fn)

    log_fn(f"Loaded {len(state_dict)} tensors for '{module_path}'")

    # Materialise on CPU before loading weights
    target_module = target_module.to_empty(device="cpu")

    missing_keys, unexpected_keys = target_module.load_state_dict(
        state_dict, strict=False
    )

    del state_dict
    gc.collect()

    if missing_keys:
        warn_fn(f"Missing keys for '{module_path}': {missing_keys}")
    if unexpected_keys:
        warn_fn(f"Unexpected keys for '{module_path}': {unexpected_keys}")

    if module_info:
        apply_required_buffers(target_module, module_info, log_fn, warn_fn)

    target_module = target_module.to(device)
    return missing_keys, unexpected_keys


# ---------------------------------------------------------------------------
# Universal grouped-layer weight loader
# ---------------------------------------------------------------------------


def load_grouped_module_weights(
    model_path: str,
    layer_paths: List[str],
    target_module: nn.Module,
    module_info: Optional[Dict[str, Any]] = None,
    device: torch.device = torch.device("cpu"),
    log_fn: Callable[[str], None] = logging.debug,
    warn_fn: Callable[[str], None] = logging.warning,
) -> None:
    """
    Universal weight loader for a LayerGroupModule (encoder/decoder stacks,
    nn.ModuleList loops, etc.).  Used by both module.py and worker.py.

    Maps HF layer paths -> local ``layers.<idx>`` indices inside the wrapper,
    using the same universal prefix resolution so any model architecture works.

    Mutates *target_module* in-place via ``load_state_dict(strict=False)``
    per shard to keep peak memory low.
    """
    if not layer_paths:
        raise ValueError("layer_paths must be non-empty for grouped layer loading")

    # Resolve real prefixes for every layer path up front (one probe per layer)
    resolved: List[Optional[str]] = []
    for lp in layer_paths:
        prefix = resolve_weight_prefix(model_path, lp)
        if prefix is None:
            warn_fn(f"Could not resolve prefix for layer '{lp}', will skip")
        resolved.append(prefix)

    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if not safetensor_files:
        raise RuntimeError(
            f"No safetensors found in {model_path} for grouped layer loading"
        )

    target_module = target_module.to_empty(device="cpu")

    loaded_keys: List[str] = []
    missing_keys_set = set(target_module.state_dict().keys())

    for shard_idx, shard_path in enumerate(safetensor_files):
        log_fn(f"Loading shard {shard_idx + 1}/{len(safetensor_files)}")
        shard_state_dict: Dict[str, torch.Tensor] = {}

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                for local_idx, (layer_path, prefix) in enumerate(
                    zip(layer_paths, resolved)
                ):
                    if prefix is None:
                        continue
                    full_prefix = prefix + "."
                    if key.startswith(full_prefix):
                        subkey = _strip_to_local_key(key, prefix)
                        new_key = f"layers.{local_idx}.{subkey}"
                        shard_state_dict[new_key] = f.get_tensor(key)
                        loaded_keys.append(new_key)
                        missing_keys_set.discard(new_key)
                        break

        if shard_state_dict:
            target_module.load_state_dict(shard_state_dict, strict=False)
            del shard_state_dict
            gc.collect()

    if missing_keys_set:
        warn_fn(f"Missing keys after grouped load: {missing_keys_set}")

    log_fn(
        f"Loaded {len(loaded_keys)} tensors across "
        f"{len(safetensor_files)} shards for {len(layer_paths)} layers"
    )

    if module_info:
        apply_required_buffers(target_module, module_info, log_fn, warn_fn)

    target_module.to(device)


def _fallback_full_scan(
    model_path: str,
    module_path: str,
    log_fn: Callable[[str], None],
    warn_fn: Callable[[str], None],
) -> Dict[str, torch.Tensor]:
    """
    When prefix resolution fails completely, scan all weight keys using the
    same progressive-strip logic to find the best matching prefix.
    """
    warn_fn(f"Full scan fallback for '{module_path}'")

    # Reuse the same candidate logic as resolve_weight_prefix
    sampled: List[str] = []
    try:
        for key in _iter_safetensor_keys(model_path):
            sampled.append(key)
            if len(sampled) >= 256:
                break
    except Exception:
        pass

    matched_prefix = ""
    parts = module_path.split(".")
    for i in range(len(parts)):
        candidate = ".".join(parts[i:])
        if any(key.startswith(candidate + ".") for key in sampled):
            matched_prefix = candidate
            break

    state_dict: Dict[str, torch.Tensor] = {}
    safetensor_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    if not safetensor_files:
        bin_files = glob.glob(os.path.join(model_path, "pytorch_model*.bin"))
        if not bin_files:
            raise ValueError(f"No weight files in {model_path}")
        for bf in bin_files:
            shard = torch.load(bf, map_location="cpu")
            for key, val in shard.items():
                if not matched_prefix or key.startswith(matched_prefix + "."):
                    local_key = (
                        key[len(matched_prefix) + 1 :] if matched_prefix else key
                    )
                    state_dict[local_key] = val
        return state_dict

    for shard_path in safetensor_files:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if not matched_prefix or key.startswith(matched_prefix + "."):
                    local_key = (
                        key[len(matched_prefix) + 1 :] if matched_prefix else key
                    )
                    state_dict[local_key] = f.get_tensor(key)
        gc.collect()

    log_fn(f"Fallback scan found {len(state_dict)} tensors (prefix='{matched_prefix}')")
    return state_dict


# ---------------------------------------------------------------------------
# Full-model loaders (unchanged)
# ---------------------------------------------------------------------------


def load_full_model(
    model_name: str,
    model_type: str,
    device: torch.device,
    log_fn: Callable[[str], None] = logging.debug,
    torch_dtype: torch.dtype = torch.float16,
) -> nn.Module:
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
    model_config = AutoConfig.from_pretrained(model_name)

    with init_empty_weights():
        if model_type in ("causal", "chat"):
            skeleton_model = AutoModelForCausalLM.from_config(model_config)
        elif model_type == "seq2seq":
            skeleton_model = AutoModelForSeq2SeqLM.from_config(model_config)
        elif model_type == "audio2text":
            skeleton_model = AutoModelForSpeechSeq2Seq.from_config(model_config)
        else:
            skeleton_model = AutoModel.from_config(model_config)

    skeleton_model.eval()

    for param in skeleton_model.parameters():
        param.requires_grad = False

    return skeleton_model


# ---------------------------------------------------------------------------
# Backwards-compatible
# ---------------------------------------------------------------------------


def load_weights_for_paths(
    model_path: str,
    layer_paths: List[str],
    log_fn: Callable[[str], None] = logging.debug,
    warn_fn: Callable[[str], None] = logging.warning,
) -> Dict[str, torch.Tensor]:
    """
    Shim: resolve the best prefix for the *first* layer_path and return
    a raw state dict.  Prefer calling load_module_weights() directly.
    """
    if not layer_paths:
        return {}

    # Use the universal resolver for every path and merge
    state_dict: Dict[str, torch.Tensor] = {}
    for lp in layer_paths:
        prefix = resolve_weight_prefix(model_path, lp)
        if prefix is None:
            warn_fn(f"load_weights_for_paths: could not resolve prefix for '{lp}'")
            partial = _fallback_full_scan(model_path, lp, log_fn, warn_fn)
        else:
            partial = _load_tensors_from_shards(model_path, prefix, log_fn)
        state_dict.update(partial)

    log_fn(f"load_weights_for_paths: {len(state_dict)} tensors total")
    return state_dict


def load_grouped_layer_weights(
    model_path: str,
    layer_paths: List[str],
    target_module: nn.Module,
    log_fn: Callable[[str], None] = logging.debug,
    warn_fn: Callable[[str], None] = logging.warning,
) -> None:
    """Shim: delegates to load_grouped_module_weights."""
    load_grouped_module_weights(
        model_path,
        layer_paths,
        target_module,
        module_info=None,
        device=torch.device("cpu"),
        log_fn=log_fn,
        warn_fn=warn_fn,
    )


def get_nested_module(
    model: torch.nn.Module, path: str, target_class_name: str = None
) -> torch.nn.Module:
    parts = path.split('.')
    current = model

    for i in range(len(parts)):
        part = parts[i]

        if part == "":
            continue

        if part == "model":
            # Only skip if there are further parts AND the next part
            # is accessible without going through .model explicitly
            has_more = len(parts) > i + 1
            if has_more and hasattr(current, "model"):
                next_part = parts[i + 1]
                if hasattr(current, next_part):
                    continue
            elif not has_more:
                pass
            else:
                continue

        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)

    return current
