import time
import psutil
import torch
import torch.nn as nn
import platform
import subprocess
import json
import os
from typing import Optional, Tuple, Dict


DEVICE_PROFILE_PATH = "logs/device_profile.json"
PROFILE_VERSION = 1


def _load_cached_profile() -> Optional[Dict]:
    """Load a previously saved device profile from disk, if any."""
    if not os.path.exists(DEVICE_PROFILE_PATH):
        return None
    try:
        with open(DEVICE_PROFILE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_profile(device_info: Dict, device_benchmark: Optional[Dict]) -> None:
    """Persist device info + benchmark so future startups can skip
    re-running the benchmark on unchanged hardware."""
    try:
        os.makedirs(os.path.dirname(DEVICE_PROFILE_PATH), exist_ok=True)
        with open(DEVICE_PROFILE_PATH, "w") as f:
            json.dump(
                {
                    "profile_version": PROFILE_VERSION,
                    "device_info": device_info,
                    "device_benchmark": device_benchmark,
                    "benchmarked_at": time.time(),
                },
                f,
                indent=4,
            )
    except OSError:
        pass  # best-effort -- a failed write just means we re-benchmark next run


def _profile_matches_current_hardware(cached: Dict, current_info: Dict) -> bool:
    """
    A cache hit means 'this is the same physical device and the same
    benchmark version', not 'nothing has changed since'. Compares only
    stable identity fields -- not memory usage, which fluctuates.
    """
    if cached.get("profile_version") != PROFILE_VERSION:
        return False

    cached_info = cached.get("device_info") or {}
    identity_fields = ("backend", "name")
    return all(
        cached_info.get(field) == current_info.get(field) for field in identity_fields
    )


def run_device_benchmark(force: bool = False) -> Tuple[Dict, Optional[Dict]]:
    """
    Run device detection + compute/bandwidth benchmark once, in the main
    process, caching the result to disk so subsequent startups on the
    same hardware don't re-run the (multi-second) benchmark.

    Parameters
    ----------
    force : bool
        If True, ignore any cached result and re-benchmark unconditionally.
    """
    current_info = get_gpu_info()

    if not force:
        cached = _load_cached_profile()
        if cached and _profile_matches_current_hardware(cached, current_info):
            return cached["device_info"], cached["device_benchmark"]

    device_benchmark = None
    if current_info.get("available"):
        device_benchmark = {
            "compute": benchmark_matmul(),
            "memory": benchmark_memory_bandwidth(),
        }

    _save_profile(current_info, device_benchmark)
    return current_info, device_benchmark


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
    Returns available memory in bytes. Gets the total free CUDA VRAM if available.
    Falls back to available system RAM if CUDA is not available. If max_vram_gb
    is provided, caps the returned memory.
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


def _detect_backend():
    """Return one of: 'cuda', 'rocm', 'mps', 'cpu'."""
    if torch.cuda.is_available():
        # ROCm builds of PyTorch expose themselves through the same
        # torch.cuda.* namespace as CUDA. torch.version.hip is only set
        # on ROCm builds, so it's the reliable way to tell them apart.
        if getattr(torch.version, "hip", None):
            return "rocm"
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _nvidia_driver_version():
    """Best-effort driver version via nvidia-smi; None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            timeout=3,
        )
        return out.decode().strip().splitlines()[0]
    except Exception:
        return None


def _rocm_driver_version():
    """Best-effort driver/ROCm version via rocm-smi; None if unavailable."""
    try:
        out = subprocess.check_output(["rocm-smi", "--showdriverversion"], timeout=3)
        return out.decode().strip()
    except Exception:
        return None


def get_gpu_info():
    backend = _detect_backend()

    if backend in ("cuda", "rocm"):
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)

        info = {
            "available": True,
            "backend": backend,
            "name": torch.cuda.get_device_name(idx),
            "total_memory_gb": round(props.total_memory / (1024**3), 2),
            "multi_processor_count": props.multi_processor_count,
        }

        if backend == "cuda":
            info["compute_capability"] = f"{props.major}.{props.minor}"
            info["cuda_version"] = torch.version.cuda
            info["driver_version"] = _nvidia_driver_version()
        else:  # rocm
            # major/minor on ROCm map to gfx-arch fields, not a CUDA
            # capability, so labeling them "compute_capability" is misleading.
            info["gcn_arch"] = f"{props.major}.{props.minor}"
            info["rocm_version"] = torch.version.hip
            info["driver_version"] = _rocm_driver_version()

        return info

    if backend == "mps":
        return {
            "available": True,
            "backend": "mps",
            "name": platform.processor() or "Apple Silicon GPU",
            # MPS doesn't expose per-device memory/props like CUDA does;
            # this is total unified system memory, not a dedicated VRAM pool.
            "total_memory_gb": None,
            "note": "Apple Silicon shares unified memory with the CPU; "
            "no per-device VRAM figure is available via PyTorch.",
        }

    return {"available": False, "backend": "cpu"}


def _synchronize(backend):
    if backend in ("cuda", "rocm"):
        torch.cuda.synchronize()
    elif backend == "mps":
        torch.mps.synchronize()
    # no-op for cpu


def _pick_matmul_dtype(backend, requested):
    """
    Fall back to a supported dtype rather than letting matmul raise.
    fp16 matmul on CPU is often unsupported/slow; MPS support for fp16/bf16
    varies by torch version, so we probe once and downgrade if it fails.
    """
    if backend == "cpu" and requested in (torch.float16, torch.bfloat16):
        return torch.float32
    return requested


def benchmark_matmul(size=8192, dtype=torch.float16, iters=20, warmup=5):
    backend = _detect_backend()
    device = "cuda" if backend in ("cuda", "rocm") else backend
    dtype = _pick_matmul_dtype(backend, dtype)

    try:
        a = torch.randn(size, size, dtype=dtype, device=device)
        b = torch.randn(size, size, dtype=dtype, device=device)
    except RuntimeError:
        # dtype unsupported on this backend/device combo at this size
        dtype = torch.float32
        a = torch.randn(size, size, dtype=dtype, device=device)
        b = torch.randn(size, size, dtype=dtype, device=device)

    for _ in range(warmup):
        c = a @ b
    _synchronize(backend)

    start = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    _synchronize(backend)
    elapsed = time.perf_counter() - start

    flops_per_matmul = 2 * (size**3)
    total_flops = flops_per_matmul * iters
    tflops = total_flops / elapsed / 1e12

    return {
        "backend": backend,
        "matrix_size": size,
        "dtype": str(dtype),
        "iterations": iters,
        "elapsed_sec": round(elapsed, 4),
        "tflops": round(tflops, 2),
    }


def benchmark_memory_bandwidth(size_mb=1024, iters=10):
    backend = _detect_backend()
    device = "cuda" if backend in ("cuda", "rocm") else backend

    n = (size_mb * 1024 * 1024) // 4  # float32 elements
    a = torch.randn(n, device=device)
    b = torch.empty_like(a)

    _synchronize(backend)
    start = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    _synchronize(backend)
    elapsed = time.perf_counter() - start

    bytes_moved = n * 4 * 2 * iters
    gbps = bytes_moved / elapsed / 1e9

    return {"backend": backend, "bandwidth_gb_s": round(gbps, 2)}
