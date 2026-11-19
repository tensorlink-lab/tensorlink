"""
tensorlink/ml/pipeline/segment.py

Runs one compiled forward-program segment (on a worker, or on the host for the
optional pre/post privacy segments).

KV cache: the real cache lives on whichever node owns the layers. Across the wire it
is represented only by ``{"__cache__": session_id, "len": n}`` (n = tokens cached
BEFORE this step). Each node resolves the reference to its own resident
``DynamicCache`` (LRU per session) and, when it ran layers, advances its count by
``new_tokens``. Any Cache object found in the outputs is turned back into the
reference so nothing cache-shaped is ever serialized.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional

import torch
from transformers import Cache, DynamicCache
from transformers.utils import ModelOutput

from tensorlink.ml.utils.injector import RemoteKVCache

MAX_CACHE_SESSIONS = 8


def is_cache_ref(v) -> bool:
    return isinstance(v, dict) and "__cache__" in v and "len" in v


def attach(obj, device):
    """Like utils.attach_tensor, but never rebuilds a Cache (attach_tensor would turn the
    host's RemoteKVCache stub into a plain DynamicCache and lose it)."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, Cache):
        return obj
    if isinstance(obj, ModelOutput):
        return obj.__class__(
            **{k: attach(v, device) for k, v in obj.items() if v is not None}
        )
    if isinstance(obj, dict):
        return {k: attach(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(attach(v, device) for v in obj)
    return obj


class SegmentRunner:
    def __init__(self, model, program, code, device):
        self.model = model
        self.program = program
        self.code = code
        self.device = device
        self._fn = program.compile_segment(code)
        self.dtype = self._param_dtype()
        self._caches: "OrderedDict[str, list]" = OrderedDict()
        self.has_layers = code.has_layers
        self.final = code.final

    # ---- dtype handling ----------------------------------------------------
    def _param_dtype(self):
        counts: dict = {}
        for path in self.code.modules:
            obj = self.model
            for part in path.split("."):
                obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
            for p in obj.parameters():
                if p.is_floating_point() and not p.is_meta:
                    counts[p.dtype] = counts.get(p.dtype, 0) + 1
        self._dtype_counts = {str(k): v for k, v in counts.items()}
        return max(counts, key=counts.get) if counts else None

    def _cast(self, obj):
        """Activations must match the weights' dtype (HF assumes it); anything floating
        that arrives in another dtype (e.g. fp32 from a peer) is cast down/up to it."""
        if isinstance(obj, torch.Tensor):
            if (
                self.dtype is not None
                and obj.is_floating_point()
                and obj.dtype != self.dtype
            ):
                return obj.to(self.dtype)
            return obj
        if isinstance(obj, ModelOutput):
            return obj.__class__(
                **{k: self._cast(v) for k, v in obj.items() if v is not None}
            )
        if isinstance(obj, dict):
            return {k: self._cast(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._cast(v) for v in obj)
        return obj

    def _describe(self, obj, prefix="", out=None, depth=0):
        out = [] if out is None else out
        if isinstance(obj, torch.Tensor):
            out.append(
                f"{prefix}:{str(obj.dtype).replace('torch.', '')}{list(obj.shape)}"
            )
        elif isinstance(obj, dict) and depth < 2:
            for k, v in obj.items():
                self._describe(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
        return out

    # ---- cache handling ---------------------------------------------------
    def _new_cache(self):
        cfg = getattr(self.model, "config", None)
        try:
            return DynamicCache(config=cfg)
        except TypeError:
            return DynamicCache()

    def _resolve(self, obj, found: list):
        if is_cache_ref(obj):
            sid, n = obj["__cache__"], int(obj["len"])
            entry = self._caches.get(sid)
            if n == 0:
                entry = [self._new_cache(), 0]
                self._caches[sid] = entry
            elif entry is None or entry[1] != n:
                have = None if entry is None else entry[1]
                raise RuntimeError(
                    f"KV cache out of sync for session {sid[:8]}: host expects {n} "
                    f"cached tokens, this worker has {have} (evicted or restarted?)"
                )
            self._caches.move_to_end(sid)
            while len(self._caches) > MAX_CACHE_SESSIONS:
                self._caches.popitem(last=False)
            found.append((entry, dict(obj)))
            return entry[0]
        if isinstance(
            obj, ModelOutput
        ):  # ModelOutput is a dict subclass: keep its type
            return obj.__class__(
                **{k: self._resolve(v, found) for k, v in obj.items() if v is not None}
            )
        if isinstance(obj, dict):
            return {k: self._resolve(v, found) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._resolve(v, found) for v in obj)
        return obj

    def _scrub(self, obj, ref: Optional[dict], cache):
        if isinstance(obj, RemoteKVCache):  # host-side stub -> wire reference
            return {"__cache__": obj.session_id, "len": obj._seen}
        if isinstance(obj, Cache):
            if ref is not None and obj is cache:
                return ref
            return None  # a cache created locally (no session): never serialised
        if isinstance(obj, ModelOutput):
            return obj.__class__(
                **{
                    k: self._scrub(v, ref, cache)
                    for k, v in obj.items()
                    if v is not None
                }
            )
        if isinstance(obj, dict):
            return {k: self._scrub(v, ref, cache) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._scrub(v, ref, cache) for v in obj)
        return obj

    # ---- run ---------------------------------------------------------------
    @torch.no_grad()
    def run(self, variables: Dict[str, Any], new_tokens: int = 0) -> Dict[str, Any]:
        found: list = []
        vin = self._resolve(self._cast(attach(variables, self.device)), found)
        try:
            out = self._fn(self.model, vin)
        except Exception as e:
            import traceback

            tb = traceback.extract_tb(e.__traceback__)[-3:]
            where = " <- ".join(f"{f.name}:{f.lineno}" for f in reversed(tb))
            raise RuntimeError(
                f"{type(e).__name__}: {e} [at {where}] [weights {self._dtype_counts}] "
                f"[inputs {self._describe(vin)[:12]}]"
            ) from e

        entry, ref = found[0] if found else (None, None)
        cache = entry[0] if entry else None
        out = self._scrub(out, ref, cache)
        if entry is not None and self.has_layers:
            entry[1] += int(new_tokens)
        return out
