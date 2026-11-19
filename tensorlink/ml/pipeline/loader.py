"""
tensorlink/ml/pipeline/loader.py

Static (no forward pass) loading of the modules a pipeline segment owns.

Hidden buffers
--------------
The legacy path ran a dummy forward over the whole skeleton to discover non-persistent
buffers (e.g. rotary ``inv_freq``) and shipped them to workers, because
``load_module_weights`` calls ``to_empty()`` which wipes every buffer.

Here the skeleton is built with ``init_empty_weights(include_buffers=False)``, so its
buffers are REAL tensors computed deterministically by the module constructors from
the model config. Only *parameters* live on the meta device. We therefore never call
``to_empty()``; we replace meta parameters in place (direct assignment, tied weights share one tensor)
and leave buffers alone. Nothing needs to be discovered or transferred, and any
buffer that is unexpectedly still on meta is reported as an error before serving.

Both sides also re-derive the segment's source from their *own* installed
``transformers`` and compare hashes with the plan, so no source code is ever shipped
or exec()'d from the network.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
)

from tensorlink.ml.partition.forward_program import build_forward_program

log = logging.getLogger(__name__)

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


class PipelineMismatch(Exception):
    """Plan was made against a different forward() than this node derives."""


def to_torch_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    return _DTYPES.get(str(dtype).replace("torch.", ""), torch.float16)


def build_skeleton(model_name: str, model_type: str = "chat", dtype="float16"):
    """Model whose parameters are on 'meta' and whose buffers are real (CPU)."""
    cfg = AutoConfig.from_pretrained(model_name)
    td = to_torch_dtype(dtype)
    cls = {
        "chat": AutoModelForCausalLM,
        "causal": AutoModelForCausalLM,
        "seq2seq": AutoModelForSeq2SeqLM,
        "audio2text": AutoModelForSpeechSeq2Seq,
    }.get(model_type, AutoModel)
    with init_empty_weights(include_buffers=False):
        model = cls.from_config(cfg, torch_dtype=td)
    # NOTE: no model.to(dtype) here: that would also round float buffers (inv_freq)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def derive_program(model, plan_fp: Optional[str] = None):
    program = build_forward_program(model).bind(model)
    if plan_fp and program.fingerprint != plan_fp:
        raise PipelineMismatch(
            "forward() differs from the one the plan was made for "
            "(different transformers version?)"
        )
    return program


def check_segment(program, seg: Dict[str, Any]):
    """Re-derive a segment from this node's install and verify it matches the plan."""
    code = program.segment(tuple(seg["start"]), tuple(seg["end"]))
    if code.sha != seg["sha"]:
        raise PipelineMismatch("segment source hash differs from the plan")
    return code


# --------------------------------------------------------------------------- #
# checkpoint access
# --------------------------------------------------------------------------- #


_BIN_CACHE: Dict[str, Dict[str, torch.Tensor]] = {}
_KEEP: List[Any] = []


def _load_bin(path: str) -> Dict[str, torch.Tensor]:
    """Legacy pytorch_model*.bin shard (weights_only: no arbitrary unpickling)."""
    if path not in _BIN_CACHE:
        try:
            sd = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except Exception:
            sd = torch.load(path, map_location="cpu", weights_only=True)
        if (
            isinstance(sd, dict)
            and "state_dict" in sd
            and isinstance(sd["state_dict"], dict)
        ):
            sd = sd["state_dict"]
        _BIN_CACHE[path] = sd
    return _BIN_CACHE[path]


def _shard_map(model_path: str) -> Dict[str, str]:
    """weight key -> shard file. Prefers safetensors, falls back to pytorch_model*.bin."""
    idx = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            wm = json.load(f)["weight_map"]
        return {k: os.path.join(model_path, v) for k, v in wm.items()}
    out: Dict[str, str] = {}
    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = shard
    if out:
        return out
    bidx = os.path.join(model_path, "pytorch_model.bin.index.json")
    if os.path.exists(bidx):
        with open(bidx) as f:
            wm = json.load(f)["weight_map"]
        return {k: os.path.join(model_path, v) for k, v in wm.items()}
    for shard in sorted(glob.glob(os.path.join(model_path, "pytorch_model*.bin"))):
        for k in _load_bin(shard):
            out[k] = shard
    return out


def _resolve_prefix(keys: Iterable[str], module_path: str) -> Optional[str]:
    parts = module_path.split(".")
    keys = list(keys)
    for i in range(len(parts)):
        cand = ".".join(parts[i:])
        if any(k.startswith(cand + ".") for k in keys):
            return cand
    return None


def _read(shard_map: Dict[str, str], wanted: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """wanted: checkpoint key -> local key. One open per shard."""
    by_shard: Dict[str, List[str]] = {}
    for k in wanted:
        by_shard.setdefault(shard_map[k], []).append(k)
    out: Dict[str, torch.Tensor] = {}
    for shard, ks in by_shard.items():
        if shard.endswith(".safetensors"):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for k in ks:
                    out[wanted[k]] = f.get_tensor(k)
        else:
            sd = _load_bin(shard)
            for k in ks:
                out[wanted[k]] = sd[k]
    return out


def _get(model, path: str):
    obj = model
    for p in path.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def _tie_siblings(model, path: str) -> List[str]:
    """Other module paths whose weights are tied to ``path`` according to the skeleton
    (shared Parameter objects) or the config (tie_word_embeddings)."""
    out: List[str] = []
    try:
        mod = _get(model, path)
        mine = {id(p) for p in mod.parameters()}
        for n, m in model.named_modules():
            if n and n != path and mine & {id(p) for p in m.parameters(recurse=False)}:
                out.append(n)
        cfg = getattr(model, "config", None)
        if cfg is not None and getattr(cfg, "tie_word_embeddings", False):
            names = {id(m): n for n, m in model.named_modules()}
            ie, oe = model.get_input_embeddings(), model.get_output_embeddings()
            a, b = names.get(id(ie)), names.get(id(oe))
            if a == path and b and b not in out:
                out.append(b)
            if b == path and a and a not in out:
                out.append(a)
    except Exception:
        pass
    return out


def _assign(
    module, sd: Dict[str, torch.Tensor], shared: Dict[int, torch.nn.Parameter]
) -> int:
    """Replace meta parameters with real ones. ``shared`` maps the id of the original
    (meta) Parameter to the real one so tied weights stay a single tensor."""
    n = 0
    for name, old in list(module.named_parameters(remove_duplicate=False)):
        new = shared.get(id(old))
        if new is None:
            t = sd.get(name)
            if t is None:
                continue
            if tuple(t.shape) != tuple(old.shape):
                raise RuntimeError(
                    f"{name}: checkpoint shape {tuple(t.shape)} != model {tuple(old.shape)}"
                )
            new = torch.nn.Parameter(t, requires_grad=False)
            shared[id(old)] = new
            _KEEP.append(old)  # keep alive: ids in `shared` must not be recycled
        owner, parts = module, name.split(".")
        for q in parts[:-1]:
            owner = getattr(owner, q)
        owner._parameters[parts[-1]] = new
        n += 1
    return n


def materialize(
    model,
    module_paths: Sequence[str],
    model_path: str,
    device,
    dtype,
    tied_groups: Sequence[Sequence[str]] = (),
) -> None:
    """Load weights for ``module_paths`` into ``model`` in place (params only)."""
    td = to_torch_dtype(dtype)
    shard_map = _shard_map(model_path)
    if not shard_map:
        raise RuntimeError(
            f"no .safetensors / pytorch_model*.bin weights in {model_path}"
        )
    keys = list(shard_map)
    shared: Dict[int, torch.nn.Parameter] = {}  # id(meta param) -> loaded param

    for path in module_paths:
        module = _get(model, path)
        prefix = _resolve_prefix(keys, path)
        sd: Dict[str, torch.Tensor] = {}
        if prefix is not None:
            wanted = {
                k: k[len(prefix) + 1 :] for k in keys if k.startswith(prefix + ".")
            }
            sd = _read(shard_map, wanted)

        # parameters the checkpoint does not store under this module (tied weights)
        have = set(sd)
        missing = [
            n
            for n, p in module.named_parameters(remove_duplicate=False)
            if n not in have and id(p) not in shared
        ]
        if missing:
            groups = [g for g in tied_groups if path in g]
            sibs = _tie_siblings(model, path)
            if sibs:
                groups.append([path] + sibs)  # derived locally, independent of the plan
            for grp in groups:
                for sib in grp:
                    sp = _resolve_prefix(keys, sib) if sib != path else None
                    if sp is None:
                        continue
                    wanted = {
                        f"{sp}.{n}": n for n in missing if f"{sp}.{n}" in shard_map
                    }
                    sd.update(_read(shard_map, wanted))
                    missing = [n for n in missing if n not in sd]
                if not missing:
                    break
            # (a tied copy is read straight from the checkpoint: duplicated on this node
            # if its sibling is elsewhere, but identical in value, so inference is exact)

        sd = {k: (v.to(td) if v.is_floating_point() else v) for k, v in sd.items()}
        _assign(module, sd, shared)  # buffers untouched
        res = None
        loaded = {id(v) for v in shared.values()}
        left = [
            n
            for n, p in module.named_parameters(remove_duplicate=False)
            if id(p) not in loaded
        ]
        if left:
            hint = [
                k
                for k in keys
                if any(t in k for t in (path.split(".")[-1], "embed", "lm_head"))
            ]
            raise RuntimeError(
                f"{path}: parameters not found in checkpoint: {left[:6]} "
                f"(prefix={prefix!r}, read={sorted(sd)[:6]}, {len(keys)} keys in {model_path}, "
                f"related keys: {hint[:6]})"
            )
        for n, p in module.named_parameters():  # weights always at the plan's dtype
            if p.is_floating_point() and p.dtype != td:
                p.data = p.data.to(td)
        module.to(device)  # moves params+buffers, does not change dtypes
        del sd, res

    bad = [
        f"{path}.{n}"
        for path in module_paths
        for n, b in _get(model, path).named_buffers()
        if b.is_meta
    ]
    if bad:
        raise RuntimeError(
            "buffers are on meta (skeleton was not built with include_buffers=False): "
            f"{bad[:6]}"
        )
    for path in module_paths:  # pipeline serves inference only
        _get(model, path).eval()
    _BIN_CACHE.clear()
    _KEEP.clear()


def load_segment_runner(
    module_info: Dict[str, Any], model_path: str, device, tied_groups=()
):
    """Worker/host: build a ready-to-run SegmentRunner for one plan entry."""
    from tensorlink.ml.pipeline.segment import SegmentRunner

    model = build_skeleton(
        module_info["name"], module_info.get("model_type", "chat"), module_info["dtype"]
    )
    program = derive_program(model, module_info.get("program_fp"))
    seg = module_info["segment"]
    code = check_segment(program, seg)
    materialize(
        model,
        code.modules,
        model_path,
        device,
        module_info["dtype"],
        tied_groups or module_info.get("tied_groups") or (),
    )
    return SegmentRunner(model, program, code, device)
