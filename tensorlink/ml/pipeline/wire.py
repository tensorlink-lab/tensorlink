"""
tensorlink/ml/pipeline/wire.py

Envelope carried by a pipeline FORWARD message. It rides on the existing FORWARD
transport (payload = envelope bytes, trailer = {"module_id", "key"}), so
Torchnode._handle_forward / check_forward are reused unchanged.

    envelope = meta_len(8, big endian) | meta_json | vars_blob

meta = {
  "v": 1,
  "step": int, "micro": int,          # key = (step, micro, module_id)
  "hops": [{"worker": id, "module_id": id}, ...],   # full route, in order
  "at": int,                          # index in hops of the receiving segment
  "new_tokens": int,                  # query tokens this step (KV cache advance)
}
When the last hop finishes it replies to the host with key
(step, micro, hops[-1].module_id), which is exactly what the host polls for.

vars_blob uses the tensor_to_bytes layout, but is decoded by a RESTRICTED reader:
only tensors, tuples, lists, dicts, primitives and transformers ModelOutput classes
are reconstructed. Worker-to-worker links are untrusted input, so arbitrary
dataclass / custom-object imports (allowed by utils.bytes_to_tensor) are refused.
"""

from __future__ import annotations

import importlib
import json
from typing import Any, Dict, Tuple

import torch
from safetensors.torch import load as st_load_bytes
from transformers.utils import ModelOutput

from tensorlink.ml.utils import tensor_to_bytes

try:  # exported name differs between layouts of the utils package
    from tensorlink.ml.utils import DTYPE_OBJ_MAP
except ImportError:  # pragma: no cover
    from tensorlink.ml.utils.utils import DTYPE_OBJ_MAP

WIRE_VERSION = 1
_ALLOWED_MODEL_OUTPUT_PREFIX = "transformers."


class WireError(Exception):
    pass


def dumps_vars(obj: Any) -> bytes:
    return tensor_to_bytes(obj)


def loads_vars(data: bytes) -> Any:
    slen = int.from_bytes(data[:4], "big")
    structure = json.loads(data[4 : 4 + slen].decode("utf-8"))
    blob = data[4 + slen :]
    tensors = st_load_bytes(bytes(blob)) if blob else {}

    def restore(o):
        if isinstance(o, list):
            return [restore(v) for v in o]
        if not isinstance(o, dict):
            return o
        if "__tensor_ref__" in o:
            dt = o["dtype"]
            if dt not in DTYPE_OBJ_MAP:
                raise WireError(f"unsupported dtype {dt!r}")
            return tensors[o["__tensor_ref__"]].to(dtype=DTYPE_OBJ_MAP[dt])
        if o.get("__tuple__"):
            return tuple(restore(v) for v in o["data"])
        if o.get("__hf_model_output__"):
            mod = str(o.get("module", ""))
            if not mod.startswith(_ALLOWED_MODEL_OUTPUT_PREFIX):
                raise WireError(f"refusing to import {mod!r}")
            cls = getattr(importlib.import_module(mod), str(o["class"]), None)
            if not (isinstance(cls, type) and issubclass(cls, ModelOutput)):
                raise WireError(f"{mod}.{o.get('class')} is not a ModelOutput")
            return cls(**restore(o["data"]))
        if o.get("__dynamic_cache__") or o.get("__dataclass__") or o.get("__custom_object__"):
            raise WireError("object type not permitted on the pipeline wire")
        return {k: restore(v) for k, v in o.items()}

    return restore(structure)


def pack(meta: Dict[str, Any], variables: Any) -> bytes:
    meta = {"v": WIRE_VERSION, **meta}
    mb = json.dumps(meta).encode()
    return len(mb).to_bytes(8, "big") + mb + dumps_vars(variables)


def unpack(data) -> Tuple[Dict[str, Any], Any]:
    data = bytes(data)
    n = int.from_bytes(data[:8], "big")
    if n <= 0 or n > len(data) - 8:
        raise WireError("bad envelope length")
    meta = json.loads(data[8 : 8 + n].decode())
    if meta.get("v") != WIRE_VERSION:
        raise WireError(f"wire version {meta.get('v')} != {WIRE_VERSION}")
    hops = meta.get("hops")
    at = meta.get("at")
    if not isinstance(hops, list) or not hops or not isinstance(at, int) or not 0 <= at < len(hops):
        raise WireError("bad route")
    return meta, loads_vars(data[8 + n :])


def next_hop(meta: Dict[str, Any]):
    """The hop after this one, or None when this segment is the last (reply to host)."""
    i = meta["at"] + 1
    return meta["hops"][i] if i < len(meta["hops"]) else None
