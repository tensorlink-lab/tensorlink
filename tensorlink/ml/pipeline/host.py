"""
tensorlink/ml/pipeline/host.py

Host (user) side of the pipeline strategy.

    forward(...) -> PipelineRunner.run(params)
        [host pre segment, optional]  ->  ONE send to worker1
        worker1 -> worker2 -> ... -> workerK        (direct worker-to-worker)
        workerK -> ONE reply to the host  ->  [host post segment, optional]

Per step the host talks to the network twice (send + reply) instead of 2 * K times.
The legacy OffloadedModule / LayerGroupModule path stays in module.py as the fallback.
"""

from __future__ import annotations

import logging
import threading
import time
import types
from typing import Any, Dict, List, Optional

import torch

from tensorlink.ml.pipeline import wire
from tensorlink.ml.pipeline.loader import (
    PipelineMismatch,
    build_skeleton,
    check_segment,
    derive_program,
    materialize,
)
from tensorlink.ml.pipeline.segment import SegmentRunner, attach, is_cache_ref
from tensorlink.ml.utils import detach_tensor
from tensorlink.ml.utils.injector import RemoteKVCache, swap_remote_cache
from tensorlink.utils.shared_memory import (
    get_from_shared_memory,
    store_in_shared_memory,
)

log = logging.getLogger(__name__)


class PipelineFallback(Exception):
    """The pipeline strategy cannot be used for this job; re-request with the legacy one."""


def is_pipeline_config(config: dict) -> bool:
    vals = [v for v in config.values() if isinstance(v, dict)]
    return bool(vals) and all(v.get("strategy") == "pipeline" for v in vals)


def _new_tokens(params: Dict[str, Any]) -> int:
    ids = params.get("input_ids")
    if hasattr(ids, "shape") and len(ids.shape) >= 1:
        return int(ids.shape[-1])
    emb = params.get("inputs_embeds")
    if hasattr(emb, "shape") and len(emb.shape) >= 2:
        return int(emb.shape[-2])
    cp = params.get("cache_position")
    if hasattr(cp, "shape"):
        return int(cp.shape[0])
    return 0


def _restore_refs(obj, stub):
    if is_cache_ref(obj):
        return stub
    try:
        from transformers.utils import ModelOutput

        if isinstance(obj, ModelOutput):  # before dict: ModelOutput subclasses dict
            return obj.__class__(
                **{k: _restore_refs(v, stub) for k, v in obj.items() if v is not None}
            )
    except Exception:  # pragma: no cover
        pass
    if isinstance(obj, dict):
        return {k: _restore_refs(v, stub) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and not hasattr(obj, "_fields"):
        return type(obj)(_restore_refs(v, stub) for v in obj)
    return obj


class PipelineRunner:
    def __init__(
        self,
        parent,
        hops: List[dict],
        pre: Optional[SegmentRunner],
        post: Optional[SegmentRunner],
        wait_timeout: float,
    ):
        self.parent = parent  # DistributedModel
        self.hops = hops  # [{"worker": id, "module_id": id}, ...]
        self.pre, self.post = pre, post
        self.timeout = wait_timeout
        self._step = 0
        self._lock = threading.Lock()

    @property
    def has_layers(self) -> bool:
        return True

    # ------------------------------------------------------------------ run
    def run(self, params: Dict[str, Any]):
        parent = self.parent
        micro = getattr(parent._thread_local, "micro", None) or 0
        with self._lock:
            step = self._step
            self._step += 1

        params = attach(params, parent.device)  # keeps the RemoteKVCache stub intact
        new_tokens = _new_tokens(params)
        wire_params, stub = swap_remote_cache(params)

        if self.pre is not None:
            # host segments see the real RemoteKVCache stub (correct get_seq_length /
            # mask sizes); its outputs are scrubbed back to wire references
            out = self.pre.run(params, new_tokens)
            variables: Any = out["__vars__"]
        else:
            variables = detach_tensor(wire_params)

        meta = {
            "step": step,
            "micro": micro,
            "hops": self.hops,
            "at": 0,
            "new_tokens": new_tokens,
        }
        blob = wire.pack(meta, variables)
        size, shm = store_in_shared_memory(blob, encoding=None)
        first = self.hops[0]
        last_id = self.hops[-1]["module_id"]
        parent.send_request(
            "send_forward",
            (
                first["worker"],
                first["module_id"],
                size,
                shm,
                [step, micro, first["module_id"]],
            ),
        )

        key = (step, micro, last_id)
        t0 = time.time()
        reply = None
        while reply is None:
            time.sleep(0.005)
            reply = parent.send_request("check_forward", key)
            if reply is None and time.time() - t0 > self.timeout:
                raise TimeoutError(
                    f"pipeline step {step} timed out after {self.timeout:.0f}s "
                    f"(route: {[h['worker'][:8] for h in self.hops]})"
                )
        rsize, rname = reply
        data = get_from_shared_memory(rsize, rname, encoding=None)
        parent.send_request("release_memory", ("forward_queue", last_id, key))

        rmeta, rvars = wire.unpack(data)
        if rmeta.get("error"):
            raise RuntimeError(f"pipeline worker error: {rmeta['error']}")

        if self.post is not None:
            out = self.post.run(_restore_refs(rvars["__vars__"], stub), new_tokens)
            result = out["__return__"]
        else:
            result = rvars["__return__"]

        if stub is not None:
            stub._seen += new_tokens
        result = _restore_refs(result, stub)
        return attach(result, parent.device)


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #


def prepare_pipeline(parent, config: dict, model_name: str, model_type: str):
    """
    Build the host-side skeleton + program, verify every plan entry against it and
    load the host (privacy) segments. Returns (model, program, entries, pre, post).
    Raises PipelineMismatch / RuntimeError when the plan cannot be honoured; the caller
    then falls back to the legacy strategy.
    """
    entries = sorted(
        ((mid, info) for mid, info in config.items()), key=lambda kv: kv[1]["index"]
    )
    first = entries[0][1]
    dtype = first["dtype"]
    model = build_skeleton(model_name, model_type, dtype)
    program = derive_program(model, first.get("program_fp"))
    for _, info in entries:
        check_segment(program, info["segment"])

    model_path = parent.cache_manager.load_model_path(model_name)
    tied = first.get("tied_groups") or ()
    pre = post = None
    for _, info in entries:
        if info["kind"] not in ("pre", "post"):
            continue
        code = check_segment(program, info["segment"])
        materialize(model, code.modules, model_path, parent.device, dtype, tied)
        runner = SegmentRunner(model, program, code, parent.device)
        if info["kind"] == "pre":
            pre = runner
        else:
            post = runner
    return model, program, entries, pre, post


def install_forward(model, program, runner: PipelineRunner):
    """Replace model.forward by a stub with the ORIGINAL signature that calls the runner."""
    ns: Dict[str, Any] = {}
    exec(
        compile(
            program.host_forward_source("_tl_pipeline"), "<tensorlink-forward>", "exec"
        ),
        ns,
    )
    model._tl_pipeline = runner
    model.forward = types.MethodType(ns["forward"], model)
