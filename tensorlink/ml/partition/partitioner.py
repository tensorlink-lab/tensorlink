"""
tensorlink/ml/partition/partitioner.py

``ModelPartitioner`` is the pipeline-strategy counterpart of the legacy
``ModelParser`` (plan.py, which stays untouched as the fallback).

It glues together:
    forward_program.build_forward_program   (static forward() analysis)
    pipeline_plan.plan_pipeline             (linear next-fit packing)

and returns the same result shape as ``ModelParser.create_distributed_config``:
    {"success", "config", "model_memory", "host_memory_used", "error"}
so the validator can try the pipeline first and fall back to ModelParser when
``success`` is False (including any ProgramError from unsupported forward()s).

Sizes come from the (meta / skeleton) model's parameters and buffers; nothing is
loaded and no forward pass is run.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .forward_program import ForwardProgram, ProgramError, build_forward_program
from .pipeline_plan import PipelinePlan, PlanOptions, WorkerSpec, plan_pipeline

_DTYPE_BYTES = {
    "float32": 4, "fp32": 4, "float": 4,
    "float16": 2, "fp16": 2, "half": 2, "bfloat16": 2, "bf16": 2,
    "int8": 1, "fp8": 1,
}


def _walk(model, path: str):
    obj = model
    for p in path.split("."):
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    return obj


def _param_bytes(module, dtype_bytes: Optional[int]) -> int:
    total = 0
    for p in module.parameters():
        total += p.numel() * (dtype_bytes or p.element_size())
    for b in module.buffers():
        total += b.numel() * b.element_size()
    return total


class ModelPartitioner:
    def __init__(self, max_inline_depth: int = 3):
        self.max_inline_depth = max_inline_depth
        self.program: Optional[ForwardProgram] = None
        self.plan: Optional[PipelinePlan] = None

    # ------------------------------------------------------------------ sizing
    def module_bytes(
        self, model, program: ForwardProgram, dtype: str = "float16",
        kv_bytes_per_layer: int = 0,
    ) -> Dict[str, int]:
        dtype_bytes = _DTYPE_BYTES.get(str(dtype).replace("torch.", ""), None)
        out: Dict[str, int] = {}
        for u in program.execution_units():
            for path in u.paths:
                b = _param_bytes(_walk(model, path), dtype_bytes)
                if u.layer is not None:
                    b += kv_bytes_per_layer
                out[path] = b
        for st in program.stmts:  # modules shared by a whole layer loop
            if st.loop:
                for path in st.loop.shared:
                    out.setdefault(path, _param_bytes(_walk(model, path), dtype_bytes))
        return out

    @staticmethod
    def kv_bytes_per_layer(model, batch_size: int, max_seq_len: int, dtype: str) -> int:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return 0
        heads = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads", 0)
        hidden = getattr(cfg, "hidden_size", 0)
        n_heads = getattr(cfg, "num_attention_heads", 0) or 1
        head_dim = getattr(cfg, "head_dim", None) or (hidden // n_heads if hidden else 0)
        per = _DTYPE_BYTES.get(str(dtype).replace("torch.", ""), 2)
        return 2 * heads * head_dim * max_seq_len * batch_size * per

    @staticmethod
    def tied_groups(model, program: ForwardProgram) -> List[List[str]]:
        """Unit keys whose parameters are the same object (e.g. lm_head <-> embed_tokens)."""
        owner: Dict[int, List[str]] = {}
        for u in program.execution_units():
            for path in u.paths:
                for p in _walk(model, path).parameters():
                    owner.setdefault(id(p), [])
                    if path not in owner[id(p)]:
                        owner[id(p)].append(path)
        groups = [g for g in owner.values() if len(g) > 1]
        # Config-declared tying (tie_word_embeddings) even when the skeleton did not
        # share the Parameter object (depends on transformers version / meta init).
        try:
            cfg = getattr(model, "config", None)
            if cfg is not None and getattr(cfg, "tie_word_embeddings", False):
                ie, oe = model.get_input_embeddings(), model.get_output_embeddings()
                names = {id(m): n for n, m in model.named_modules()}
                a, b = names.get(id(ie)), names.get(id(oe))
                unit_paths = {p for u in program.execution_units() for p in u.paths}
                if a and b and a != b and a in unit_paths and b in unit_paths:
                    if not any(a in g and b in g for g in groups):
                        groups.append([a, b])
        except Exception:
            pass
        return groups

    # -------------------------------------------------------------------- main
    def create_pipeline_config(
        self,
        model,
        workers: Sequence[dict],
        *,
        dtype: str = "float16",
        batch_size: int = 1,
        max_seq_len: int = 2048,
        host_pre_units: int = 0,
        host_post_units: int = 0,
        host_max_memory_bytes: int = 0,
        reserve_frac: float = 0.10,
        max_workers: int = 6,
        model_name: str = "",
        model_type: str = "chat",
        is_module=None,
    ) -> dict:
        """``workers``: [{"id": str, "memory": int_bytes}, ...]"""
        try:
            program = build_forward_program(
                model, is_module=is_module, max_inline_depth=self.max_inline_depth
            ).bind(model, is_module)
            kv = self.kv_bytes_per_layer(model, batch_size, max_seq_len, dtype)
            mb = self.module_bytes(model, program, dtype, kv)
            opts = PlanOptions(
                host_pre_units=host_pre_units,
                host_post_units=host_post_units,
                host_budget_bytes=host_max_memory_bytes,
                reserve_frac=reserve_frac,
                max_workers=max_workers,
                tied_groups=self.tied_groups(model, program),
            )
            plan = plan_pipeline(
                program, mb, [WorkerSpec(w["id"], int(w["memory"])) for w in workers], opts
            )
        except ProgramError as e:
            return self._result(False, error=f"pipeline unsupported: {e}")
        except Exception as e:  # never let analysis crash the validator
            return self._result(False, error=f"pipeline analysis failed: {e!r}")

        self.program, self.plan = program, plan
        if not plan.success:
            return self._result(False, error=plan.error)
        extra = {
            "name": model_name,
            "model_type": model_type,
            "dtype": str(dtype).replace("torch.", ""),
            "training": False,
            "program_fp": plan.program_fp,
            "tied_groups": opts.tied_groups,
        }
        return self._result(
            True, config=plan.to_config(extra), model_memory=plan.total_bytes,
            host_memory_used=plan.host_bytes,
        )

    def create_pipeline_config_for_name(
        self, model_name: str, workers: dict, *, model_type: str = "chat",
        dtype: str = "float16", **kw
    ) -> dict:
        """Validator entry point. ``workers`` is the validator's {id: {"gpu_memory": n}}.
        Never raises: any failure returns success=False so the caller can fall back
        to ModelParser."""
        try:
            from tensorlink.ml.pipeline.loader import build_skeleton

            model = build_skeleton(model_name, model_type, dtype)
            ws = [
                {"id": wid, "memory": int((w or {}).get("gpu_memory", 0))}
                for wid, w in (workers or {}).items()
            ]
            return self.create_pipeline_config(
                model, ws, dtype=dtype, model_name=model_name, model_type=model_type, **kw
            )
        except Exception as e:
            return self._result(False, error=f"pipeline skeleton failed: {e!r}")

    @staticmethod
    def _result(success, config=None, model_memory=0, host_memory_used=0, error=""):
        return {
            "success": success,
            "config": config or {},
            "model_memory": model_memory,
            "host_memory_used": host_memory_used,
            "error": error,
        }
