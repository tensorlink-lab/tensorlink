"""
tensorlink/ml/partition/pipeline_plan.py

Pure planning: given the ordered execution units of a ``ForwardProgram`` and the
available workers, decide who runs which contiguous slice of the forward pass.

    [host pre]  ->  worker1 -> worker2 -> ... -> workerK  ->  [host post]

* Host pre / post are an optional privacy feature (off by default). When enabled
  the host must hold at least ``host_pre_units`` / ``host_post_units`` weight-bearing
  units at the matching end of the model, otherwise planning FAILS (the privacy
  guarantee would not hold).
* ``host_fill``: the host may ALSO spend its memory budget on extra units at either
  end (a chunk of the decoder loop included) so fewer worker hops are needed. Privacy
  counts are then a floor, not the total. Never fails: it takes what fits.
* Tied weights (e.g. embed_tokens / lm_head): when privacy is on (or fill is on and the
  budget allows) and one half lands on the host, both halves stay on the host.
* Workers are filled one at a time (next-fit) in linear forward order, largest
  free memory first. A worker is never revisited.

No torch import here; sizes are supplied by the caller.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Position = Tuple[int, int]


@dataclass
class WorkerSpec:
    id: str
    free_bytes: int


@dataclass
class PlanOptions:
    host_pre_units: int = 0  # units of layers / module components
    host_post_units: int = 0
    host_budget_bytes: int = 0  # memory the host may spend on its own segments
    host_fill: bool = (
        False  # use the budget for extra host units beyond the privacy floor
    )
    reserve_frac: float = 0.10  # worker memory kept free for activations / runtime
    max_workers: int = 6
    tied_groups: Sequence[Sequence[str]] = ()  # groups of module paths sharing weights

    @property
    def privacy(self) -> bool:
        return self.host_pre_units > 0 or self.host_post_units > 0


@dataclass
class PlanSegment:
    index: int
    owner: str  # "host" or a worker id
    kind: str  # "pre" | "worker" | "post"
    start: Position
    end: Position
    modules: List[str]
    bytes: int
    live_in: List[str]
    live_out: List[str]
    final: bool
    has_layers: bool
    sha: str
    units: Tuple[int, int] = (0, 0)  # [first, last) unit indices


@dataclass
class PipelinePlan:
    success: bool
    error: str = ""
    segments: List[PlanSegment] = field(default_factory=list)
    fingerprint: str = ""
    privacy: bool = False
    program_fp: str = ""
    total_bytes: int = 0
    host_bytes: int = 0

    @property
    def worker_segments(self) -> List[PlanSegment]:
        return [s for s in self.segments if s.kind == "worker"]

    def to_config(self, extra: Optional[dict] = None) -> dict:
        """Legacy-shaped config ``{entry_name: module_info}`` so the validator's job
        machinery (module_id = sha256(module_info), "offloaded" in type, memory,
        assigned_workers) works unchanged. ``extra`` is merged into every entry
        (model name, dtype, ...)."""
        out = {}
        n = len(self.segments)
        for s in self.segments:
            host = s.owner == "host"
            out[f"pipeline_segment_{s.index}"] = {
                **(extra or {}),
                "type": "loaded" if host else "offloaded_segment",
                "strategy": "pipeline",
                "module": "PipelineSegment",
                "module_path": "",
                "kind": s.kind,
                "index": s.index,
                "n_segments": n,
                "memory": s.bytes,
                "assigned_workers": [] if host else [s.owner],
                "plan_fp": self.fingerprint,
                "segment": {
                    "start": list(s.start),
                    "end": list(s.end),
                    "sha": s.sha,
                    "modules": s.modules,
                    "live_in": s.live_in,
                    "live_out": s.live_out,
                    "final": s.final,
                    "has_layers": s.has_layers,
                },
            }
        return out


def _fail(msg: str) -> PipelinePlan:
    return PipelinePlan(success=False, error=msg)


def _expand_for_ties(
    n_units: int,
    unit_of: Dict[str, int],
    opts: PlanOptions,
    pre: int,
    post: int,
    force: bool = False,
):
    """Grow the host prefix/suffix until every tied group is entirely on the host or
    entirely off it. Applies when privacy is on, or when ``force`` (host fill)."""
    if not (opts.privacy or force):
        return pre, post
    mid = n_units / 2.0
    changed = True
    while changed:
        changed = False
        for grp in opts.tied_groups:
            idxs = sorted({unit_of[m] for m in grp if m in unit_of})
            if len(idxs) < 2:
                continue
            on_host = [i for i in idxs if i < pre or i >= n_units - post]
            if not on_host:
                continue
            for i in idxs:
                if i < pre or i >= n_units - post:
                    continue
                if i < mid:
                    pre = i + 1
                else:
                    post = n_units - i
                changed = True
    return pre, post


def plan_pipeline(
    program,
    module_bytes: Dict[str, int],
    workers: Sequence[WorkerSpec],
    opts: Optional[PlanOptions] = None,
) -> PipelinePlan:
    opts = opts or PlanOptions()
    units = program.execution_units()
    n = len(units)
    if n == 0:
        return _fail("model has no weight-bearing units")

    def ubytes(u) -> int:
        return sum(module_bytes.get(p, 0) for p in u.paths)

    sizes = [ubytes(u) for u in units]
    total = sum(sizes)
    unit_of: Dict[str, int] = {}
    for i, u in enumerate(units):
        for p in u.paths:
            unit_of[p] = i

    pre, post = opts.host_pre_units, opts.host_post_units
    if pre < 0 or post < 0:
        return _fail("host_pre_units / host_post_units must be >= 0")

    pre, post = _expand_for_ties(n, unit_of, opts, pre, post)
    if pre + post >= n:
        return _fail(
            f"privacy needs {pre} pre + {post} post host units but the model only has {n}; "
            "nothing would be left to distribute"
        )

    host_bytes = sum(sizes[:pre]) + sum(sizes[n - post :])
    if opts.privacy and opts.host_budget_bytes and host_bytes > opts.host_budget_bytes:
        return _fail(
            f"host segments need {host_bytes} bytes but the host budget is "
            f"{opts.host_budget_bytes}; cannot provide the requested privacy"
        )

    # optional: spend the rest of the host budget on extra units at both ends
    if opts.host_fill and opts.host_budget_bytes > 0:
        budget = int(opts.host_budget_bytes * (1.0 - opts.reserve_frac))

        def span_bytes(a: int, b: int) -> int:
            return sum(sizes[:a]) + sum(sizes[n - b :]) if (a or b) else 0

        blocked = {"front": False, "back": False}
        side = "front"
        while not (blocked["front"] and blocked["back"]):
            if not blocked[side]:
                np_, nq = (pre + 1, post) if side == "front" else (pre, post + 1)
                np_, nq = _expand_for_ties(n, unit_of, opts, np_, nq, force=True)
                # always leave at least one unit for the workers
                if np_ + nq >= n or span_bytes(np_, nq) > budget:
                    blocked[side] = True
                else:
                    pre, post = np_, nq
            side = "back" if side == "front" else "front"
        host_bytes = span_bytes(pre, post)

    # pack the middle across workers (next-fit, largest first)
    ws = sorted(workers, key=lambda w: w.free_bytes, reverse=True)
    mid_lo, mid_hi = pre, n - post
    groups: List[Tuple[WorkerSpec, int, int]] = (
        []
    )  # (worker, first_unit, last_unit_excl)
    wi, i = 0, mid_lo
    while i < mid_hi:
        if wi >= len(ws):
            return _fail(
                f"workers exhausted: {mid_hi - i} unit(s) / "
                f"{sum(sizes[i:mid_hi])} bytes could not be placed"
            )
        w = ws[wi]
        cap = int(w.free_bytes * (1.0 - opts.reserve_frac))
        used, j = 0, i
        while j < mid_hi and used + sizes[j] <= cap:
            used += sizes[j]
            j += 1
        if j == i:  # nothing fits on this worker; skip it (never revisited)
            if sizes[i] > int(ws[0].free_bytes * (1.0 - opts.reserve_frac)):
                return _fail(
                    f"unit {units[i].key!r} ({sizes[i]} bytes) does not fit on the "
                    "largest worker"
                )
            wi += 1
            continue
        groups.append((w, i, j))
        i, wi = j, wi + 1

    if len(groups) > opts.max_workers:
        return _fail(f"plan needs {len(groups)} workers, limit is {opts.max_workers}")

    # cut positions
    def cut_after(unit_idx: int) -> Position:
        return units[unit_idx].end

    bounds: List[Tuple[str, str, int, int]] = []  # (owner, kind, u0, u1)
    if pre:
        bounds.append(("host", "pre", 0, pre))
    for w, a, b in groups:
        bounds.append((w.id, "worker", a, b))
    if post:
        bounds.append(("host", "post", n - post, n))

    segs: List[PlanSegment] = []
    for idx, (owner, kind, u0, u1) in enumerate(bounds):
        start = (0, 0) if idx == 0 else cut_after(u0 - 1)
        end = program.end_pos if idx == len(bounds) - 1 else cut_after(u1 - 1)
        code = program.segment(start, end)
        segs.append(
            PlanSegment(
                index=idx,
                owner=owner,
                kind=kind,
                start=start,
                end=end,
                modules=code.modules,
                bytes=sum(module_bytes.get(m, 0) for m in code.modules),
                live_in=code.live_in,
                live_out=code.live_out,
                final=code.final,
                has_layers=code.has_layers,
                sha=code.sha,
                units=(u0, u1),
            )
        )

    # modules re-used by statements in another segment are loaded there too; make sure
    # that duplicated weight still fits the worker it lands on
    cap_of = {w.id: int(w.free_bytes * (1.0 - opts.reserve_frac)) for w in ws}
    for sg in segs:
        if sg.kind == "worker" and sg.bytes > cap_of.get(sg.owner, sg.bytes):
            return _fail(
                f"segment {sg.index} needs {sg.bytes} bytes (shared modules are loaded on "
                f"every segment that calls them) but worker capacity is {cap_of[sg.owner]}"
            )

    fp = hashlib.sha256(
        json.dumps(
            [program.fingerprint] + [(s.owner, s.start, s.end, s.sha) for s in segs]
        ).encode()
    ).hexdigest()
    return PipelinePlan(
        success=True,
        segments=segs,
        fingerprint=fp,
        program_fp=program.fingerprint,
        privacy=opts.privacy,
        total_bytes=total,
        host_bytes=sum(s.bytes for s in segs if s.owner == "host"),
    )
