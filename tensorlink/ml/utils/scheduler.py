"""
tensorlink/nodes/public_model_scheduler.py

Decision engine for automatic model loading and demand upkeep on public
(hosted) validator nodes.

DistributedValidator owns turning the actions this module returns into real
work (spawning DistributedModel instances, tearing them down, etc) -- this
file only decides *what* should happen, never *how*.

Policy summary
--------------
- Private validators never auto-load anything; `plan()` is a no-op and
  paid-request capacity is decided by the validator itself (no scheduler
  involvement needed, since private validators only ever serve direct jobs).
- Public validators keep DEFAULT_MODELS warm and auto-load any model whose
  recent demand crosses a popularity threshold.
- Paid jobs always get a home. If there's free room, a new instance is
  simply spun up. If the validator is already at capacity, the
  lowest-demand free-tier instance is evicted to make room -- unless every
  free instance belongs to a model with "considerable" demand of its own,
  in which case nothing is evicted and the caller should queue/decline the
  paid request instead.
- Models under sustained demand pressure (more requests than a single
  instance can reasonably serve within the tracking window) are scaled out
  to additional instances, up to a per-model and a global cap. Scale-ups
  are rate-limited with a cooldown to avoid thrashing.
- Duplicate instances that are no longer under pressure are scaled back in,
  always leaving at least one instance for anything with active demand or
  that is a DEFAULT_MODEL.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import math
import time


class Tier(str, Enum):
    """Why a given hosted instance exists."""

    FREE = "free"  # auto-loaded due to popularity, or kept warm as a default
    PAID = "paid"  # spun up to serve a specific paid job


@dataclass
class InstanceInfo:
    """A snapshot of a single hosted model instance, as seen by the scheduler."""

    job_id: str
    model_name: str
    tier: Tier
    state: str  # "initializing" | "distributing" | "ready"
    created_at: float


@dataclass
class LaunchAction:
    model_name: str
    tier: Tier
    reason: str


@dataclass
class SchedulerActions:
    """The set of actions the validator should take this tick."""

    launch: List[LaunchAction] = field(default_factory=list)
    scale_down: List[str] = field(default_factory=list)  # job_ids to remove
    deferred: List[str] = field(default_factory=list)  # reasons for skipped actions


class PublicModelScheduler:
    """
    Tracks per-model demand and turns it into loading / scaling decisions.

    The validator is responsible for:
      - calling `record_request` whenever a generate call comes in
      - calling `plan` periodically to auto-load/scale public models
      - calling `capacity_for_paid_request` when a paid job arrives, to see
        if a free-tier instance needs to be evicted to make room
      - actually executing the InstanceInfo/LaunchAction bookkeeping (this
        class never talks to a node, worker, or model directly)
    """

    def __init__(
        self,
        model_cache: dict,
        default_models: Optional[List[str]] = None,
        save_fn: Optional[Callable[[dict], None]] = None,
        *,
        tracking_days: float = 7,
        min_requests_threshold: int = 10,
        max_auto_models: int = 10,
        max_instances_per_model: int = 4,
        requests_per_instance: int = 15,
        scale_up_cooldown: float = 300,
        min_instance_age_for_scale_down: float = 120,
        free_protection_multiplier: float = 3.0,
        time_fn: Callable[[], float] = time.time,
    ):
        self.model_cache = model_cache
        self.default_models = list(default_models or [])
        self._save_fn = save_fn
        self._time = time_fn

        self.tracking_days = tracking_days
        self.min_requests_threshold = min_requests_threshold
        self.max_auto_models = max_auto_models
        self.max_instances_per_model = max_instances_per_model
        self.requests_per_instance = requests_per_instance
        self.scale_up_cooldown = scale_up_cooldown
        self.min_instance_age_for_scale_down = min_instance_age_for_scale_down
        self.free_protection_multiplier = free_protection_multiplier

        # model_name -> timestamp of last scale-up, to enforce cooldowns
        self._last_scale_up: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Demand tracking
    # ------------------------------------------------------------------

    def ensure_model_entry(self, model_name: str):
        """Make sure a model has a well-formed entry in the shared cache."""
        if model_name not in self.model_cache:
            self.model_cache[model_name] = {
                "distribution": None,
                "demand_metrics": {
                    "request_timestamps": [],
                    "total_requests": 0,
                    "last_accessed": None,
                },
            }

    def _save(self):
        if self._save_fn is not None:
            self._save_fn(self.model_cache)

    def record_request(self, model_name: str):
        """Record a single demand signal (a generate call) for a model."""
        self.ensure_model_entry(model_name)

        now = self._time()
        metrics = self.model_cache[model_name]["demand_metrics"]
        metrics["request_timestamps"].append(now)
        metrics["total_requests"] = metrics.get("total_requests", 0) + 1
        metrics["last_accessed"] = now

        cutoff = now - (self.tracking_days * 24 * 3600)
        metrics["request_timestamps"] = [
            ts for ts in metrics["request_timestamps"] if ts >= cutoff
        ]

        self._save()

    def get_recent_request_count(
        self, model_name: str, days: Optional[float] = None
    ) -> int:
        """Number of requests for a model within the tracking window."""
        if model_name not in self.model_cache:
            return 0

        days = self.tracking_days if days is None else days
        cutoff = self._time() - (days * 24 * 3600)
        timestamps = (
            self.model_cache[model_name]
            .get("demand_metrics", {})
            .get("request_timestamps", [])
        )
        return sum(1 for ts in timestamps if ts >= cutoff)

    def get_popular_models(self) -> List[str]:
        """Models whose recent demand crosses the auto-load threshold, busiest first."""
        popularity = [
            (name, self.get_recent_request_count(name)) for name in self.model_cache
        ]
        popularity = [
            (name, count)
            for name, count in popularity
            if count >= self.min_requests_threshold
        ]
        popularity.sort(key=lambda item: item[1], reverse=True)
        return [name for name, _ in popularity]

    def cleanup_old_requests(self):
        """Trim expired timestamps; drop cold, distribution-less cache entries."""
        cutoff = self._time() - (self.tracking_days * 24 * 3600)
        changed = False

        for model_name in list(self.model_cache.keys()):
            entry = self.model_cache[model_name]
            metrics = entry.get("demand_metrics")
            if not metrics:
                continue

            before = len(metrics["request_timestamps"])
            metrics["request_timestamps"] = [
                ts for ts in metrics["request_timestamps"] if ts >= cutoff
            ]
            after = len(metrics["request_timestamps"])
            if before != after:
                changed = True

            if after == 0 and entry.get("distribution") is None:
                del self.model_cache[model_name]
                changed = True

        if changed:
            self._save()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_considerable_demand(self, model_name: str) -> bool:
        """Whether a free model has enough of its own demand to be protected from eviction."""
        return self.get_recent_request_count(model_name) >= (
            self.min_requests_threshold * self.free_protection_multiplier
        )

    @staticmethod
    def _total_instances(instances: Dict[str, List[InstanceInfo]]) -> int:
        return sum(len(v) for v in instances.values())

    # ------------------------------------------------------------------
    # Paid-request capacity handling
    # ------------------------------------------------------------------

    def capacity_for_paid_request(
        self,
        model_name: str,
        instances: Dict[str, List[InstanceInfo]],
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide whether a new paid job for `model_name` can go ahead right now.

        Returns (allowed, evict_job_id):
          - (True, None)        -> there's free room, just launch
          - (True, <job_id>)     -> evict this free instance first, then launch
          - (False, None)        -> at capacity, nothing evictable; queue/decline
        """
        if self._total_instances(instances) < self.max_auto_models:
            return True, None

        # At global capacity: find the least-valuable free instance to evict.
        # Ranked by the demand of the model it belongs to (ascending), then
        # age (oldest first), so we free the least-missed instance first.
        candidates = []
        for other_model, instance_list in instances.items():
            if other_model == model_name:
                continue
            for inst in instance_list:
                if inst.tier != Tier.FREE:
                    continue
                if other_model in self.default_models:
                    continue
                if self._is_considerable_demand(other_model):
                    continue
                candidates.append(inst)

        if not candidates:
            return False, None

        candidates.sort(
            key=lambda inst: (
                self.get_recent_request_count(inst.model_name),
                inst.created_at,
            )
        )
        return True, candidates[0].job_id

    # ------------------------------------------------------------------
    # Background planning: auto-load + scale up/down
    # ------------------------------------------------------------------

    def plan(
        self, mode: str, instances: Dict[str, List[InstanceInfo]]
    ) -> SchedulerActions:
        """
        Periodic pass that decides which public models to auto-load, scale
        up under pressure, or scale down once idle. Never evicts a running
        job -- that's only ever done via `capacity_for_paid_request`, in
        direct response to a paid job that needs the room.
        """
        actions = SchedulerActions()

        if mode != "public":
            return actions

        total_instances = self._total_instances(instances)
        now = self._time()

        universe = set(self.default_models) | set(instances.keys())
        for name in self.model_cache:
            if self.get_recent_request_count(name) > 0:
                universe.add(name)

        for model_name in universe:
            current = instances.get(model_name, [])
            ready = [inst for inst in current if inst.state == "ready"]
            demand = self.get_recent_request_count(model_name)
            is_default = model_name in self.default_models

            # ---- Ensure a base instance exists ----
            if not current:
                should_load = is_default or demand >= self.min_requests_threshold
                if should_load:
                    if total_instances >= self.max_auto_models:
                        actions.deferred.append(
                            f"{model_name}: wants a base instance but validator is at capacity"
                        )
                    else:
                        actions.launch.append(
                            LaunchAction(
                                model_name, Tier.FREE, "auto-load: popular/default model"
                            )
                        )
                        total_instances += 1
                continue

            # ---- Scale up under sustained pressure ----
            if ready:
                pressure = demand / len(ready)
                at_model_cap = len(current) >= self.max_instances_per_model
                cooldown_elapsed = (
                    now - self._last_scale_up.get(model_name, 0)
                    >= self.scale_up_cooldown
                )

                if pressure > self.requests_per_instance and not at_model_cap:
                    if not cooldown_elapsed:
                        actions.deferred.append(
                            f"{model_name}: under pressure but scale-up cooldown active"
                        )
                    elif total_instances >= self.max_auto_models:
                        actions.deferred.append(
                            f"{model_name}: under pressure but validator is at capacity"
                        )
                    else:
                        actions.launch.append(
                            LaunchAction(
                                model_name,
                                Tier.FREE,
                                f"scale-up: {demand} requests / {len(ready)} instance(s)",
                            )
                        )
                        self._last_scale_up[model_name] = now
                        total_instances += 1
                        continue

            # ---- Scale down idle duplicates ----
            # `current` is non-empty in this branch, so we never scale a
            # model down to zero here -- full teardown of a model that no
            # longer has any traffic is handled elsewhere, by the
            # on-chain job-liveness check in check_node(). This section
            # only trims *excess duplicates* once demand has cooled off.
            floor = 1
            desired = max(floor, math.ceil(demand / self.requests_per_instance))
            desired = min(desired, self.max_instances_per_model)

            if len(current) > desired:
                removable = [
                    inst
                    for inst in current
                    if inst.state == "ready"
                    and (now - inst.created_at) >= self.min_instance_age_for_scale_down
                ]
                # Free instances before paid (protect paid capacity); among
                # those, retire the newest duplicate first -- it was the
                # last one added to cover a burst, so it's the safest to
                # let go of once that burst has passed.
                removable.sort(key=lambda inst: (inst.tier != Tier.FREE, -inst.created_at))

                to_remove = len(current) - desired
                for inst in removable[:to_remove]:
                    actions.scale_down.append(inst.job_id)

        return actions
