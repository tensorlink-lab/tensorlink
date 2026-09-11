from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List
from datetime import datetime, timedelta
import logging
import time
import json
import os

if TYPE_CHECKING:
    from tensorlink.p2p.smart_node import Smartnode

NETWORK_STATS = "logs/network_stats.json"
DHT_STATE = "logs/dht_state.json"
CATEGORIES = ["workers", "validators", "users", "jobs", "proposals"]

DAILY_STAT_DEFAULTS = {
    "workers": 0,
    "validators": 0,
    "users": 0,
    "jobs": 0,
    "proposals": 0,
    "available_capacity": 0,
    "used_capacity": 0,
    "total_capacity": 0,
    "avg_tflops": None,
    "avg_bandwidth_gb_s": None,
    "total_benchmarked": 0,
    "by_backend": {},
    "by_gpu_model": {},
}

WEEKLY_STAT_DEFAULTS = {
    "week_start": 0,
    "week_end": 0,
    "days_count": 0,
    "avg_workers": 0,
    "avg_validators": 0,
    "avg_users": 0,
    "avg_jobs": 0,
    "avg_proposals": 0,
    "avg_available_capacity": 0,
    "avg_used_capacity": 0,
    "avg_total_capacity": 0,
    "avg_tflops": None,
    "avg_bandwidth_gb_s": None,
    "total_benchmarked": 0,
}

THIRTY_DAYS_SECONDS = 60 * 60 * 24 * 30
SEVEN_DAYS_SECONDS = 60 * 60 * 24 * 7
ONE_DAY_SECONDS = 60 * 60 * 24
CLEAN_ARCHIVE_FREQ = 10
MAX_DAILY_STATS = 1093


def _normalize_daily_stat(stat: Dict) -> Dict:
    """Backfill fields on stats written before device/benchmark tracking
    existed, so nothing downstream has to special-case missing keys."""
    return {**DAILY_STAT_DEFAULTS, **stat}


def _normalize_weekly_stat(stat: Dict) -> Dict:
    """Backfill fields on weekly rollups written before this schema
    existed (or partially written), so downstream code never has to
    special-case missing keys."""
    return {**WEEKLY_STAT_DEFAULTS, **stat}


def _load_historical_stats() -> Dict:
    """Load historical statistics from file."""
    if os.path.exists(NETWORK_STATS):
        try:
            with open(NETWORK_STATS, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    # Default structure
    return {
        "daily": [],  # Last 90 days of data
        "weekly": [],  # Last 52 weeks of weekly aggregates
        "monthly": [],  # Last 1200 months (100 years) of monthly aggregates
    }


def _is_entity_current(entity_data, cutoff_time: float = None) -> bool:
    """Check if entity should be included based on last_seen timestamp."""
    if not entity_data or not isinstance(entity_data, dict):
        return True

    last_seen = entity_data.get("last_seen")
    if last_seen is None:
        return True

    try:
        if isinstance(last_seen, str):
            last_seen = float(last_seen)

        if cutoff_time is None:
            cutoff_time = time.time() - THIRTY_DAYS_SECONDS

        return last_seen >= cutoff_time
    except (ValueError, TypeError):
        return True


def _is_entity_active_today(entity_data, target_date: datetime) -> bool:
    """Check if an entity was active (last_seen) today."""
    if not entity_data or not isinstance(entity_data, dict):
        return False

    last_seen = entity_data.get("last_seen")
    if last_seen is None:
        return False

    try:
        if isinstance(last_seen, str):
            last_seen = float(last_seen)

        entity_date = datetime.fromtimestamp(last_seen).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return entity_date == target_date

    except (ValueError, TypeError):
        return False


def _count_entities_for_date(entities: Dict, target_date: datetime) -> int:
    """Count entities active on a specific date."""
    target_ts = target_date.timestamp()
    count = 0

    for entity_data in entities.values():
        if not isinstance(entity_data, dict):
            continue

        last_seen = entity_data.get("last_seen")
        if last_seen is None:
            continue

        try:
            if isinstance(last_seen, str):
                last_seen = float(last_seen)

            if abs(last_seen - target_ts) <= ONE_DAY_SECONDS:
                count += 1
        except (ValueError, TypeError):
            continue

    return count


def _filter_old_entities(
    entities: Dict, cutoff_seconds: int = THIRTY_DAYS_SECONDS
) -> Dict:
    """Filter entities older than specified cutoff time."""
    cutoff_time = time.time() - cutoff_seconds
    filtered = {}

    for entity_id, entity_data in entities.items():
        if _is_entity_current(entity_data, cutoff_time):
            filtered[entity_id] = entity_data

    return filtered


def _fill_missing_daily_days(daily_stats: List[Dict]) -> List[Dict]:
    """Ensure daily stats have no missing days by filling gaps with zeros."""
    if not daily_stats:
        return []

    # Convert to date objects
    parsed = {datetime.fromtimestamp(s["timestamp"]).date(): s for s in daily_stats}
    min_day = min(parsed.keys())
    max_day = max(parsed.keys())

    filled = []
    current = min_day
    while current <= max_day:
        if current in parsed:
            filled.append(_normalize_daily_stat(parsed[current]))
        else:
            # Insert zero-filled entry
            filled.append(
                {
                    "date": current.strftime("%Y-%m-%d"),
                    "timestamp": datetime(
                        current.year, current.month, current.day
                    ).timestamp(),
                    "last_updated": 0,
                    **DAILY_STAT_DEFAULTS,
                }
            )

        current = current + timedelta(days=1)

    # Always sorted
    filled.sort(key=lambda x: x["timestamp"])
    return filled


class Keeper:
    def __init__(self, node: Smartnode):
        """Cleans up old data stored in node and keeps a log of network statistics that maintains."""
        self.node = node
        self.network_stats = _load_historical_stats()

        # Memory cache for querying
        self.archive_cache = self._load_archive_from_disk()
        self.archive_cache_time = time.time()
        self.current_state_cache = None
        self.current_state_cache_time = 0

        # Status cache for API
        self._status_cache = None
        self._status_cache_time = 0

        # Cleaning update interval
        self.clean_ticker = 0
        self.last_daily_stats_update = 0

    def _load_archive_from_disk(self) -> Dict:
        """Load archive into memory (done on startup)"""
        default = {cat: {} for cat in CATEGORIES}
        default.update(
            {"available_capacity": 0, "used_capacity": 0, "total_capacity": 0}
        )

        if not os.path.exists(DHT_STATE):
            return default

        try:
            with open(DHT_STATE, "r") as f:
                data = json.load(f)

            # Ensure all required fields exist
            for key in ["available_capacity", "used_capacity", "total_capacity"]:
                if key not in data:
                    data[key] = 0

            return data

        except json.JSONDecodeError:
            self.node.debug_print(
                "Archive file corrupted, using defaults.",
                level=logging.ERROR,
                colour="red",
                tag="Keeper",
            )
            return default

    def _get_current_state(self, force_refresh: bool = False) -> Dict:
        """Get current state with caching (refreshes every 60 seconds)."""
        current_time = time.time()

        if (
            not force_refresh
            and self.current_state_cache
            and (current_time - self.current_state_cache_time) < 60
        ):
            return self.current_state_cache

        # Build fresh state
        state = {cat: {} for cat in CATEGORIES}
        state["timestamp"] = current_time

        # Load and count entities in one pass
        cutoff_time = current_time - THIRTY_DAYS_SECONDS

        for category in CATEGORIES:
            collection = getattr(self.node, category)
            for entity_id in collection:
                entity_data = self.node.dht.query(entity_id)
                if _is_entity_current(entity_data, cutoff_time):
                    state[category][entity_id] = entity_data

        # Add capacity info
        capacity = self._calculate_worker_capacities()
        state.update(capacity)

        # Cache it
        self.current_state_cache = state
        self.current_state_cache_time = current_time

        return state

    def _get_merged_entities(self) -> Dict:
        """Merge current + archived entities efficiently."""
        current = self._get_current_state()

        return {
            category: {**self.archive_cache.get(category, {}), **current[category]}
            for category in CATEGORIES
        }

    def _archive_daily_to_weekly(self):
        """Archive daily statistics older than 90 days into weekly aggregates."""
        current_time = time.time()
        two_years_ago = current_time - THIRTY_DAYS_SECONDS * 24
        today = datetime.fromtimestamp(current_time).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today_timestamp = today.timestamp()

        daily_to_archive = [
            stat
            for stat in self.network_stats["daily"]
            if stat["timestamp"] < two_years_ago and stat["timestamp"] < today_timestamp
        ]

        if not daily_to_archive:
            return

        # Group daily stats by week (ISO week)
        weekly_groups = {}
        for daily_stat in daily_to_archive:
            stat_date = datetime.fromtimestamp(daily_stat["timestamp"])
            iso_year, iso_week, _ = stat_date.isocalendar()
            week_key = f"{iso_year}-W{iso_week:02d}"
            weekly_groups.setdefault(week_key, []).append(daily_stat)

        # --- everything below is now OUTSIDE the daily_stat loop ---
        metric_keys = [
            "workers",
            "validators",
            "users",
            "jobs",
            "proposals",
            "available_capacity",
            "used_capacity",
            "total_capacity",
        ]

        for week_key, stats in weekly_groups.items():
            # Skip if already archived
            if any(w["week"] == week_key for w in self.network_stats["weekly"]):
                continue

            n = len(stats)
            tflops_vals = [
                s.get("avg_tflops") for s in stats if s.get("avg_tflops") is not None
            ]
            bandwidth_vals = [
                s.get("avg_bandwidth_gb_s")
                for s in stats
                if s.get("avg_bandwidth_gb_s") is not None
            ]

            weekly_stat = {
                "week": week_key,
                "week_start": min(s["timestamp"] for s in stats),
                "week_end": max(s["timestamp"] for s in stats),
                "days_count": n,
                **{
                    f"avg_{key}": sum(s.get(key, 0) for s in stats) / n
                    for key in metric_keys
                },
                "avg_tflops": (
                    round(sum(tflops_vals) / len(tflops_vals), 2)
                    if tflops_vals
                    else None
                ),
                "avg_bandwidth_gb_s": (
                    round(sum(bandwidth_vals) / len(bandwidth_vals), 2)
                    if bandwidth_vals
                    else None
                ),
                "total_benchmarked": sum(s.get("total_benchmarked", 0) for s in stats),
            }
            self.network_stats["weekly"].append(weekly_stat)

        # Remove archived daily stats
        self.network_stats["daily"] = [
            s
            for s in self.network_stats["daily"]
            if s["timestamp"] >= two_years_ago or s["timestamp"] >= today_timestamp
        ]

        # Keep last 104 weeks
        self.network_stats["weekly"].sort(key=lambda x: x["week_start"])
        self.network_stats["weekly"] = self.network_stats["weekly"][-104:]

        self.node.debug_print(
            f"Archived {len(daily_to_archive)} daily stats into {len(weekly_groups)} weeks",
            level=logging.INFO,
            colour="blue",
            tag="Keeper",
        )

    def _update_daily_statistics(self):
        current_time = time.time()
        current_date = datetime.fromtimestamp(current_time).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today_str = current_date.strftime("%Y-%m-%d")

        # Find or create todays statistic
        existing_stat_index = None
        for i, stat in enumerate(self.network_stats["daily"]):
            if stat.get("date") == today_str:
                existing_stat_index = i
                break

        try:
            merged_data = self._get_merged_entities()
            capacity = self._calculate_worker_capacities()
            device_breakdown = self.get_device_breakdown()

            daily_stat = {
                "date": today_str,
                "timestamp": current_date.timestamp(),
                "last_updated": current_time,
                **{
                    cat: _count_entities_for_date(merged_data[cat], current_date)
                    for cat in CATEGORIES
                },
                **capacity,
                **device_breakdown,
            }

            # Adjust validators count (add 1 for self)
            daily_stat["validators"] += 1

            if existing_stat_index is not None:
                self.network_stats["daily"][existing_stat_index] = daily_stat
                action = "Updated"

            else:
                self.network_stats["daily"].append(daily_stat)
                self.network_stats["daily"].sort(key=lambda x: x["timestamp"])
                action = "Created"

            self._archive_daily_to_weekly()
            self._save_network_stats()

            self.node.debug_print(
                f"{action} daily stats - W:{daily_stat['workers']} V:{daily_stat['validators']} "
                f"U:{daily_stat['users']} J:{daily_stat['jobs']} P:{daily_stat['proposals']}",
                level=logging.INFO,
                colour="cyan",
                tag="Keeper",
            )

            self.last_daily_stats_update = current_time

        except Exception as e:
            self.node.debug_print(
                f"Error updating daily stats: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="Keeper",
            )

    def _calculate_worker_capacities(self) -> Dict:
        """Calculate total available and used capacity from workers."""
        total_available = 0
        total_used = 0

        if hasattr(self.node, "all_workers"):
            for worker_data in self.node.all_workers.values():
                total = worker_data.get('total_gpu_memory', 0)
                available = worker_data.get('gpu_memory', 0)
                total_available += available
                total_used += total - available

        return {
            'available_capacity': total_available,
            'used_capacity': total_used,
            'total_capacity': total_available + total_used,
        }

    def _save_network_stats(self):
        """Save network statistics to file."""
        os.makedirs(os.path.dirname(NETWORK_STATS), exist_ok=True)
        with open(NETWORK_STATS, "w") as f:
            json.dump(self.network_stats, f, indent=4)

    def get_current_statistics(self) -> Dict:
        """
        Get the most current statistics, including live-updated daily data for today.

        Returns:
            Dictionary with the most up-to-date statistics
        """
        self._update_daily_statistics()

        today_str = datetime.now().strftime("%Y-%m-%d")
        today_stats = next(
            (s for s in self.network_stats["daily"] if s.get("date") == today_str), None
        )

        if today_stats:
            return {**today_stats, "is_live": True}

        # Fallback to summary
        summary = self.get_network_summary()
        current = summary.get("current", {})
        return {
            "date": today_str,
            "timestamp": time.time(),
            "last_updated": time.time(),
            **current,
            "is_live": True,
        }

    def get_daily_statistics(self, days: int = 30) -> List[Dict]:
        """Get daily statistics for last N days."""
        if days <= 0:
            return []
        days = min(days, MAX_DAILY_STATS)
        return [_normalize_daily_stat(s) for s in self.network_stats["daily"][-days:]]

    def get_weekly_statistics(self, weeks: int = 12) -> List[Dict]:
        """Get weekly statistics for last N weeks."""
        if weeks <= 0:
            return []
        return [
            _normalize_weekly_stat(s) for s in self.network_stats["weekly"][-weeks:]
        ]

    def get_network_summary(self) -> Dict:
        """Get summary of current network statistics."""
        try:
            merged = self._get_merged_entities()
            current_date = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )

            current_counts = {
                cat: _count_entities_for_date(merged[cat], current_date)
                for cat in CATEGORIES
            }

            # Add capacity and adjust validators
            current_counts.update(self._calculate_worker_capacities())

            return {
                "current": current_counts,
                "recent_daily": self.get_daily_statistics(7),
                "recent_weekly": self.get_weekly_statistics(4),
                "total_days_tracked": len(self.network_stats["daily"]),
                "total_weeks_archived": len(self.network_stats["weekly"]),
                "retention_policy": {"daily_days": 90, "weekly_weeks": 104},
            }

        except Exception as e:
            self.node.debug_print(
                f"Error generating network summary: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="Keeper",
            )
            return {"error": str(e)}

    def get_network_status(
        self,
        days: int = 30,
        include_weekly: bool = False,
        include_summary: bool = True,
        include_device: bool = True,
        cache_duration: int = 300,
    ) -> Dict:
        """Get network statistics with caching for API consumption."""
        current_time = time.time()

        if (
            self._status_cache
            and (current_time - self._status_cache_time) < cache_duration
        ):
            return self._status_cache

        try:
            raw_daily = self.get_daily_statistics(days)
            daily_stats = _fill_missing_daily_days(raw_daily)

            result = {
                "daily": {
                    "labels": [s["date"] for s in daily_stats],
                    "datasets": {
                        cat: [s.get(cat, 0) for s in daily_stats]
                        for cat in CATEGORIES + ["total_capacity", "used_capacity"]
                    },
                    "timestamps": [s["timestamp"] for s in daily_stats],
                }
            }

            if include_weekly:
                weekly_stats = self.get_weekly_statistics(days // 7)
                result["weekly"] = {
                    "labels": [s["week"] for s in weekly_stats],
                    "datasets": {
                        f"avg_{cat}": [s.get(f"avg_{cat}", 0) for s in weekly_stats]
                        for cat in CATEGORIES + ["total_capacity", "used_capacity"]
                    },
                    "week_starts": [s["week_start"] for s in weekly_stats],
                    "week_ends": [s["week_end"] for s in weekly_stats],
                }

            if include_device:
                result["device"] = {
                    "labels": [s["date"] for s in daily_stats],
                    "datasets": {
                        "avg_tflops": [s.get("avg_tflops") for s in daily_stats],
                        "avg_bandwidth_gb_s": [
                            s.get("avg_bandwidth_gb_s") for s in daily_stats
                        ],
                        "total_benchmarked": [
                            s.get("total_benchmarked", 0) for s in daily_stats
                        ],
                    },
                    "backend_breakdown_by_day": {
                        s["date"]: s.get("by_backend", {}) for s in daily_stats
                    },
                    "gpu_model_breakdown_by_day": {
                        s["date"]: s.get("by_gpu_model", {}) for s in daily_stats
                    },
                    "current": self.get_device_breakdown(),
                    "timestamps": [s["timestamp"] for s in daily_stats],
                }

            if include_summary:
                result["summary"] = self.get_network_summary()

            result["metadata"] = {
                "total_days_available": len(self.network_stats["daily"]),
                "total_weeks_available": len(self.network_stats["weekly"]),
                "requested_days": days,
                "generated_at": current_time,
                "generated_at_iso": datetime.now().isoformat(),
            }

            self._status_cache = result
            self._status_cache_time = current_time
            return result

        except Exception as e:
            self.node.debug_print(
                f"Error getting network status: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="Keeper",
            )
            return {"error": str(e)}

    def clean_old_data(self):
        """Clean jobs and users older than 7 days from saved files. Proposals are never deleted."""
        try:
            entities_cleaned = 0

            # Clean archive and refresh cache
            if os.path.exists(DHT_STATE):
                with open(DHT_STATE, "r") as f:
                    archive = json.load(f)

                # Only clean jobs and users (7 day cutoff), never proposals
                for cat in ["jobs", "users"]:
                    if cat in archive:
                        original = len(archive[cat])
                        archive[cat] = _filter_old_entities(
                            archive[cat], SEVEN_DAYS_SECONDS
                        )
                        entities_cleaned += original - len(archive[cat])

                with open(DHT_STATE, "w") as f:
                    json.dump(archive, f, indent=4)

                # Refresh cache
                self.archive_cache = archive
                self.archive_cache_time = time.time()

            if entities_cleaned > 0:
                self.node.debug_print(
                    f"Cleaned {entities_cleaned} old jobs/users (>7 days)",
                    level=logging.INFO,
                    colour="green",
                    tag="Keeper",
                )

        except Exception as e:
            self.node.debug_print(
                f"Error cleaning old data: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="Keeper",
            )

    def write_state(self):
        """Write current DHT state to archive file."""
        try:
            current = self._get_current_state(force_refresh=True)

            # Update archive
            for cat in CATEGORIES:
                self.archive_cache[cat].update(current[cat])

            self.archive_cache.update(
                {
                    k: current[k]
                    for k in [
                        "available_capacity",
                        "used_capacity",
                        "total_capacity",
                        "timestamp",
                    ]
                }
            )

            os.makedirs(os.path.dirname(DHT_STATE), exist_ok=True)
            with open(DHT_STATE, "w") as f:
                json.dump(self.archive_cache, f, indent=4)

            self._update_daily_statistics()

            self.node.debug_print(
                "DHT state saved to archive",
                level=logging.INFO,
                colour="green",
                tag="Keeper",
            )

        except Exception as e:
            self.node.debug_print(
                f"Error saving DHT state: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="Keeper",
            )

    def load_previous_state(self):
        """Load DHT state from archive file."""
        if not os.path.exists(DHT_STATE):
            self.node.debug_print(
                "No DHT state file found", level=logging.INFO, tag="Keeper"
            )
            return

        try:
            with open(DHT_STATE, "r") as f:
                state = json.load(f)

            for category in CATEGORIES:
                if category in state and category != "timestamp":
                    # For jobs and users, only load recent ones (7 days)
                    if category in ["jobs", "users"]:
                        filtered = _filter_old_entities(
                            state[category], SEVEN_DAYS_SECONDS
                        )
                    else:
                        # For workers, validators, proposals use 30 days
                        filtered = _filter_old_entities(
                            state[category], THIRTY_DAYS_SECONDS
                        )

                    self.node.dht.routing_table.update(filtered)

                    if category == "proposals":
                        self.node.proposals = list(filtered.keys())

            self.node.debug_print(
                "DHT state loaded successfully from archive",
                level=logging.INFO,
                tag="Keeper",
            )

        except Exception as e:
            self.node.debug_print(
                f"Error loading DHT state: {e}",
                colour="bright_red",
                level=logging.ERROR,
                tag="Keeper",
            )

    def clean_node(self):
        """Clean up inactive nodes and old jobs in node memory"""

        def clean_nodes(nodes):
            to_remove = []

            # Iterate thru all recently stored connections and delete inactive ones
            for node_id in nodes:
                if node_id not in self.node.nodes:
                    to_remove.append(node_id)
                elif self.node.nodes[node_id].terminate_flag.is_set():
                    role = self.node.nodes[node_id].role
                    to_remove.append(node_id)
                    del self.node.nodes[node_id]

                    # If we are a validator deleting a worker, remove it from the worker stats
                    if role.startswith("W") and hasattr(self.node, "all_workers"):
                        del self.node.all_workers[node_id]

            for node_id in to_remove:
                nodes.remove(node_id)

        clean_nodes(self.node.workers)
        clean_nodes(self.node.validators)
        clean_nodes(self.node.users)

        # Periodically clean old archived data
        if self.clean_ticker >= CLEAN_ARCHIVE_FREQ:
            self.clean_old_data()
            self.clean_ticker = 0

        self.clean_ticker += 1

    def get_proposals(self, days: int | None = None, limit: int | None = None) -> Dict:
        """
        Retrieve proposals from the archive cache.

        Args:
            days (int, optional): Only include proposals from the last N days.
            limit (int, optional): Maximum number of proposals to return.

        Returns:
            Dict[str, Dict]: {proposal_id: proposal_data} sorted and filtered.
        """
        proposals = self.archive_cache.get("proposals", {})
        if not proposals:
            return {}

        proposals_list = list(proposals.items())

        # Filter by recency if days specified
        if days is not None:
            cutoff = time.time() - (days * 24 * 60 * 60)
            proposals_list = [
                (pid, p)
                for pid, p in proposals_list
                if isinstance(p, dict) and float(p.get("timestamp", 0)) >= cutoff
            ]

        # Sort by chosen field
        def sort_key(item):
            data = item[1]
            return data.get("distribution_id", 0)

        proposals_list.sort(key=sort_key)
        if limit is not None:
            proposals_list = proposals_list[:limit]

        # Return as dict with proposal_id keys
        return {pid: p for pid, p in proposals_list}

    def clear_persistent_state(
        self,
        *,
        clear_files: bool = True,
        clear_memory: bool = True,
        clear_network_stats: bool = True,
    ):
        """
        Fully reset DHT + keeper state.

        Args:
            clear_files: Delete on-disk state files (DHT_STATE, NETWORK_STATS)
            clear_memory: Clear in-memory caches and node/DHT state
            clear_network_stats: Reset historical statistics in memory
        """
        try:
            # Clear on-disk files
            if clear_files:
                for path in [DHT_STATE, NETWORK_STATS]:
                    if os.path.exists(path):
                        os.remove(path)
                        self.node.debug_print(
                            f"Deleted state file: {path}",
                            level=logging.INFO,
                            colour="yellow",
                            tag="Keeper",
                        )

            # Reset in-memory caches
            if clear_memory:
                # Reset archive + caches
                self.archive_cache = {cat: {} for cat in CATEGORIES}
                self.archive_cache.update(
                    {
                        "available_capacity": 0,
                        "used_capacity": 0,
                        "total_capacity": 0,
                    }
                )
                self.archive_cache_time = time.time()

                self.current_state_cache = None
                self.current_state_cache_time = 0

                self._status_cache = None
                self._status_cache_time = 0

                # Clear node-local state (safe no-ops if missing)
                if hasattr(self.node, "dht"):
                    self.node.dht.routing_table.clear()

                for attr in ["workers", "validators", "users", "proposals"]:
                    if hasattr(self.node, attr):
                        setattr(self.node, attr, [] if attr == "proposals" else {})

                if hasattr(self.node, "all_workers"):
                    self.node.all_workers.clear()

            # Reset statistics
            if clear_network_stats:
                self.network_stats = {
                    "daily": [],
                    "weekly": [],
                    "monthly": [],
                }
                self.last_daily_stats_update = 0

            self.clean_ticker = 0

            self.node.debug_print(
                "Persistent state fully reset",
                level=logging.INFO,
                colour="green",
                tag="Keeper",
            )

        except Exception as e:
            self.node.debug_print(
                f"Error clearing persistent state: {e}",
                level=logging.ERROR,
                colour="bright_red",
                tag="Keeper",
            )

    def get_device_breakdown(self) -> Dict:
        """Summarize device types and benchmark results across all_workers."""
        by_backend = {}
        by_gpu_model = {}
        tflops_values = []
        bandwidth_values = []

        if hasattr(self.node, "all_workers"):
            for worker_data in self.node.all_workers.values():
                device = worker_data.get("device") or {}
                benchmark = worker_data.get("benchmark") or {}

                backend = device.get("backend", "unknown")
                by_backend[backend] = by_backend.get(backend, 0) + 1

                name = device.get("name", "unknown")
                by_gpu_model[name] = by_gpu_model.get(name, 0) + 1

                compute = benchmark.get("compute") or {}
                if "tflops" in compute:
                    tflops_values.append(compute["tflops"])

                memory = benchmark.get("memory") or {}
                if "bandwidth_gb_s" in memory:
                    bandwidth_values.append(memory["bandwidth_gb_s"])

        return {
            "by_backend": by_backend,
            "by_gpu_model": by_gpu_model,
            "avg_tflops": (
                round(sum(tflops_values) / len(tflops_values), 2)
                if tflops_values
                else None
            ),
            "avg_bandwidth_gb_s": (
                round(sum(bandwidth_values) / len(bandwidth_values), 2)
                if bandwidth_values
                else None
            ),
            "total_benchmarked": len(tflops_values),
        }
