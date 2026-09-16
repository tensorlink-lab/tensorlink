from collections import defaultdict
from typing import Union, Optional, List
import logging
import re

import torch
import torch.nn as nn

from tensorlink.ml.utils.gpu_benchmark import estimate_memory
from tensorlink.ml.utils.injector import find_loop_in_module_hierarchy
from tensorlink.ml.utils.loading import load_model_skeleton
from tensorlink.ml.utils import resolve_module_from_path  # noqa: F401


class AssignmentError(Exception):
    """Raised when a module cannot be assigned to any worker."""


def _create_grouped_entry(parent_path: str, group: list) -> dict:
    """Create a single config entry for a group of consecutive layers."""
    if len(group) == 1:
        _, path, cfg = group[0]
        return {path: cfg}

    layer_indices = [idx for idx, _, _ in group]
    paths = [path for _, path, _ in group]
    configs = [cfg for _, _, cfg in group]

    start_idx = min(layer_indices)
    end_idx = max(layer_indices)
    grouped_path = f"{parent_path}{start_idx}-{end_idx}"

    total_memory = sum(cfg.get("memory", 0) for cfg in configs)
    worker = configs[0]["assigned_workers"][0]

    grouped_config = {
        "type": "offloaded_group",
        "name": configs[0].get("name", ""),
        "assigned_workers": [worker],
        "layer_range": (start_idx, end_idx),
        "layer_paths": paths,
        "memory": total_memory,
        "module": configs[0].get("module", ""),
        "training": configs[0].get("training", False),
        "optimizer_type": configs[0].get("optimizer_type", "adam"),
        "num_layers": len(group),
    }

    if "parent_module_path" in configs[0]:
        grouped_config["parent_module_path"] = configs[0]["parent_module_path"]

    return {grouped_path: grouped_config}


def _extract_layer_groups(config: dict) -> dict:
    """Bucket offloaded, layer-shaped paths (e.g. ``model.layers.3``) by their
    shared parent prefix (e.g. ``model.layers.``), preserving layer index and
    original config for each entry.
    """
    layer_groups = defaultdict(list)

    for path, cfg in config.items():
        if cfg.get("type") != "offloaded":
            continue

        match = re.match(r'^(.+\.)(\d+)$', path)
        if not match:
            continue

        parent_path, layer_idx = match.group(1), int(match.group(2))
        layer_groups[parent_path].append((layer_idx, path, cfg))

    return layer_groups


def _group_consecutive_same_worker(
    parent_path: str, layers: list, new_config: dict
) -> set:
    """Merge consecutive same-worker layers under 'parent_path' into grouped
    entries in 'new_config'. Returns the set of original paths consumed.
    """
    processed_paths = set()
    layers = sorted(layers, key=lambda entry: entry[0])

    current_group = []
    current_worker = None

    for layer_idx, path, cfg in layers:
        worker = cfg["assigned_workers"][0] if cfg["assigned_workers"] else None

        if worker == current_worker and current_group:
            current_group.append((layer_idx, path, cfg))
            continue

        if current_group:
            new_config.update(_create_grouped_entry(parent_path, current_group))
            processed_paths.update(p for _, p, _ in current_group)

        current_group = [(layer_idx, path, cfg)]
        current_worker = worker

    if current_group:
        new_config.update(_create_grouped_entry(parent_path, current_group))
        processed_paths.update(p for _, p, _ in current_group)

    return processed_paths


def _group_sequential_layers(config: dict) -> dict:
    """
    Group consecutive layers assigned to the same worker into single entries.

    For example:
        model.layers.0 -> worker1
        model.layers.1 -> worker1
        model.layers.2 -> worker1

    Becomes:
        model.layers.0-2 -> worker1
    """
    layer_groups = _extract_layer_groups(config)

    new_config = {}
    processed_paths = set()
    for parent_path, layers in layer_groups.items():
        processed_paths |= _group_consecutive_same_worker(
            parent_path, layers, new_config
        )

    # Add all non-layer modules that weren't grouped
    for path, cfg in config.items():
        if path not in processed_paths:
            new_config[path] = cfg

    return new_config


def _module_type_name(module) -> str:
    """Short, human-readable class name for a module (or pass strings through)."""
    return module if isinstance(module, str) else type(module).__name__


def _is_loop_iterable_module(module: nn.Module, module_path: str) -> bool:
    """
    Detect if this module is iterated over in a loop during the forward pass.
    Uses the same loop detection logic as the injector.

    Returns:
        True if the module is iterated in a loop, False otherwise
    """
    try:
        # Only check this level; a ValueError means no loop was found here.
        find_loop_in_module_hierarchy(module, max_depth=1)
        return True
    except ValueError:
        return isinstance(module, nn.ModuleList)


def _log_assignment_summary(config: dict, workers_state: dict):
    """Log a summary of the final assignment after configuration is complete."""
    print("\n" + "=" * 80)
    print("ASSIGNMENT SUMMARY:")
    print("=" * 80)

    worker_assignments = defaultdict(list)
    for module_path, module_config in config.items():
        if "offloaded" in module_config.get("type", ""):
            worker_id = module_config["assigned_workers"][0]
            worker_assignments[worker_id].append(
                {
                    "path": module_path,
                    "memory": module_config.get("memory", 0),
                    "module_type": module_config.get("module", "Unknown"),
                }
            )

    for worker_id in sorted(worker_assignments.keys()):
        assignments = worker_assignments[worker_id]
        total_memory = sum(a["memory"] for a in assignments)

        print(f"\n{worker_id}:")
        print(f"  Total Memory: {total_memory / 1e6:.2f}MB")
        print(f"  Remaining: {workers_state[worker_id]['gpu_memory'] / 1e6:.2f}MB")
        print(f"  Modules ({len(assignments)}):")

        for assignment in assignments:
            print(f"    • {assignment['path']}")
            mem_mb = assignment["memory"] / 1e6
            print(f"      [{assignment['module_type']}] - {mem_mb:.2f}MB")

    unassigned = [
        path for path, cfg in config.items() if cfg.get("type") == "unassigned"
    ]
    if unassigned:
        print(f"\n⚠ UNASSIGNED MODULES ({len(unassigned)}):")
        for path in unassigned:
            print(f"  • {path}")

    print("=" * 80 + "\n")


def _estimate_module_memory(
    module, training, max_seq_len, optimizer_type, batch_size, depth, count_activations
) -> float:
    """Estimate this module's memory footprint, dropping activation memory
    for non-root modules when the caller doesn't want it counted twice.
    """
    memory, breakdown = estimate_memory(
        module,
        training=training,
        seq_length=max_seq_len,
        optimizer_type=optimizer_type,
        batch_size=batch_size,
        recursive=True,
        count_activations=True,
        include_kv_cache=(depth == 0),
    )
    if not count_activations:
        memory -= breakdown.get("activations", 0)
    return memory


def _resolve_force_host(
    *,
    module,
    module_path,
    depth,
    input_obfuscation,
    obfuscation_layer_assigned,
    tied_embed_path,
    tied_lm_head_path,
    host_max_memory_bytes,
):
    """
    Decide whether this module must be forced onto the host, either
    because it's the input-obfuscation boundary layer or because it's one
    half of a tied embedding pair.

    Returns:
        (force_host, obfuscation_layer_assigned)
    """
    force_host = False

    if input_obfuscation and not obfuscation_layer_assigned:
        # Keep the first substantial layer on host for input obfuscation,
        # so raw inputs are transformed before being sent to workers.
        has_params = any(True for _ in module.parameters(recurse=False))
        has_no_children = len(list(module.children())) == 0

        if has_params and has_no_children:
            # Leaf layer with parameters: a good obfuscation-layer candidate.
            force_host = True
            obfuscation_layer_assigned = True
        elif depth <= 1:
            # At shallow depth, force it even for containers so we still
            # capture embedding layers or other initial processing.
            force_host = True

        if force_host and host_max_memory_bytes == 0:
            raise ValueError(
                "input_obfuscation=True requires host_max_memory_bytes > 0 "
                "to keep the input transformation layer on the host."
            )

    # Force host loading for tied embeddings to avoid weight duplication
    # across workers.
    force_host = force_host or (
        (tied_embed_path and module_path == tied_embed_path)
        or (tied_lm_head_path and module_path == tied_lm_head_path)
    )

    return force_host, obfuscation_layer_assigned


class ModelParser:
    """
    Parses a PyTorch model and constructs a distributed execution configuration
    for Tensorlink by analyzing module structure, memory requirements, and
    forward-pass behavior. It will assign individual models, modules, or groups
    of sequential layers in the model

    The ModelParser is responsible for:
    - Walking the module hierarchy.
    - Estimating memory usage per submodule.
    - Assigning modules to workers or host.
    - Detecting and rewriting forward loops for offloaded execution.
    - Producing a configuration graph used by DistributedModel.

    This class does not execute the model itself, but prepares the metadata and
    transformed forward methods required for distributed inference or training.
    """

    def __init__(self, user_memory: int = 0, verbose=False):
        """
        Initialize a ModelParser instance.

        Parameters
        ----------
        verbose : bool, optional
            If True, enables verbose logging during model parsing, memory estimation,
            and assignment steps. Default is False.
        """
        self.model_name = ""
        self.assigned_workers = defaultdict(list)
        self.assigned_memory = 0
        self.verbose = verbose
        self.module_paths = {}  # Track all module paths
        self._host_max_module_bytes = 0
        self._tied_worker = None  # worker id holding the first-placed tied module
        self._tied_on_host = False  # True if the first-placed tied module went to host
        self._tied_reserved = (
            False  # True once we've pre-reserved room for the counterpart
        )

    def _log(self, indent: str, message: str) -> None:
        """Print ``message`` (prefixed with ``indent``) when verbose logging is on."""
        if self.verbose:
            print(f"{indent}{message}")

    def create_distributed_config(
        self,
        model: Union[nn.Module, str],
        workers: dict,
        training: bool,
        trusted: bool,
        dtype: torch.dtype = torch.float32,
        input_obfuscation: bool = False,
        optimizer_type: str = "adam",
        optimizer_spec: Optional[dict] = None,
        host_max_memory_bytes: int = 0,
        host_max_module_bytes: int = 0,
        host_max_depth: int = 1,
        max_offload_depth: int = 3,
        max_seq_len: int = 4096,
        batch_size: int = 1,
        model_type: str = "chat",
        force_tied_to_host: bool = True,
    ):
        """
        Build a distributed execution configuration for a model by assigning its
        submodules across available workers and optionally the local host.

        This method recursively walks the model graph, estimates memory usage for
        each submodule (parameters, optimizer state, activations, and KV cache),
        and determines whether the module should be:
          - kept on the local host,
          - fully offloaded to a remote worker,
          - split into children and recursively assigned, or
          - marked as unassigned.

        The result is a config dictionary describing how the model should be
        partitioned for distributed inference or training.

        Args:
            model : Union[nn.Module, str]
                Either a PyTorch model instance or a HuggingFace model name. If a string
                is provided, the model is instantiated with empty weights using
                `AutoConfig` to avoid loading parameters into memory.

            workers : dict
                Mapping of worker_id -> worker metadata. Each worker entry must contain
                at least:
                    {
                        "gpu_memory": <bytes available on worker GPU>
                    }
                This memory is decremented as modules are assigned.

            training : bool
                Whether the configuration is for training or inference. Training mode
                increases memory estimates to include gradients, optimizer state, and
                activation storage.

            trusted : bool
                Indicates whether workers are trusted. Used for downstream logic such as
                security policies, encryption, or obfuscation decisions.

            dtype: torch.dtype, optional (default=torch.float32)

            input_obfuscation : bool, optional (default=False)
                Whether inputs should be obfuscated when sent to workers. This flag is
                propagated into the distributed config for runtime enforcement.

            optimizer_type : str, optional (default="adam")
                Optimizer type used for memory estimation (e.g. "adam", "sgd"). This
                affects optimizer state size during training.

            optimizer_spec : dict, optional
                Extra optimizer configuration to attach to each assigned module
                (e.g. learning rate, betas, weight decay). Stored in the config and
                passed to workers.

            host_max_memory_bytes : int, optional (default=0)
                Maximum number of bytes the local host is allowed to consume for loading
                small submodules. If 0, the host will not keep modules locally.

            host_max_module_bytes : int, optional (default=0)
                Maximum bytes size the local host is allowed to consume for an individual
                submodule. If 0, the host will consider module size.

            host_max_depth : int, optional (default=2)
                Maximum recursion depth at which the host is allowed to keep modules.
                Prevents deep layers from being pinned locally.

            max_offload_depth : int, optional (default=3)
                Maximum recursion depth for offloading. If exceeded, the module is marked
                as unassigned and an AssignmentError may be raised in verbose mode.

            max_seq_len : int, optional (default=4096)
                Maximum sequence length used for estimating activation and KV cache
                memory during inference or training.

            batch_size : int, optional (default=1)
                Batch size used for memory estimation of activations and optimizer state.

            model_type : str, optional (default="chat")
                Logical model type (e.g. "chat", "vision", "embedding"). Stored in the
                config and used by downstream execution logic.

            force_tied_to_host : bool, optional (default=False)
                If True, tied modules (e.g. input embeddings and lm_head) MUST be
                loaded on the local host rather than merely co-located with each
                other on the same worker. Raises AssignmentError immediately if
                host_max_memory_bytes is insufficient, instead of silently falling
                back to worker placement.

        Returns:
            dict: A dictionary with the following keys:
                - success : bool
                    Whether assignment completed successfully.
                - config : dict
                    Mapping of module_path -> assignment spec, where each entry may be:
                        {
                            "type": "loaded" | "offloaded" | "unassigned",
                            "device": "host" (if loaded),
                            "assigned_workers": [worker_id] (if offloaded),
                            "memory": bytes,
                            "module": str,
                            "module_path": str,
                            "training": bool,
                            "optimizer_spec": dict,
                            "batch_size": int,
                            "model_type": str,
                            "parent_module_path": str (optional, for pipelining)
                        }
                - model_memory : int
                    Total estimated memory footprint of the model under the provided
                    parameters (including activations and KV cache).
                - host_memory_used: int
                    Assigned memory to validator
                - error : Optional[str]
                    None on success. On failure, a short description of what went
                    wrong (an AssignmentError message, or "<ExceptionType>: <msg>"
                    for any other unexpected failure), so callers can log/surface
                    *why* assignment failed instead of just a bare success=False.

        Notes:
            - Modules that are too large or loop-iterable are recursively split into
              children until they can be assigned.
            - Sequential layers may later be grouped for pipeline parallelism via
              `_group_sequential_layers`.
            - Worker memory is decremented as assignments occur to prevent overcommit.
            - If assignment fails, `success=False` is returned and config may be partial.
        """
        self._reset_run_state()

        if optimizer_spec is None:
            optimizer_spec = {}

        config = {}
        success = True
        error = None
        model_memory = 0
        model_label = model if isinstance(model, str) else type(model).__name__

        # An empty/missing workers dict can't host anything, and used to fall
        # through silently into "no modules assigned" with zero explanation.
        # Surface it immediately instead.
        if not workers:
            logging.warning(
                "ModelParser.create_distributed_config: no workers were provided "
                "(workers=%r); nothing can be offloaded for %r.",
                workers,
                model_label,
            )

        try:
            if isinstance(model, str):
                self.model_name = model
                self._log("", f"Loading model skeleton for '{model}'...")
                model = load_model_skeleton(self.model_name, model_type)
                self._log("", f"Loaded model skeleton for '{self.model_name}'.")

            workers_state = self._build_workers_state(workers or {})

            self._host_max_module_bytes = host_max_module_bytes or 1e15

            # Fail fast if force_tied_to_host can't possibly be honored
            if force_tied_to_host and host_max_memory_bytes == 0:
                raise ValueError(
                    "force_tied_to_host=True requires host_max_memory_bytes > 0 to "
                    "keep tied modules on the host."
                )

            model_memory, _ = estimate_memory(
                model,
                training=training,
                seq_length=max_seq_len,
                optimizer_type=optimizer_type,
                batch_size=batch_size,
                recursive=True,
                count_activations=True,
                include_kv_cache=True,
            )
            self._log(
                "",
                f"Estimated model memory: {model_memory / 1e6:.2f}MB "
                f"across {len(workers_state)} worker(s) and "
                f"{host_max_memory_bytes / 1e6:.2f}MB host budget.",
            )

            tied_embed_path, tied_lm_head_path = self._detect_tied_weights(model)

            config, _, _ = self._recurse_module(
                module=model,
                root_module=None,
                module_path="model",
                workers_state=workers_state,
                training=training,
                trusted=trusted,
                input_obfuscation=input_obfuscation,
                last_worker=None,
                optimizer_type=optimizer_type,
                optimizer_spec=optimizer_spec,
                host_max_memory_bytes=host_max_memory_bytes,
                host_max_depth=host_max_depth,
                max_offload_depth=max_offload_depth,
                max_seq_len=max_seq_len,
                batch_size=batch_size,
                model_type=model_type,
                tied_embed_path=tied_embed_path,
                tied_lm_head_path=tied_lm_head_path,
                force_tied_to_host=force_tied_to_host,
            )

            config = _group_sequential_layers(config)

            if self.verbose:
                _log_assignment_summary(config, workers_state)

        except AssignmentError as e:
            success = False
            error = str(e)
            logging.warning(
                "ModelParser.create_distributed_config: assignment failed for "
                "%r: %s",
                model_label,
                error,
            )
        except Exception as e:
            # Anything other than AssignmentError (bad worker data, a model
            # that failed to load, an unexpected estimate_memory error, etc.)
            # used to escape this function entirely and get lost wherever the
            # caller's own error handling was (or wasn't). Catch, log with a
            # full traceback, and report it back structurally instead.
            success = False
            error = f"{type(e).__name__}: {e}"
            logging.error(
                "ModelParser.create_distributed_config: unexpected error while "
                "building distribution for %r: %s",
                model_label,
                error,
                exc_info=True,
            )

        return {
            "success": success,
            "config": config,
            "model_memory": model_memory,
            "host_memory_used": self.assigned_memory,
            "error": error,
        }

    # ------------------------------------------------------------------
    # create_distributed_config helpers
    # ------------------------------------------------------------------

    def _reset_run_state(self) -> None:
        """Reset the per-run bookkeeping used to track tied-weight placement."""
        self.assigned_memory = 0
        self._tied_worker = None
        self._tied_on_host = False
        self._tied_reserved = False

    @staticmethod
    def _build_workers_state(workers: dict) -> dict:
        """Snapshot each worker's available GPU memory for this run."""
        return {
            wid: {"gpu_memory": w["gpu_memory"], "original_memory": w["gpu_memory"]}
            for wid, w in workers.items()
        }

    @staticmethod
    def _detect_tied_weights(model: nn.Module):
        """
        Detect tied input-embedding / output-embedding (lm_head) weights so
        they can be forced onto the same device later.

        Returns:
            (tied_embed_path, tied_lm_head_path): either may be None if the
            model has no tied embeddings, or doesn't expose the standard
            HuggingFace embedding accessors.
        """
        tied_embed_path = None
        tied_lm_head_path = None

        has_embedding_accessors = hasattr(model, "get_output_embeddings") and hasattr(
            model, "get_input_embeddings"
        )
        if not has_embedding_accessors:
            return tied_embed_path, tied_lm_head_path

        out = model.get_output_embeddings()
        inp = model.get_input_embeddings()
        if out is None or inp is None or out.weight.data_ptr() != inp.weight.data_ptr():
            return tied_embed_path, tied_lm_head_path

        for name, mod in model.named_modules():
            if mod is inp:
                tied_embed_path = f"model.{name}"
            if mod is out:
                tied_lm_head_path = f"model.{name}"

        return tied_embed_path, tied_lm_head_path

    def _recurse_module(
        self,
        module: nn.Module,
        module_path: str,
        workers_state: dict,
        training: bool,
        trusted: bool,
        input_obfuscation: bool,
        root_module: nn.Module = None,
        last_worker: Optional[str] = None,
        depth: int = 0,
        optimizer_type="adam",
        optimizer_spec=None,
        host_max_memory_bytes: int = 0,
        host_max_depth: int = 1,
        max_offload_depth: int = 3,
        max_seq_len: int = 2048,
        batch_size: int = 1,
        model_type: str = "chat",
        count_activations: bool = True,
        obfuscation_layer_assigned: bool = False,
        tied_embed_path: Optional[str] = None,
        tied_lm_head_path: Optional[str] = None,
        force_tied_to_host: bool = False,
    ):
        """
        Assign a single module (and recursively its children, if needed) to the
        host or a worker. This is the orchestrator for one recursion step; the
        actual decisions live in the `_*` helper methods below so each step of
        the assignment pipeline can be read (and tested) on its own.

        Returns:
            (config, last_successful_worker, obfuscation_layer_assigned)
        """
        config = {}
        indent = "  " * depth

        if root_module is None:
            root_module = module
        if optimizer_spec is None:
            optimizer_spec = {}

        self._log(indent, f"Processing: {module_path}")

        memory = _estimate_module_memory(
            module,
            training,
            max_seq_len,
            optimizer_type,
            batch_size,
            depth,
            count_activations,
        )
        self._log(indent, f"  Memory required: {memory / 1e6:.2f}MB")

        is_tied_module = module_path in (tied_embed_path, tied_lm_head_path)

        force_host, obfuscation_layer_assigned = _resolve_force_host(
            module=module,
            module_path=module_path,
            depth=depth,
            input_obfuscation=input_obfuscation,
            obfuscation_layer_assigned=obfuscation_layer_assigned,
            tied_embed_path=tied_embed_path,
            tied_lm_head_path=tied_lm_head_path,
            host_max_memory_bytes=host_max_memory_bytes,
        )

        host_entry = self._try_host_placement(
            module=module,
            module_path=module_path,
            memory=memory,
            depth=depth,
            training=training,
            optimizer_spec=optimizer_spec,
            batch_size=batch_size,
            model_type=model_type,
            input_obfuscation=input_obfuscation,
            force_host=force_host,
            is_tied_module=is_tied_module,
            tied_embed_path=tied_embed_path,
            tied_lm_head_path=tied_lm_head_path,
            host_max_memory_bytes=host_max_memory_bytes,
            host_max_depth=host_max_depth,
            force_tied_to_host=force_tied_to_host,
            indent=indent,
        )
        if host_entry is not None:
            config[module_path] = host_entry
            return config, None, obfuscation_layer_assigned

        self._require_tied_host_placement(
            is_tied_module, force_tied_to_host, module_path, host_max_depth, indent
        )

        is_loop_iterable = _is_loop_iterable_module(module, module_path)

        assigned_worker = self._resolve_assigned_worker(
            module_path=module_path,
            memory=memory,
            workers_state=workers_state,
            last_worker=last_worker,
            is_tied_module=is_tied_module,
            is_loop_iterable=is_loop_iterable,
            depth=depth,
            indent=indent,
        )

        if assigned_worker:
            config[module_path] = self._place_on_worker(
                module=module,
                module_path=module_path,
                memory=memory,
                training=training,
                optimizer_spec=optimizer_spec,
                batch_size=batch_size,
                model_type=model_type,
                assigned_worker=assigned_worker,
                is_tied_module=is_tied_module,
                tied_embed_path=tied_embed_path,
                tied_lm_head_path=tied_lm_head_path,
                workers_state=workers_state,
                indent=indent,
            )
            return config, assigned_worker, obfuscation_layer_assigned

        if is_tied_module:
            # Didn't fit on host or on any single worker: splitting a tied
            # module across devices would break the tied-weight guarantee.
            self._reject_unplaceable_tied_module(config, module_path, memory, indent)

        self._reject_if_max_depth_exceeded(
            config, module_path, memory, depth, max_offload_depth
        )

        return self._recurse_into_children(
            module=module,
            module_path=module_path,
            root_module=root_module,
            workers_state=workers_state,
            training=training,
            trusted=trusted,
            input_obfuscation=input_obfuscation,
            last_worker=last_worker,
            depth=depth,
            optimizer_type=optimizer_type,
            optimizer_spec=optimizer_spec,
            host_max_memory_bytes=host_max_memory_bytes,
            host_max_depth=host_max_depth,
            max_offload_depth=max_offload_depth,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            model_type=model_type,
            obfuscation_layer_assigned=obfuscation_layer_assigned,
            tied_embed_path=tied_embed_path,
            tied_lm_head_path=tied_lm_head_path,
            force_tied_to_host=force_tied_to_host,
            memory=memory,
            is_loop_iterable=is_loop_iterable,
            config=config,
            indent=indent,
        )

    def _try_host_placement(
        self,
        *,
        module,
        module_path,
        memory,
        depth,
        training,
        optimizer_spec,
        batch_size,
        model_type,
        input_obfuscation,
        force_host,
        is_tied_module,
        tied_embed_path,
        tied_lm_head_path,
        host_max_memory_bytes,
        host_max_depth,
        force_tied_to_host,
        indent,
    ) -> Optional[dict]:
        """
        Attempt to keep `module` on the local host.

        Returns:
            The config entry dict if the module was placed on host, otherwise
            None (meaning the caller should continue on to worker assignment).
        """
        remaining_host_budget = host_max_memory_bytes - self.assigned_memory
        fits_host_budget = (
            host_max_memory_bytes
            and memory <= remaining_host_budget
            and depth <= host_max_depth
            and memory <= self._host_max_module_bytes
        )

        if not (fits_host_budget or force_host):
            return None

        if force_host and memory > remaining_host_budget:
            if is_tied_module and (force_tied_to_host or self._tied_on_host):
                self._log(
                    indent,
                    f"  FAILED: tied module {module_path} cannot be placed on host "
                    f"(requires {memory / 1e6:.2f}MB, "
                    f"{remaining_host_budget / 1e6:.2f}MB available)",
                )
                raise AssignmentError(
                    f"Unable to place tied module {module_path} on host: exceeds "
                    f"remaining host budget ({memory / 1e6:.2f}MB required, "
                    f"{remaining_host_budget / 1e6:.2f}MB available)."
                )

            self._log(
                indent,
                f"  WARNING: Obfuscation layer too large for host "
                f"({memory / 1e6:.2f}MB > {remaining_host_budget / 1e6:.2f}MB available)",
            )
            return None

        prev_assigned = self.assigned_memory
        try:
            self.assigned_memory += memory
            entry = {
                "type": "loaded",
                "device": "host",
                "name": self.model_name,
                "memory": memory,
                "module": _module_type_name(module),
                "module_path": module_path,
                "training": training,
                "optimizer_spec": optimizer_spec,
                "batch_size": batch_size,
                "model_type": model_type,
                "input_boundary": bool(input_obfuscation and depth == 0),
            }

            if module_path == tied_lm_head_path and tied_embed_path:
                entry["tied_to"] = tied_embed_path

            # Remember that the first tied half landed on host, so the
            # counterpart is forced through the host-co-location check above.
            if is_tied_module and not self._tied_on_host and self._tied_worker is None:
                self._tied_on_host = True

            why = "obfuscation boundary" if force_host else "host budget"
            self._log(indent, f"  Kept on host ({why}) - {memory / 1e6:.2f}MB")

            return entry
        except Exception:
            self.assigned_memory = prev_assigned
            raise

    def _require_tied_host_placement(
        self, is_tied_module, force_tied_to_host, module_path, host_max_depth, indent
    ) -> None:
        """
        Raise if `force_tied_to_host` is set and the tied module fell through
        the host-placement branch entirely, instead of silently letting it
        continue on to worker assignment.
        """
        if not (is_tied_module and force_tied_to_host):
            return

        self._log(
            indent,
            f"  FAILED: tied module {module_path} did not qualify for host "
            f"placement (host_max_depth={host_max_depth}, "
            f"host_max_module_bytes={self._host_max_module_bytes / 1e6:.2f}MB) "
            "but force_tied_to_host=True",
        )
        raise AssignmentError(
            f"force_tied_to_host=True but tied module {module_path} does not "
            f"qualify for host placement under the current host_max_depth "
            f"/ host_max_module_bytes settings."
        )

    def _resolve_tied_forced_worker(self, module_path, memory, workers_state, indent):
        """
        If this module is one half of a tied pair and the other half already
        landed on a specific worker, return that worker so both halves are
        co-located. Returns None if this is the first half of the pair.
        """
        if self._tied_worker is None:
            return None

        worker_info = workers_state.get(self._tied_worker)

        if self._tied_reserved:
            # Room for this exact module was already carved out when the first
            # tied half was placed: consume it, don't re-decrement (that would
            # double-charge the same bytes).
            self._tied_reserved = False
            self._log(
                indent,
                f"  Using pre-reserved slot on worker {self._tied_worker} "
                "(co-located with tied counterpart)",
            )
            return self._tied_worker

        if worker_info and worker_info["gpu_memory"] >= memory:
            worker_info["gpu_memory"] -= memory
            self._log(
                indent,
                f"  Forcing tied module onto worker {self._tied_worker} "
                "(co-located with tied counterpart)",
            )
            return self._tied_worker

        # No reservation and no room; falling through would let this land
        # elsewhere (or recurse/split) and silently break the tied-weight
        # guarantee. Fail loudly instead.
        self._log(
            indent,
            f"  FAILED: worker {self._tied_worker} lacks memory "
            f"({memory / 1e6:.2f}MB) to co-locate tied module {module_path}",
        )
        raise AssignmentError(
            f"Unable to co-locate tied module {module_path} with its "
            f"counterpart on worker {self._tied_worker}: insufficient "
            f"reserved/available memory ({memory / 1e6:.2f}MB required)."
        )

    def _resolve_assigned_worker(
        self,
        *,
        module_path,
        memory,
        workers_state,
        last_worker,
        is_tied_module,
        is_loop_iterable,
        depth,
        indent,
    ):
        """Decide which worker (if any) this module should be assigned to."""
        if is_tied_module:
            tied_forced_worker = self._resolve_tied_forced_worker(
                module_path, memory, workers_state, indent
            )
            if tied_forced_worker is not None:
                return tied_forced_worker

        if is_loop_iterable and depth > 0:
            self._log(indent, "  Module is loop-iterable, will recurse into children")
            return None

        return self._try_assign_worker(memory, module_path, workers_state, last_worker)

    def _reserve_tied_counterpart_room(
        self, assigned_worker, memory, workers_state, indent
    ):
        """Carve out room on `assigned_worker` for the tied counterpart so
        nothing assigned afterward can eat into it.
        """
        self._tied_worker = assigned_worker
        self._tied_on_host = False

        worker_info = workers_state.get(assigned_worker)
        if worker_info and worker_info["gpu_memory"] >= memory:
            worker_info["gpu_memory"] -= memory
            self._tied_reserved = True
            self._log(
                indent,
                f"  Reserved {memory / 1e6:.2f}MB on {assigned_worker} for tied counterpart",
            )
        else:
            self._log(
                indent,
                f"  WARNING: could not reserve room on {assigned_worker} for tied "
                "counterpart; it may fail to co-locate later",
            )

    def _place_on_worker(
        self,
        *,
        module,
        module_path,
        memory,
        training,
        optimizer_spec,
        batch_size,
        model_type,
        assigned_worker,
        is_tied_module,
        tied_embed_path,
        tied_lm_head_path,
        workers_state,
        indent,
    ) -> dict:
        """Build the config entry for a module fully offloaded to a worker,
        and update bookkeeping (worker assignments, tied-pair reservations).
        """
        entry = {
            "type": "offloaded",
            "name": self.model_name,
            "assigned_workers": [assigned_worker],
            "memory": memory,
            "module": _module_type_name(module),
            "module_path": module_path,
            "training": training,
            "optimizer_spec": optimizer_spec,
            "batch_size": batch_size,
            "model_type": model_type,
        }

        if module_path == tied_lm_head_path and tied_embed_path:
            entry["tied_to"] = tied_embed_path

        # If this is the first half of a tied pair being freshly assigned
        # (not forced onto an already-known tied worker), remember the
        # worker and reserve room for the counterpart.
        if is_tied_module and self._tied_worker is None:
            self._reserve_tied_counterpart_room(
                assigned_worker, memory, workers_state, indent
            )

        self.assigned_workers[assigned_worker].append(
            {"memory": memory, "module": module, "module_path": module_path}
        )

        self._log(indent, f"  Assigned to {assigned_worker}")

        return entry

    def _reject_unplaceable_tied_module(
        self, config, module_path, memory, indent
    ) -> None:
        """Mark and raise for a tied module that fits neither the host nor any
        single worker (splitting it would break the tied-weight guarantee).
        """
        config[module_path] = {
            "type": "unassigned",
            "required_memory": memory,
            "module_path": module_path,
            "reason": (
                "Tied module could not be placed on host or a single worker; "
                "splitting it would break weight tying."
            ),
        }
        self._log(
            indent,
            f"  FAILED: tied module {module_path} cannot be split across "
            f"workers/host (requires {memory / 1e6:.2f}MB)",
        )
        raise AssignmentError(
            f"Unable to assign tied module {module_path}: it does not fit on "
            f"host or any single worker ({memory / 1e6:.2f}MB required), and "
            f"splitting it would violate the tied-weight guarantee."
        )

    def _reject_if_max_depth_exceeded(
        self, config, module_path, memory, depth, max_offload_depth
    ) -> None:
        """Mark the module unassigned once max recursion depth is hit."""
        if depth < max_offload_depth:
            return

        config[module_path] = {
            "type": "unassigned",
            "required_memory": memory,
            "module_path": module_path,
            "reason": f"Exceeded max recursion depth ({max_offload_depth})",
        }
        if self.verbose:
            raise AssignmentError(
                f"Unable to assign {module_path}: exceeded max depth {max_offload_depth}"
            )

    def _recurse_into_children(
        self,
        *,
        module,
        module_path,
        root_module,
        workers_state,
        training,
        trusted,
        input_obfuscation,
        last_worker,
        depth,
        optimizer_type,
        optimizer_spec,
        host_max_memory_bytes,
        host_max_depth,
        max_offload_depth,
        max_seq_len,
        batch_size,
        model_type,
        obfuscation_layer_assigned,
        tied_embed_path,
        tied_lm_head_path,
        force_tied_to_host,
        memory,
        is_loop_iterable,
        config,
        indent,
    ):
        """
        Split `module` into its children and recursively assign each one,
        then stamp `parent_module_path` on any children that were offloaded.
        """
        reason = "is loop-iterable" if is_loop_iterable else "too large"
        self._log(
            indent,
            f"  Module {module_path} ({memory / 1e6:.2f}MB) {reason}, "
            "recursing into children...",
        )

        children = list(module.named_children())
        if not children:
            config[module_path] = {
                "type": "unassigned",
                "required_memory": memory,
                "module_path": module_path,
            }
            self._log(indent, "  No children to recurse into - FAILED")
            raise AssignmentError(
                f"Unable to assign {module_path}: no children to distribute"
            )

        prev_child_worker = last_worker
        last_successful_worker = last_worker

        for child_name, child_module in children:
            child_path = f"{module_path}.{child_name}"

            try:
                child_config, child_last_worker, obfuscation_layer_assigned = (
                    self._recurse_module(
                        module=child_module,
                        root_module=root_module,
                        module_path=child_path,
                        workers_state=workers_state,
                        training=training,
                        trusted=trusted,
                        last_worker=prev_child_worker,
                        input_obfuscation=input_obfuscation,
                        depth=depth + 1,
                        optimizer_type=optimizer_type,
                        optimizer_spec=optimizer_spec,
                        host_max_memory_bytes=host_max_memory_bytes,
                        host_max_depth=host_max_depth,
                        max_offload_depth=max_offload_depth,
                        max_seq_len=max_seq_len,
                        batch_size=batch_size,
                        count_activations=False,
                        obfuscation_layer_assigned=obfuscation_layer_assigned,
                        tied_embed_path=tied_embed_path,
                        tied_lm_head_path=tied_lm_head_path,
                        force_tied_to_host=force_tied_to_host,
                    )
                )

                config.update(child_config)

                if child_last_worker:
                    prev_child_worker = child_last_worker
                    last_successful_worker = child_last_worker

            except AssignmentError as e:
                self._log(indent, f"  Child {child_path} failed: {e}")
                raise

        # Stamp parent_module_path on any offloaded children we just processed.
        for child_name, _ in children:
            child_path = f"{module_path}.{child_name}"
            child_cfg = config.get(child_path)
            if child_cfg and child_cfg.get("type") == "offloaded":
                child_cfg["parent_module_path"] = module_path

        return config, last_successful_worker, obfuscation_layer_assigned

    def _try_assign_worker(
        self,
        memory: float,
        module_path: str,
        workers_state: dict,
        last_worker: Optional[str],
    ):
        """Try to assign module to a worker, preferring the last used worker."""
        worker_priority = []
        for wid, winfo in workers_state.items():
            if wid == last_worker:
                worker_priority.insert(0, (wid, winfo))
            else:
                worker_priority.append((wid, winfo))

        if len(worker_priority) > 1:
            first_worker = worker_priority[0]
            rest = sorted(
                worker_priority[1:], key=lambda x: x[1]["gpu_memory"], reverse=True
            )
            worker_priority = [first_worker] + rest

        for worker_id, worker_info in worker_priority:
            if worker_info["gpu_memory"] >= memory:
                worker_info["gpu_memory"] -= memory
                return worker_id

        return None

    # ------------------------------------------------------------------
    # Introspection utilities
    # ------------------------------------------------------------------

    def get_module_path_info(self, module_path: str) -> dict:
        """
        Get information about a specific module path.

        Args:
            module_path: The path to query (e.g., "model.layers.0")

        Returns:
            Dictionary with module information
        """
        return self.module_paths.get(module_path, {})

    def list_all_module_paths(self) -> List[str]:
        """
        Get a list of all module paths in the model.

        Returns:
            Sorted list of module paths
        """
        return sorted(self.module_paths.keys())

    def export_module_hierarchy(self, filename: str = "model_hierarchy.txt"):
        """
        Export the complete module hierarchy to a file.

        Args:
            filename: Output filename
        """
        with open(filename, 'w') as f:
            f.write("MODEL HIERARCHY\n")
            f.write("=" * 80 + "\n\n")

            for path in sorted(self.module_paths.keys()):
                info = self.module_paths[path]
                depth = path.count('.')
                indent = "  " * depth

                f.write(f"{indent}{path}\n")
                f.write(f"{indent}  Type: {info['type']}\n")
                f.write(f"{indent}  Params: {info['param_count']:,}\n")
                f.write(f"{indent}  Memory: ~{info['memory_mb']:.1f}MB\n")
                f.write("\n")

        print(f"Module hierarchy exported to {filename}")
