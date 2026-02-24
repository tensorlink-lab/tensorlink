import json

from tensorlink.ml.utils import estimate_memory, load_model_skeleton
from tensorlink.ml.injector import find_loop_in_module_hierarchy
from collections import defaultdict
from typing import Union, Optional, Dict, List, Any
import torch.nn as nn
import textwrap
import ast
import inspect
import re


class AssignmentError(Exception):
    """Raised when a module cannot be assigned to any worker."""

    pass


def _create_grouped_entry(parent_path: str, group: list) -> dict:
    """
    Create a single config entry for a group of consecutive layers.
    """
    if len(group) == 1:
        # Single layer, return as-is
        _, path, cfg = group[0]
        return {path: cfg}

    # Multiple layers - create grouped entry
    layer_indices = [idx for idx, _, _ in group]
    paths = [path for _, path, _ in group]
    configs = [cfg for _, _, cfg in group]

    start_idx = min(layer_indices)
    end_idx = max(layer_indices)

    # Use range notation in the key
    grouped_path = f"{parent_path}{start_idx}-{end_idx}"

    # Merge configurations
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

    # Preserve parent_module_path if present
    if "parent_module_path" in configs[0]:
        grouped_config["parent_module_path"] = configs[0]["parent_module_path"]

    return {grouped_path: grouped_config}


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
    # Group paths by their parent and extract layer patterns
    layer_groups = defaultdict(list)

    for path, cfg in config.items():
        if cfg.get("type") != "offloaded":
            continue

        # Match patterns like "model.layers.0", "model.encoder.layer.5", etc.
        match = re.match(r'^(.+\.)(\d+)$', path)
        if match:
            parent_path = match.group(1)  # e.g., "model.layers."
            layer_idx = int(match.group(2))
            layer_groups[parent_path].append((layer_idx, path, cfg))

    # Create new grouped config
    new_config = {}
    processed_paths = set()

    for parent_path, layers in layer_groups.items():
        # Sort by layer index
        layers.sort(key=lambda x: x[0])

        # Group consecutive layers with same worker
        current_group = []
        current_worker = None

        for layer_idx, path, cfg in layers:
            worker = cfg["assigned_workers"][0] if cfg["assigned_workers"] else None

            if worker == current_worker and current_group:
                # Extend current group
                current_group.append((layer_idx, path, cfg))
            else:
                # Save previous group if exists
                if current_group:
                    new_config.update(_create_grouped_entry(parent_path, current_group))
                    processed_paths.update(p for _, p, _ in current_group)

                # Start new group
                current_group = [(layer_idx, path, cfg)]
                current_worker = worker

        # Don't forget the last group
        if current_group:
            new_config.update(_create_grouped_entry(parent_path, current_group))
            processed_paths.update(p for _, p, _ in current_group)

    # Add all non-layer modules that weren't grouped
    for path, cfg in config.items():
        if path not in processed_paths:
            new_config[path] = cfg

    return new_config


def _is_loop_iterable_module(module: nn.Module, module_path: str) -> bool:
    """
    Detect if this module is iterated over in a loop during the forward pass.
    Uses the same loop detection logic as the injector.

    Returns:
        True if the module is iterated in a loop, False otherwise
    """
    try:
        # Try to find a loop in this module or its immediate children
        module_with_loop, loop_node, path = find_loop_in_module_hierarchy(
            module, max_depth=1  # Only check this level
        )
        # If we found a loop at this level, it's loop-iterable
        return True
    except ValueError:
        # No loop found, check if modulelist
        if isinstance(module, nn.ModuleList):
            return True
        return False


def _log_assignment_summary(config: dict, workers_state: dict):
    """
    Log a summary of the final assignment after configuration is complete.
    """
    print("\n" + "=" * 80)
    print("ASSIGNMENT SUMMARY:")
    print("=" * 80)

    # Group by worker
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

    # Print per-worker assignments
    for worker_id in sorted(worker_assignments.keys()):
        assignments = worker_assignments[worker_id]
        total_memory = sum(a["memory"] for a in assignments)

        print(f"\n{worker_id}:")
        print(f"  Total Memory: {total_memory / 1e6:.2f}MB")
        print(f"  Remaining: {workers_state[worker_id]['gpu_memory'] / 1e6:.2f}MB")
        print(f"  Modules ({len(assignments)}):")

        for assignment in assignments:
            print(f"    • {assignment['path']}")
            print(
                f"      [{assignment['module_type']}] - {assignment['memory'] / 1e6:.2f}MB"
            )

    # Print unassigned modules if any
    unassigned = [
        path for path, cfg in config.items() if cfg.get("type") == "unassigned"
    ]
    if unassigned:
        print(f"\n⚠ UNASSIGNED MODULES ({len(unassigned)}):")
        for path in unassigned:
            print(f"  • {path}")

    print("=" * 80 + "\n")


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

    def create_distributed_config(
        self,
        model: Union[nn.Module, str],
        workers: dict,
        training: bool,
        trusted: bool,
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

        Notes:
            - Modules that are too large or loop-iterable are recursively split into
              children until they can be assigned.
            - Sequential layers may later be grouped for pipeline parallelism via
              `_group_sequential_layers`.
            - Worker memory is decremented as assignments occur to prevent overcommit.
            - If assignment fails, `success=False` is returned and config may be partial.
        """
        self.assigned_memory = 0
        if optimizer_spec is None:
            optimizer_spec = {}

        if isinstance(model, str):
            self.model_name = model
            model = load_model_skeleton(self.model_name, model_type)

        workers_state = {
            wid: {"gpu_memory": w["gpu_memory"], "original_memory": w["gpu_memory"]}
            for wid, w in workers.items()
        }

        config = {}
        success = True

        self._host_max_module_bytes = host_max_module_bytes
        if host_max_module_bytes == 0:
            self._host_max_module_bytes = 1e15  # Set to massive number if not specified

        model_memory, breakdown = estimate_memory(
            model,
            training=training,
            seq_length=max_seq_len,
            optimizer_type=optimizer_type,
            batch_size=batch_size,
            recursive=True,
            count_activations=True,
            include_kv_cache=True,
        )

        try:
            # We must detect any tied embedding weights here and force them
            # to host loading if so
            tied_embed_path = None
            tied_lm_head_path = None

            if hasattr(model, "get_output_embeddings") and hasattr(
                model, "get_input_embeddings"
            ):
                out = model.get_output_embeddings()
                inp = model.get_input_embeddings()
                if out is not None and inp is not None:
                    if out.weight.data_ptr() == inp.weight.data_ptr():
                        for name, mod in model.named_modules():
                            if mod is inp:
                                tied_embed_path = f"model.{name}"
                            if mod is out:
                                tied_lm_head_path = f"model.{name}"

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
            )

            config = _group_sequential_layers(config)

            # Log final assignment summary
            if self.verbose:
                _log_assignment_summary(config, workers_state)

        except AssignmentError as e:
            success = False

        return {
            "success": success,
            "config": config,
            "model_memory": model_memory,
            "host_memory_used": self.assigned_memory,
        }

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
    ):
        config = {}

        indent = "  " * depth

        if root_module is None:
            root_module = module

        # Log current module being processed
        if self.verbose:
            print(f"{indent}Processing: {module_path}")

        # Get memory of current module
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

        if self.verbose:
            print(f"{indent}  Memory required: {memory / 1e6:.2f}MB")

        # --- Input obfuscation enforcement ---
        force_host = False
        if input_obfuscation and not obfuscation_layer_assigned:
            # Keep the first substantial layer on host for input obfuscation
            # This ensures raw inputs are transformed before being sent to workers

            # Check if this is a leaf module with parameters (actual layer, not container)
            has_params = any(True for _ in module.parameters(recurse=False))
            has_no_children = len(list(module.children())) == 0

            if has_params and has_no_children:
                # This is a leaf layer with parameters - good candidate for obfuscation layer
                force_host = True
                obfuscation_layer_assigned = True
            elif depth <= 1:
                # At shallow depth, still enforce obfuscation even for containers
                # to ensure we capture embedding layers or initial processing
                force_host = True

            # If we need to force host but have no host memory budget, that's an error
            if force_host and host_max_memory_bytes == 0:
                raise ValueError(
                    f"input_obfuscation=True requires host_max_memory_bytes > 0 to keep "
                    f"the input transformation layer on the host."
                )

        # Force host loading for tied embedding to avoid weight duplication across workers
        force_host = force_host or (
            (tied_embed_path and module_path == tied_embed_path)
            or (tied_lm_head_path and module_path == tied_lm_head_path)
        )


        # Local host module if we have the memory OR input obfuscation is enabled
        if (
            host_max_memory_bytes
            and memory <= host_max_memory_bytes - self.assigned_memory
            and depth <= host_max_depth
            and memory <= self._host_max_module_bytes
        ) or force_host:
            # Double-check we can actually fit this on host if forced
            if force_host and memory > host_max_memory_bytes - self.assigned_memory:
                if self.verbose:
                    print(
                        f"{indent}  WARNING: Obfuscation layer too large for host ({memory / 1e6:.2f}MB > {(host_max_memory_bytes - self.assigned_memory) / 1e6:.2f}MB available)"
                    )
                # Don't force it if it truly won't fit
                force_host = False
            else:
                # Check if this is a tied module
                prev_assigned = self.assigned_memory
                try:
                    self.assigned_memory += memory
                    config[module_path] = {
                        "type": "loaded",
                        "device": "host",
                        "name": self.model_name,
                        "memory": memory,
                        "module": (
                            f"{type(module)}".split(".")[-1].split(">")[0][:-1]
                            if not isinstance(module, str)
                            else module
                        ),
                        "module_path": module_path,
                        "training": training,
                        "optimizer_spec": optimizer_spec,
                        "batch_size": batch_size,
                        "model_type": model_type,
                        "input_boundary": (
                            True if input_obfuscation and depth == 0 else False
                        ),
                    }

                    if module_path == tied_lm_head_path and tied_embed_path:
                        config[module_path]["tied_to"] = tied_embed_path

                    if self.verbose:
                        why = "obfuscation boundary" if force_host else "host budget"
                        print(f"{indent}  Kept on host ({why}) — {memory / 1e6:.2f}MB")

                    return config, None, obfuscation_layer_assigned

                except Exception:
                    self.assigned_memory = prev_assigned
                    raise

        # Check if module is loop-iterable before trying to assign
        is_loop_iterable = _is_loop_iterable_module(module, module_path)

        if is_loop_iterable and depth > 0:
            if self.verbose:
                print(f"{indent}  Module is loop-iterable, will recurse into children")
            # Don't try to assign, skip to recursion
            assigned_worker = None
        else:
            assigned_worker = self._try_assign_worker(
                memory, module_path, workers_state, last_worker
            )

        # full module fits on a worker and the module is not iterated on during the forward pass
        if assigned_worker:
            config[module_path] = {
                "type": "offloaded",
                "name": self.model_name,
                "assigned_workers": [assigned_worker],
                "memory": memory,
                "module": (
                    f"{type(module)}".split(".")[-1].split(">")[0][:-1]
                    if not isinstance(module, str)
                    else module
                ),
                "module_path": module_path,
                "training": training,
                "optimizer_spec": optimizer_spec,
                "batch_size": batch_size,
                "model_type": model_type,
            }

            # Check if this is a tied module
            if module_path == tied_lm_head_path and tied_embed_path:
                config[module_path]["tied_to"] = tied_embed_path

            self.assigned_workers[assigned_worker].append(
                {
                    "memory": memory,
                    "module": module,
                    "module_path": module_path,
                }
            )

            if self.verbose:
                print(f"{indent}  Assigned to {assigned_worker}")

            return config, assigned_worker, obfuscation_layer_assigned

        # Check if we've exceeded max recursion depth
        if depth >= max_offload_depth:
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

        # Module is either too large OR is loop-iterable - recurse into children
        if self.verbose:
            reason = "is loop-iterable" if is_loop_iterable else "too large"
            print(
                f"{indent}  Module {module_path} ({memory / 1e6:.2f}MB) {reason}, recursing into children..."
            )

        children = list(module.named_children())

        if not children:
            config[module_path] = {
                "type": "unassigned",
                "required_memory": memory,
                "module_path": module_path,
            }
            if self.verbose:
                print(f"{indent}  No children to recurse into - FAILED")
            raise AssignmentError(
                f"Unable to assign {module_path}: no children to distribute"
            )

        child_workers = set()
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
                    )
                )

                config.update(child_config)

                if child_last_worker:
                    prev_child_worker = child_last_worker
                    last_successful_worker = child_last_worker
                    child_workers.add(child_last_worker)

            except AssignmentError as e:
                if self.verbose:
                    print(f"{indent}  Child {child_path} failed: {e}")
                raise

        # Add parent_module_path
        # Get the children that were just processed (belong to this parent)
        for child_name, _ in children:
            child_path = f"{module_path}.{child_name}"
            if child_path in config:
                child_cfg = config[child_path]
                if child_cfg.get("type") == "offloaded":
                    child_cfg["parent_module_path"] = module_path

        return config, last_successful_worker, obfuscation_layer_assigned

    def _try_assign_worker(
        self,
        memory: float,
        module_path: str,
        workers_state: dict,
        last_worker: Optional[str],
    ):
        """
        Try to assign module to a worker, preferring last worker
        """
        # Sort workers, using the previous worker first, and then from descending capacity
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

        # Try to assign to a worker
        for worker_id, worker_info in worker_priority:
            if worker_info["gpu_memory"] >= memory:
                worker_info["gpu_memory"] -= memory
                return worker_id

        return None

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


def extract_loop_components(for_node: ast.For, tree: ast.AST) -> Dict[str, str]:
    """Extract code before loop, in loop, and after loop"""
    # This is a simplified version - you'd need more robust extraction
    return {
        'pre_loop_code': '',  # Code before the loop
        'loop_var': ast.unparse(for_node.target),  # Loop variable name
        'loop_body': ast.unparse(for_node.body),  # What happens in loop
        'post_loop_code': '',  # Code after the loop
    }


def resolve_module_from_path(model: nn.Module, path: str):
    """Return (parent_module, child_module, child_name)."""
    parts = path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    child_name = parts[-1]
    child = getattr(parent, child_name)
    return parent, child, child_name


def is_layer_loop(for_node: ast.For, layer_range: List[int]) -> bool:
    """Check if this for loop iterates over the layer range we're offloading"""
    # Look for patterns like:
    # for layer in self.layers:
    # for i, layer in enumerate(self.layers):
    # for i in range(len(self.layers)):

    if isinstance(for_node.iter, ast.Attribute):
        # for layer in self.layers
        return for_node.iter.attr in [
            'layers',
            'layer',
            'blocks',
            'h',
            'encoder',
            'decoder',
        ]

    elif isinstance(for_node.iter, ast.Subscript):
        # for layer in self.layers[start:end]
        # for layer in self.layers[0:12]
        if isinstance(for_node.iter.value, ast.Attribute):
            attr_name = for_node.iter.value.attr
            if attr_name in ['layers', 'layer', 'blocks', 'h', 'encoder', 'decoder']:
                return True
        return False

    elif isinstance(for_node.iter, ast.Call):
        # for i, layer in enumerate(self.layers)
        # for i, layer in enumerate(self.layers[start:end])
        # for i in range(start, end)
        # for i in range(len(self.layers))
        if isinstance(for_node.iter.func, ast.Name):
            func_name = for_node.iter.func.id

            if func_name == 'enumerate':
                # Check what's being enumerated
                if for_node.iter.args:
                    enum_target = for_node.iter.args[0]
                    if isinstance(enum_target, ast.Attribute):
                        return enum_target.attr in ['layers', 'layer', 'blocks', 'h']
                    elif isinstance(enum_target, ast.Subscript):
                        return _is_layer_subscript(enum_target)
                return False

            elif func_name == 'range':
                # Check if range matches our layer range
                return _range_matches_layers(for_node.iter, layer_range)

    return False


def _range_matches_layers(range_call: ast.Call, layer_range: List[int]) -> bool:
    """Check if a range() call matches our layer range"""
    try:
        if not range_call.args:
            return False

        # range(n) - single argument
        if len(range_call.args) == 1:
            end = _eval_node(range_call.args[0])
            # Could be range(len(self.layers)) or range(12)
            return end is not None

        # range(start, end) - two arguments
        elif len(range_call.args) == 2:
            start = _eval_node(range_call.args[0])
            end = _eval_node(range_call.args[1])

            if start is not None and end is not None:
                return start == layer_range[0] and end == layer_range[1]
            return True

        # range(start, end, step) - three arguments
        elif len(range_call.args) == 3:
            start = _eval_node(range_call.args[0])
            end = _eval_node(range_call.args[1])
            step = _eval_node(range_call.args[2])

            # Only accept step=1 for our purposes
            if step is not None and step != 1:
                return False

            if start is not None and end is not None:
                return start == layer_range[0] and end == layer_range[1]
            return True
    except:
        pass

    return False


def _is_layer_subscript(subscript: ast.Subscript) -> bool:
    """Check if a subscript accesses a layer container"""
    if isinstance(subscript.value, ast.Attribute):
        return subscript.value.attr in [
            'layers',
            'layer',
            'blocks',
            'h',
            'encoder',
            'decoder',
        ]
    return False


def _eval_node(node: ast.AST) -> Any:
    """Safely evaluate a constant AST node"""
    try:
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Older Python versions
            return node.n
        elif isinstance(node, ast.Call):
            # Handle len(self.layers) pattern
            if isinstance(node.func, ast.Name) and node.func.id == 'len':
                # Can't evaluate len() without the actual object, return None
                return None
        return None
    except:
        return None


def analyze_forward_loop(forward_method, layer_range: List[int]) -> Dict[str, Any]:
    """
    Analyze the forward method to extract loop structure.
    Returns dict with pre_loop_code, loop_var, post_loop_code.
    """
    try:
        source = inspect.getsource(forward_method)
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        # Find the for loop over layers
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check if this loops over a range or module list
                if is_layer_loop(node, layer_range):
                    return extract_loop_components(node, tree)

        return None

    except Exception as e:
        print(f"Could not analyze forward method: {e}")
        return None
