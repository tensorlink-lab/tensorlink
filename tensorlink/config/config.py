"""
Single source of truth for Tensorlink configuration.

Used by run_node.py, run-node.sh (via `python -m tensorlink.config <section.key>`),
the DistributedValidator process and DistributedModel (including the background user
node it starts when none is passed).
"""

import json
import logging
import os
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger("tensorlink.config")
ENV_FILE = "TENSORLINK_CONFIG"
USER_FILE = Path.home() / ".config" / "tensorlink" / "config.json"


@dataclass
class MLConfig:
    trusted: bool = False  # Trusted exec (e.g. use of pickle)
    hosted_modules: bool = True  # Allow node to load modules
    benchmark: bool = False  # Run benchmark
    max_memory_gb: float = 0  # Max allowable hosted memory use (0 = whole device)
    max_module_gb: float = 0.5  # Max allowable hosted module size
    host_fill: bool = (
        False  # Validator: fill hosted jobs' host segments up to the budget
    )


@dataclass
class NodeConfig:
    type: str = "worker"
    mode: str = "private"
    endpoint: bool = False
    endpoint_url: str = "127.0.0.1"
    endpoint_port: int = 64747
    priority_nodes: List = field(default_factory=list)
    logging: str = "INFO"


@dataclass
class ChainConfig:
    address: str = ""
    mining: bool = False
    mining_script: str = ""
    seed_validators: List = field(default_factory=list)


@dataclass
class UserOptions:  # DistributedModel on a user node
    host_memory_gb: Union[float, str] = (
        0.0  # GB this device may host itself; "auto"; 0 = off
    )
    host_memory_fraction: float = 0.5  # share of device memory used by "auto"
    host_fill: bool = True  # use that memory to host extra chunks
    host_pre_units: int = 0  # privacy floor at the start of the model
    host_post_units: int = 0  # privacy floor at the end of the model
    strategy: str = "auto"  # "auto" | "pipeline" | "legacy"
    dtype: Optional[str] = None  # e.g. "float16"
    device: Optional[str] = None  # e.g. "cuda", "mps", "cpu"


@dataclass
class TensorlinkConfig:
    ml: MLConfig = field(default_factory=MLConfig)
    node: NodeConfig = field(default_factory=NodeConfig)
    chain: ChainConfig = field(default_factory=ChainConfig)
    user: UserOptions = field(default_factory=UserOptions)
    path: Optional[Path] = None
    sources: Dict[str, str] = field(default_factory=dict)


_SECTIONS = {
    "ml": MLConfig,
    "node": NodeConfig,
    "chain": ChainConfig,
    "user": UserOptions,
}


def network_flags(mode: str):
    """Return (local, upnp, on_chain) for a node.mode value."""
    if mode == "local":
        return True, False, False
    if mode == "public":
        return False, True, True
    if mode == "private":
        return False, True, False
    raise ValueError(f"Unknown network mode: {mode}")


def log_level(cfg: "TensorlinkConfig") -> int:
    name = str(cfg.node.logging).upper()
    if name == "VERBOSE":
        return 5
    if not hasattr(logging, name):
        raise ValueError(
            f"Invalid logging level '{name}'. "
            f"Must be one of: CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET"
        )
    return getattr(logging, name)


def _coerce(cls, name: str, value: Any) -> Any:
    f = {f.name: f for f in fields(cls)}[name]
    if name == "host_memory_gb":
        return "auto" if str(value).lower() == "auto" else float(value)
    if isinstance(value, str):
        d = f.default if f.default is not MISSING else None
        if isinstance(d, bool):  # check bool before int
            return value.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(d, int):
            return int(value)
        if isinstance(d, float):
            return float(value)
        if f.default_factory is list:  # env: JSON list
            return json.loads(value)
    return value


def _discover(path, search_dirs) -> Optional[Path]:
    cands = [path] if path else []
    if os.environ.get(ENV_FILE):
        cands.append(os.environ[ENV_FILE])
    cands += [Path(d) / "config.json" for d in search_dirs]
    cands += [Path.cwd() / "config.json", USER_FILE]
    return next((Path(c) for c in cands if c and Path(c).is_file()), None)


def load(
    path=None, search_dirs=(), overrides: Optional[dict] = None
) -> TensorlinkConfig:
    """path=False disables file discovery."""
    cfg = TensorlinkConfig()
    raw = {}
    file = None if path is False else _discover(path, search_dirs)
    if file:
        try:
            raw = json.loads(file.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file}: {e}") from e
        cfg.path = file
    else:
        log.info("No config.json found, using defaults")

    for sec, cls in _SECTIONS.items():
        obj = getattr(cfg, sec)
        known = {f.name for f in fields(cls)}
        data = raw.get(sec, {})
        for k in set(data) - known:
            log.warning("Unknown key %s.%s in %s", sec, k, file)
        for name in known:
            if name in data:
                setattr(obj, name, _coerce(cls, name, data[name]))
                cfg.sources[f"{sec}.{name}"] = str(file)
            env = os.environ.get(f"TENSORLINK_{sec}_{name}".upper())
            if env is not None:
                setattr(obj, name, _coerce(cls, name, env))
                cfg.sources[f"{sec}.{name}"] = "env"
            ov = (overrides or {}).get(sec, {}).get(name)
            if ov is not None:
                setattr(obj, name, _coerce(cls, name, ov))
                cfg.sources[f"{sec}.{name}"] = "argument"
    return cfg


def get_path(cfg: TensorlinkConfig, dotted: str):
    sec, name = dotted.split(".", 1)
    return getattr(getattr(cfg, sec), name)
