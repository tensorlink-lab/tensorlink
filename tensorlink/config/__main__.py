"""`python -m tensorlink.config node.type` prints one config value (used by run-node.sh)."""

import sys

from .config import get_path, load

try:
    print(str(get_path(load(), sys.argv[1])).strip().lower())
except Exception as e:
    print(
        f"Warning: could not read {sys.argv[1] if len(sys.argv) > 1 else '?'} ({e})",
        file=sys.stderr,
    )
    sys.exit(1)
