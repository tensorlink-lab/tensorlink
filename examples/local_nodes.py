"""
Minimal Local Tensorlink Network Example

This script launches a local Validator, Worker, and User node,
then connects them into a fully functional local P2P network.

This mirrors the pytest `connected_nodes` fixture exactly.
"""

import time
import logging
from tensorlink.nodes import (
    User,
    Worker,
    Validator,
    UserConfig,
    WorkerConfig,
    ValidatorConfig,
)

PRINT_LEVEL = logging.DEBUG
LOCAL = True
UPNP = False
ON_CHAIN = False


if __name__ == "__main__":
    # --- Launch nodes ---
    user = User(
        config=UserConfig(
            upnp=UPNP,
            on_chain=ON_CHAIN,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
        )
    )
    time.sleep(1)

    validator = Validator(
        config=ValidatorConfig(
            upnp=UPNP,
            on_chain=ON_CHAIN,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
            endpoint=True,
            endpoint_ip="127.0.0.1",
        )
    )
    time.sleep(1)

    worker = Worker(
        config=WorkerConfig(
            upnp=UPNP,
            on_chain=ON_CHAIN,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
        )
    )
    time.sleep(1)

    # --- Connect nodes (local only) ---
    val_key, val_host, val_port = validator.send_request("info", None)

    worker.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    time.sleep(1)
    user.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    time.sleep(1)

    print("Local Tensorlink network is live.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        user.cleanup()
        worker.cleanup()
        validator.cleanup()
