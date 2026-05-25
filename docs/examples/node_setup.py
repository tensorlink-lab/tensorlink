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
    BaseNode,
)
from typing import Tuple, List

PRINT_LEVEL = logging.DEBUG
LOCAL = True
UPNP = False
ON_CHAIN = False


def launch_nodes() -> Tuple[User, Validator, Worker]:
    """
    Launches a local Validator, Worker, and User node.
    Good for testing distributed models on a single worker
    via the User with DistributedModel.
    """
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
            endpoint_url="127.0.0.1",
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

    return user, validator, worker


def launch_nodes_no_user() -> Tuple[Validator, Worker, Worker]:
    """
    Launches a local Validato and two Worker nodes.
    Good for testing distributed models across two workers
    via the API (no user node required)."""
    validator = Validator(
        config=ValidatorConfig(
            upnp=UPNP,
            on_chain=ON_CHAIN,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
            endpoint=True,
            endpoint_url="127.0.0.1",
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

    worker2 = Worker(
        config=WorkerConfig(
            upnp=UPNP,
            on_chain=ON_CHAIN,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
            duplicate="1",
        )
    )
    time.sleep(1)

    return validator, worker, worker2


def connect_nodes(
    validator: Validator,
    nodes: List[BaseNode],
    timeout: int = 5,
    delay: float = 1.0,
) -> None:
    """
    Connects a list of nodes to a validator.

    Args:
        validator: The validator node acting as bootstrap.
        nodes: Nodes to connect to the validator.
        timeout: Connection timeout in seconds.
        delay: Delay between connections.
    """

    val_key, val_host, val_port = validator.send_request("info", None)

    for node in nodes:
        node.connect_node(
            val_host,
            val_port,
            node_id=val_key,
            timeout=timeout,
        )
        time.sleep(delay)


if __name__ == "__main__":
    # Launch and connect nodes
    user, validator, worker = launch_nodes()
    connect_nodes(validator, [worker, user])

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
