# tests/conftest.py
import logging
import time
import pytest

from tensorlink.nodes import (
    User,
    Validator,
    Worker,
    UserConfig,
    WorkerConfig,
    ValidatorConfig,
)


PRINT_LEVEL = logging.DEBUG
LOCAL = True
UPNP = False


@pytest.fixture(scope="function")
def nodes():
    """
    Create Tensorlink nodes once per test session.
    """

    user = User(
        config=UserConfig(
            upnp=UPNP,
            off_chain_test=LOCAL,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
        )
    )
    time.sleep(1)

    validator = Validator(
        config=ValidatorConfig(
            upnp=UPNP,
            off_chain_test=LOCAL,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
            endpoint=False,
        )
    )
    time.sleep(1)

    worker = Worker(
        config=WorkerConfig(
            upnp=UPNP, off_chain_test=LOCAL, local_test=LOCAL, print_level=PRINT_LEVEL
        )
    )
    time.sleep(1)

    yield validator, user, worker

    # Hard cleanup
    user.cleanup()
    worker.cleanup()
    validator.cleanup()
    time.sleep(3)


@pytest.fixture(scope="function")
def connected_nodes(nodes):
    """
    Fully connected local Tensorlink test network.
    """

    validator, user, worker = nodes

    val_key, val_host, val_port = validator.send_request("info", None)

    worker.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    time.sleep(1)
    user.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    time.sleep(1)

    return validator, user, worker, (val_key, val_host, val_port)
