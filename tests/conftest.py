# tests/conftest.py
import logging
import time
import pytest
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tensorlink.nodes import (
    User,
    Validator,
    Worker,
    UserConfig,
    WorkerConfig,
    ValidatorConfig,
)


PRINT_LEVEL = logging.DEBUG
ON_CHAIN = False
LOCAL = True
UPNP = False


@pytest.fixture(scope="function")
def uwv_nodes():
    """
    Create User-Worker-Validator node group for tests.
    Only one node group fixture should be used per test to avoid having 6 processes active.
    """
    user = User(
        config=UserConfig(
            upnp=UPNP,
            on_chain=ON_CHAIN,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
        )
    )

    validator = Validator(
        config=ValidatorConfig(
            upnp=UPNP,
            on_chain=ON_CHAIN,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
            endpoint=False,
            endpoint_ip="127.0.0.1",
            load_previous_state=False,
        )
    )

    worker = Worker(
        config=WorkerConfig(
            upnp=UPNP,
            on_chain=ON_CHAIN,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
            load_previous_state=False,
        )
    )

    yield user, worker, validator

    # Hard cleanup
    user.cleanup()
    worker.cleanup()
    validator.cleanup()
    time.sleep(3)


@pytest.fixture(scope="function")
def wwv_nodes():
    """
    Create Worker-Worker-Validator node group for tests.
    Only one node group fixture should be used per test to avoid having 6 processes active.
    """
    validator = Validator(
        config=ValidatorConfig(
            upnp=UPNP,
            on_chain=ON_CHAIN,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
            endpoint=True,
            endpoint_ip="127.0.0.1",
            load_previous_state=False,
        )
    )

    worker = Worker(
        config=WorkerConfig(
            upnp=UPNP,
            on_chain=ON_CHAIN,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
            load_previous_state=False,
        )
    )

    worker2 = Worker(
        config=WorkerConfig(
            upnp=UPNP,
            on_chain=ON_CHAIN,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
            load_previous_state=False,
            duplicate="1",
        )
    )

    yield worker, worker2, validator

    # Hard cleanup
    worker.cleanup()
    worker2.cleanup()
    validator.cleanup()
    time.sleep(3)


@pytest.fixture(scope="function")
def connected_uwv_nodes(uwv_nodes):
    """
    Fully connected User-Worker-Validator network.
    """
    user, worker, validator = uwv_nodes

    val_key, val_host, val_port = validator.send_request("info", None)

    time.sleep(1)
    worker.connect_node(val_host, val_port, node_id=val_key, timeout=10)
    time.sleep(1)
    user.connect_node(val_host, val_port, node_id=val_key, timeout=10)
    time.sleep(3)

    return user, worker, validator, (val_key, val_host, val_port)


@pytest.fixture(scope="function")
def connected_wwv_nodes(wwv_nodes):
    """
    Fully connected Worker-Worker-Validator network.
    """
    worker, worker2, validator = wwv_nodes

    val_key, val_host, val_port = validator.send_request("info", None)

    time.sleep(1)
    worker.connect_node(val_host, val_port, node_id=val_key, timeout=10)
    time.sleep(1)
    worker2.connect_node(val_host, val_port, node_id=val_key, timeout=10)
    time.sleep(3)

    return worker, worker2, validator, (val_key, val_host, val_port)
