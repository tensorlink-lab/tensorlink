from tensorlink.nodes import (
    User,
    Validator,
    Worker,
    UserConfig,
    WorkerConfig,
    ValidatorConfig,
)

from contextlib import contextmanager
import time
import pytest
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PRINT_LEVEL = 5
ON_CHAIN = False
LOCAL = True
UPNP = False
ENABLE_HOSTED_MODULES = True

DEFAULT_MAX_MEMORY_GB = 0.35
DEFAULT_MAX_MODULE_GB = 0.05

HOST_FILL_ENV = "TENSORLINK_ML_HOST_FILL"


@contextmanager
def _host_fill_env(host_fill):
    """
    Set ml.host_fill for the validator process via the shared Tensorlink config env
    override. host_fill=None leaves the config / environment untouched.
    """
    if host_fill is None:
        yield
        return

    previous = os.environ.get(HOST_FILL_ENV)
    os.environ[HOST_FILL_ENV] = "1" if host_fill else "0"

    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(HOST_FILL_ENV, None)
        else:
            os.environ[HOST_FILL_ENV] = previous


def pytest_addoption(parser):
    parser.addoption(
        "--print-level",
        action="store",
        default=5,
        type=int,
        help="Tensorlink node print level",
    )
    parser.addoption(
        "--max-memory-gb",
        action="store",
        default=DEFAULT_MAX_MEMORY_GB,
        type=float,
        help=f"Per-node memory cap in GB (default {DEFAULT_MAX_MEMORY_GB})",
    )
    parser.addoption(
        "--max-module-gb",
        action="store",
        default=DEFAULT_MAX_MODULE_GB,
        type=float,
        help=f"Validator max module size in GB (default {DEFAULT_MAX_MODULE_GB})",
    )


@pytest.fixture(scope="session")
def print_level(pytestconfig):
    return pytestconfig.getoption("--print-level")


@pytest.fixture(scope="session")
def max_memory_gb(pytestconfig):
    return pytestconfig.getoption("--max-memory-gb")


@pytest.fixture(scope="session")
def max_module_gb(pytestconfig):
    return pytestconfig.getoption("--max-module-gb")


@pytest.fixture(scope="module")
def worker_memory_gb(max_memory_gb):
    """Per-module worker cap. Test modules can override this per param."""
    return max_memory_gb


@pytest.fixture(scope="module")
def validator_memory_gb(max_memory_gb):
    return max_memory_gb


@pytest.fixture(scope="module")
def validator_module_gb(max_module_gb):
    return max_module_gb


@pytest.fixture(scope="module")
def enable_hosting():
    """
    Whether the validator may host modules itself (Validator enable_hosting).
    """
    return ENABLE_HOSTED_MODULES


@pytest.fixture(scope="module")
def host_fill():
    """
    Test modules can override this per param. Only matters when enable_hosting
    is True.
    """
    return None


@pytest.fixture(scope="module")
def uwv_nodes(print_level, max_memory_gb, max_module_gb, enable_hosting, host_fill):
    """
    Create User-Worker-Validator node group for tests.
    Only one node group fixture should be used per test to avoid having 6 processes active.
    """
    with _host_fill_env(host_fill):
        user = User(
            config=UserConfig(
                upnp=UPNP,
                on_chain=ON_CHAIN,
                local_test=LOCAL,
                print_level=print_level,
                max_memory_gb=max_memory_gb,
            ),
            benchmark=False,
        )
        time.sleep(1)
        validator = Validator(
            config=ValidatorConfig(
                upnp=UPNP,
                on_chain=ON_CHAIN,
                local_test=LOCAL,
                print_level=print_level,
                endpoint=False,
                endpoint_url="127.0.0.1",
                load_previous_state=False,
                max_memory_gb=max_memory_gb,
            ),
            enable_hosting=enable_hosting,
            max_module_gb=max_module_gb,
            benchmark=False,
        )
        time.sleep(1)
        worker = Worker(
            config=WorkerConfig(
                upnp=UPNP,
                on_chain=ON_CHAIN,
                local_test=LOCAL,
                print_level=print_level,
                load_previous_state=False,
                max_memory_gb=max_memory_gb,
            ),
            benchmark=False,
        )

        time.sleep(1)

        yield user, worker, validator

        # Hard cleanup
        user.cleanup()
        worker.cleanup()
        validator.cleanup()
        time.sleep(30)


@pytest.fixture(scope="module")
def wwv_nodes(
    print_level,
    max_memory_gb,
    max_module_gb,
    worker_memory_gb,
    validator_memory_gb,
    validator_module_gb,
    enable_hosting,
    host_fill,
):
    """
    Create Worker-Worker-Validator node group for tests.
    Only one node group fixture should be used per test to avoid having 6 processes active.
    """
    with _host_fill_env(host_fill):
        validator = Validator(
            config=ValidatorConfig(
                upnp=UPNP,
                on_chain=ON_CHAIN,
                local_test=LOCAL,
                print_level=print_level,
                endpoint=True,
                endpoint_url="127.0.0.1",
                load_previous_state=False,
                max_memory_gb=validator_memory_gb,
            ),
            enable_hosting=enable_hosting,
            max_module_gb=validator_module_gb,
            benchmark=False,
        )

        worker = Worker(
            config=WorkerConfig(
                upnp=UPNP,
                on_chain=ON_CHAIN,
                local_test=LOCAL,
                print_level=print_level,
                load_previous_state=False,
                max_memory_gb=worker_memory_gb,
            ),
            benchmark=False,
        )

        worker2 = Worker(
            config=WorkerConfig(
                upnp=UPNP,
                on_chain=ON_CHAIN,
                local_test=LOCAL,
                print_level=print_level,
                load_previous_state=False,
                duplicate="1",
                max_memory_gb=worker_memory_gb,
            ),
            benchmark=False,
        )

        time.sleep(1)

        yield worker, worker2, validator

        # Hard cleanup
        worker.cleanup()
        worker2.cleanup()
        validator.cleanup()
        time.sleep(10)


@pytest.fixture(scope="module")
def wv_nodes(
    print_level,
    worker_memory_gb,
    validator_memory_gb,
    validator_module_gb,
    enable_hosting,
    host_fill,
):
    """
    Create Worker-Validator node group for tests where a model is divided between
    a single worker and the validator's hosted modules.
    """
    with _host_fill_env(host_fill):
        validator = Validator(
            config=ValidatorConfig(
                upnp=UPNP,
                on_chain=ON_CHAIN,
                local_test=LOCAL,
                print_level=print_level,
                endpoint=True,
                endpoint_url="127.0.0.1",
                load_previous_state=False,
                max_memory_gb=validator_memory_gb,
            ),
            enable_hosting=enable_hosting,
            max_module_gb=validator_module_gb,
            benchmark=False,
        )

        worker = Worker(
            config=WorkerConfig(
                upnp=UPNP,
                on_chain=ON_CHAIN,
                local_test=LOCAL,
                print_level=print_level,
                load_previous_state=False,
                max_memory_gb=worker_memory_gb,
            ),
            benchmark=False,
        )

        time.sleep(1)

        yield worker, validator

        # Hard cleanup
        worker.cleanup()
        validator.cleanup()
        time.sleep(10)


@pytest.fixture(scope="module")
def connected_uwv_nodes(
    uwv_nodes,
):
    """
    Fully connected User-Worker-Validator network.
    """
    user, worker, validator = uwv_nodes

    time.sleep(3)

    val_key, val_host, val_port = validator.send_request("info", None)

    worker.connect_node(val_host, val_port, node_id=val_key, timeout=10)
    time.sleep(1)
    user.connect_node(val_host, val_port, node_id=val_key, timeout=10)

    time.sleep(1)

    return user, worker, validator, (val_key, val_host, val_port)


@pytest.fixture(scope="module")
def connected_wwv_nodes(wwv_nodes):
    """
    Fully connected Worker-Worker-Validator network.
    """
    worker, worker2, validator = wwv_nodes

    time.sleep(3)

    val_key, val_host, val_port = validator.send_request("info", None)

    worker.connect_node(val_host, val_port, node_id=val_key, timeout=10)
    time.sleep(1)
    worker2.connect_node(val_host, val_port, node_id=val_key, timeout=10)
    time.sleep(3)

    return worker, worker2, validator, (val_key, val_host, val_port)


@pytest.fixture(scope="module")
def connected_wv_nodes(wv_nodes):
    """
    Fully connected Worker-Validator network.
    """
    worker, validator = wv_nodes

    time.sleep(3)

    val_key, val_host, val_port = validator.send_request("info", None)

    worker.connect_node(val_host, val_port, node_id=val_key, timeout=10)
    time.sleep(3)

    return worker, validator, (val_key, val_host, val_port)
