from tensorlink.nodes.user_thread import UserThread
from tensorlink.nodes.validator_thread import ValidatorThread
from tensorlink.nodes.worker_thread import WorkerThread

from multiprocessing import Queue
import pytest
import hashlib


@pytest.fixture(scope="module")
def nodes():
    # Initialize nodes
    worker = WorkerThread(Queue(), Queue(), on_chain=False, local_test=True, upnp=False)
    user = UserThread(Queue(), Queue(), on_chain=False, local_test=True, upnp=False)
    validator = ValidatorThread(
        Queue(),
        Queue(),
        on_chain=False,
        local_test=True,
        upnp=False,
        endpoint=False,
    )

    # Start nodes
    worker.start()
    user.start()
    validator.start()

    # Yield nodes to be used in tests
    yield worker, user, validator

    # Stop nodes after all tests complete
    worker.stop()
    user.stop()
    validator.stop()


def test_node_start(nodes):
    # The nodes are started by the fixture, so you can add assertions or interactions here
    worker, user, validator = nodes

    # Connect worker and user to validator
    worker.connect_node(validator.host, validator.port, validator.rsa_key_hash)
    user.connect_node(validator.host, validator.port, validator.rsa_key_hash)

    assert (
        worker.is_alive()
    )  # Example: you should have a method to check the node state
    assert user.is_alive()
    assert validator.is_alive()


def test_node_storage(nodes):
    """Basic test of send, receive, and core functionality"""
    worker, user, validator = nodes

    key = hashlib.sha256(b"a").hexdigest()
    val = {"test": "test"}

    # Validator state and environment management
    validator.dht.store(key, val)
    val1 = validator.dht.query(key)
    val2 = worker.dht.query(key)

    assert (val1 == val2 and val2 == val, "DHT Storage/Query Error")

    # Sending basic data to each other
    # worker.ping_node()
    pass


# def test_job_request(nodes):
#     """Ensure job request and job-related functionalities work"""
#     worker, user, validator = nodes
#
#     user.request_job()


def test_send_ghost_data():
    """Test the sending of information we cannot handle or were not expecting."""
    pass


def test_node_spam():
    """Test the blocking of nodes that overload a connection."""
    pass
