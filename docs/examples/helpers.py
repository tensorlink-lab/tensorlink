"""
This file just contains some helper functions for the examples.
"""

from collections import deque
import requests
import time
import logging
from typing import Tuple, List
from tensorlink.nodes import (
    User,
    Worker,
    Validator,
    UserConfig,
    WorkerConfig,
    ValidatorConfig,
    BaseNode,
)


PRINT_LEVEL = logging.DEBUG
LOCAL = True
UPNP = False
ON_CHAIN = False
MAX_HISTORY_TURNS = 10


def chat_loop(generate_fn, model=None, tokenizer=None):
    history = deque(maxlen=MAX_HISTORY_TURNS)

    while True:
        text = input("You: ").strip()
        if text.lower() == "exit":
            break

        history.append({"role": "user", "content": text})

        try:
            if model is not None and tokenizer is not None:
                reply = generate_fn(model, tokenizer, list(history))
            else:
                reply = generate_fn(list(history))

            print(f"Assistant: {reply}\n")
            history.append({"role": "assistant", "content": reply})

        except Exception as e:
            print(f"Error: {e}\n")
            history.pop()


def request_model(server_url: str, model_name: str):
    """Request model to be loaded on the local network via API."""
    payload = {
        "hf_name": model_name,
        "model_type": "causal",
        "time": 300,  # seconds
    }

    response = requests.post(
        url=f"{server_url}/request-model",
        json=payload,
        timeout=30,
    )
    assert response.status_code == 200
    time.sleep(10)

    print(response.status_code)
    print(response.json())


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
