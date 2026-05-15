"""
Runs a distributed LLM using tensorlink's OpenAI-compatible HTTP endpoint.
Spawns a local validator (with endpoint enabled) + worker network, requests
the model via API, then chats via /v1/chat/completions.
"""

import requests
import logging
import time
from contextlib import contextmanager
from tensorlink.nodes import Worker, Validator, WorkerConfig, ValidatorConfig

from helpers import chat_loop


# Model config
MODEL_NAME = "Qwen/Qwen3-0.6B"  # HF model name
N_WORKERS = 2  # Number of worker nodes to spawn locally
WORKER_MEMORY_GB = 1.4  # Dedicated RAM (Gb) for each worker

# Inference config
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful assistant."}

# Network config
SERVER_URL = "http://127.0.0.1:64747"
LOCAL = True
UPNP = False
ON_CHAIN = False
PRINT_LEVEL = 5  # uses standard logging notation (e.g. logging.DEBUG), 5 is a custom verbose+ mode.


@contextmanager
def spawn_nodes():
    """
    Spawns a validator and N_WORKERS worker nodes,
    handling graceful shutdown after use.
    """
    _validator = Validator(
        config=ValidatorConfig(
            upnp=UPNP,
            on_chain=ON_CHAIN,
            local_test=LOCAL,
            print_level=PRINT_LEVEL,
            endpoint=True,
            endpoint_url="127.0.0.1",
            load_previous_state=False,
        )
    )
    time.sleep(0.5)

    _workers = []
    for i in range(N_WORKERS):
        _worker = Worker(
            config=WorkerConfig(
                upnp=UPNP,
                on_chain=ON_CHAIN,
                local_test=LOCAL,
                print_level=PRINT_LEVEL,
                load_previous_state=False,
                duplicate=str(i) if i > 0 else "",
                max_memory_gb=WORKER_MEMORY_GB,
            )
        )
        _workers.append(_worker)
        time.sleep(0.5)

    val_key, val_host, val_port = _validator.send_request("info", ())
    for w in _workers:
        w.connect_node(val_host, val_port, node_id=val_key, timeout=10)

    try:
        yield _workers, _validator
    finally:
        for w in _workers:
            w.cleanup()
        _validator.cleanup()


def request_model():
    """Request model to be loaded on the local network via API."""
    response = requests.post(
        url=f"{SERVER_URL}/request-model",
        json={"hf_name": MODEL_NAME, "model_type": "causal", "time": 300},
        timeout=30,
    )
    assert response.status_code == 200
    time.sleep(10)


def generate(history: list) -> str:
    """Use tensorlink chat endpoint for inference."""
    messages = [SYSTEM_PROMPT] + history
    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        timeout=30,
    )
    return response.json()["choices"][-1]["message"]["content"].strip()


if __name__ == "__main__":
    try:
        with spawn_nodes() as (workers, validator):
            request_model()
            # time.sleep(1000)
            chat_loop(generate)

    except KeyboardInterrupt:
        print("\nExiting...")
