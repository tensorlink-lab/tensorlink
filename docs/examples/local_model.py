"""
Runs a distributed LLM using tensorlink's DistributedModel interface.
Spawns a local validator + worker network and runs inference directly
via tensorlink's nn.Module wrapper.
"""

import torch
import logging
import time
from contextlib import contextmanager
from transformers import AutoTokenizer
from tensorlink.nodes import Worker, Validator, WorkerConfig, ValidatorConfig
from tensorlink.ml import DistributedModel

from helpers import chat_loop


MODEL_NAME = "Qwen/Qwen3-1.5B"  # HF model name
N_WORKERS = 1  # Number of worker nodes to spawn locally
WORKER_MEMORY_GB = 6  # Dedicated RAM (Gb) for each worker

# Inference config
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
USE_ENDPOINT = False  # Set True to use HTTP endpoint instead of DistributedModel
SERVER_URL = "http://127.0.0.1:64747"

# Network config
LOCAL = True
UPNP = False
ON_CHAIN = False
PRINT_LEVEL = logging.INFO

SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful assistant."}


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

    workers = []
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
        workers.append(_worker)
        time.sleep(0.5)

    val_key, val_host, val_port = _validator.send_request("info", ())
    for w in workers:
        w.connect_node(val_host, val_port, node_id=val_key, timeout=10)

    try:
        yield workers, _validator
    finally:
        for w in workers:
            w.cleanup()
        _validator.cleanup()


def generate(model, tokenizer, history: list) -> str:
    """Use tensorlink nn.Module wrapper for inference."""
    messages = [SYSTEM_PROMPT] + history
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    try:
        with spawn_nodes() as (workers, validator):
            model = DistributedModel(model=MODEL_NAME, training=False)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            chat_loop(generate, model, tokenizer)

    except KeyboardInterrupt:
        print("\nExiting...")
