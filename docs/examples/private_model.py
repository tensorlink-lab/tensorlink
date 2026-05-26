"""
Distributed Model on Private Cluster

This example demonstrates how to run a DistributedModel across a private cluster of personal
devices using Tensorlink and PyTorch. Instead of relying on public nodes, you can form a closed
network of machines (laptops, desktops, servers) and distribute model execution across them.

These same devices can also be exposed through Tensorlink's HTTP API for remote inference,
which is covered in private_cluster_api.py. In both cases, each participating machine must
run the Tensorlink node binary with an appropriate config.json.

To connect devices in a private cluster, you have two options:
    Worker → Validator: Add the validator's IP:PORT to each worker's priority_nodes.
    Validator → Workers: Add all worker IP:PORT pairs to the validator's priority_nodes.

Once nodes are connected, a Python User node can attach to the cluster and execute models
using DistributedModel, or you can submit model and inference requests to the validator
endpoint (if enabled).

Worker 1 (config.json)
Runs both a worker and validator and exposes an HTTP endpoint on the local network:
{
  "config": {
    "node": {
      "type": "both",
      "mode": "private",
      "endpoint": true,
      "endpoint_url": "0.0.0.0",
      "endpoint_port": 64747,
      "logging": "INFO"
    },
    "ml": {
      "trusted": false
    }
  }
}

Worker 2 (config.json)
Connects to the validator by specifying its IP:PORT:
{
  "config": {
    "node": {
      "type": "worker",
      "mode": "private",
      "priority_nodes": [
        ["192.168.2.42", 38751]
      ],
      "logging": "INFO"
    },
    "ml": {
      "trusted": false
    }
  }
}
"""

import torch
from collections import deque
from transformers import AutoTokenizer
from tensorlink.ml import DistributedModel
from node_setup import connect_nodes, launch_nodes

MODEL_NAME = "Qwen/Qwen3-8B"

MAX_HISTORY_TURNS = 6
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.4

if __name__ == "__main__":
    # A User node is required when driving inference directly from Python.
    # Validator and Worker nodes can be spun up on other devices using the node
    # binary or inside a python script.
    user, validator, worker = launch_nodes()
    connect_nodes(validator, [user, worker])

    model = DistributedModel(model=MODEL_NAME, training=False, node=user)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Build a rolling chat history as plain text turns.
    # The deque caps context at MAX_HISTORY_TURNS messages.
    history: deque[str] = deque(maxlen=MAX_HISTORY_TURNS)
    history.append("<|im_start|>system\nYou are a helpful assistant.<|im_end|>")
    history.append("<|im_start|>user\nHello world!<|im_end|>")

    prompt = "\n".join(history) + "\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print("Assistant:", reply)

    # Append assistant reply and send a follow-up turn
    history.append(f"<|im_start|>assistant\n{reply}<|im_end|>")
    history.append("<|im_start|>user\nTell me something interesting.<|im_end|>")

    prompt = "\n".join(history) + "\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print("Assistant:", reply)
