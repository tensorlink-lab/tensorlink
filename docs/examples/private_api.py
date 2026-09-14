"""
Distributed Model API on Private Cluster

This example demonstrates how to run a DistributedModel across a private cluster of personal
devices using the Tensorlink API. Instead of relying on public nodes, you can form a closed
network of machines (laptops, desktops, servers) and distribute model execution across them.

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

from helpers import connect_nodes, launch_nodes_no_user, request_model
import requests
import time

SERVER_URL = "http://127.0.0.1:64747"
MODEL_NAME = "Qwen/Qwen3-8B"

MAX_HISTORY_TURNS = 6
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.4

if __name__ == "__main__":
    # Validator and Worker nodes can be spun up on other devices using the node binary
    # or inside a python script. No User node is needed for the API path.
    validator, worker, worker2 = launch_nodes_no_user()
    connect_nodes(validator, [worker, worker2])

    # Request model via API
    payload = {"hf_name": MODEL_NAME}
    response = requests.post(f"{SERVER_URL}/v1/models/request", json=payload)
    assert response.status_code == 200

    # Await model initialization
    while True:
        response = requests.get(
            f"{SERVER_URL}/v1/models/status",
            params={"model": MODEL_NAME},
        ).json()
        if response["status"] == "active":
            break
        time.sleep(1)

    # Chat loop using the OpenAI-compatible endpoint
    history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello world!"},
    ]

    chat_payload = {
        "model": MODEL_NAME,
        "messages": history,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }

    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=chat_payload,
        timeout=120,
    )

    assert response.status_code == 200

    result = response.json()
    assistant_message = result["choices"][0]["message"]
    print("Assistant:", assistant_message["content"])

    # Append the reply and continue the conversation
    history.append(assistant_message)
    history.append({"role": "user", "content": "Tell me something interesting."})

    chat_payload["messages"] = history
    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=chat_payload,
        timeout=120,
    )

    assert response.status_code == 200
    result = response.json()
    print("Assistant:", result["choices"][0]["message"]["content"])
