"""
Distributed Model on Private Cluster

This example demonstrates how to run a DistributedModel across a private cluster of personal devices using Tensorlink
and PyTorch. Instead of relying on public nodes, you can form a closed network of machines (laptops, desktops, servers)
and distribute model execution across them.

These same devices can also be exposed through Tensorlink’s HTTP API for remote inference, which is covered further
below. In both cases, each participating machine must run the Tensorlink node binary with an appropriate config.json.

To connect devices in a private cluster, you have two options:
    Worker → Validator: Add the validator’s IP:PORT to each worker’s priority_nodes.
    Validator → Workers: Add all worker IP:PORT pairs to the validator’s priority_nodes.

Once nodes are connected, a Python User node can attach to the cluster and execute models using DistributedModel,
or you can submit model and inference requests to the validator endpoint (if enabled).

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

Next, the user can either access the validator via HTTP and request a model, or they can 
leverage DistributedModel for access.

import requests


# Request model
response = requests.post(
    url=f"192.168.2.x:64747/request-model",
    json={"model": MODEL_NAME, "model_type": "causal", "time": 300},
    timeout=30,
)

chat_response = requests.post(
    f"{SERVER_URL}/v1/chat/completions",
    json={
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_new_tokens": MAX_NEW_TOKENS,
    },
    timeout=300,
)


from tensorlink import DistributedModel


model = DistributedModel(

"""
