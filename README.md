<p align="center">
  <img src="https://raw.githubusercontent.com/mattjhawken/tensorlink/main/docs/images/logo.png" alt="Logo" width="400" style="max-width:100%; border-radius:12px;">
</p>

<h3 align="center">Peer-to-peer AI Inference & Distributed Execution with PyTorch</h3>

<p align="center">
  <img src="https://img.shields.io/github/v/release/mattjhawken/tensorlink?label=Latest%20Release&color=ff69b4" alt="Latest Release Version" />
  <img src="https://img.shields.io/github/downloads/mattjhawken/tensorlink/total?label=Node%20Downloads&color=e5e52e" alt="Node Downloads"/>
  <img src="https://img.shields.io/github/stars/mattjhawken/tensorlink?style=social" alt="GitHub Repo stars"/>
  <a href="https://discord.gg/aCW2kTNzJ2">
    <img src="https://img.shields.io/badge/Join%20Discord-5865F2?logo=discord&logoColor=white" alt="Join us on Discord"/>
  </a>
  <a href="https://smartnodes.ca/tensorlink/docs" target="_blank">
    <img src="https://img.shields.io/badge/Documentation-1d72b8?logo=readthedocs&logoColor=white" alt="Documentation"/>
  </a>
</p>

## Table of Contents

- [What is Tensorlink?](#what-is-tensorlink)
- [Quick Start](#quick-start)
- [Learn More](#learn-more)
- [Contributing](#contributing)

---

## What is Tensorlink?

Tensorlink is a Python library and decentralized compute platform for running PyTorch and Hugging Face models across
peer-to-peer networks. It enables you to easily distribute and remotely access models across devices, whether pooling
your own hardware or tapping into public peer-to-peer resources.

Whether you want to run LLMs exceeding local memory, deploy private inference infrastructure, build agentic workflows
with on-demand compute, or conduct distributed training, Tensorlink handles the coordination. Hardware owners can also
expose their GPUs as private API endpoints, or contribute resources to the public network and earn rewards.

### Key Features

- **Run Large Models** - Automatic offloading and model sharding across peers
- **Native PyTorch & REST API** - Use models directly in Python or via HTTP endpoints
- **Streaming Generation** - Token-by-token streaming for real-time responses
- **Privacy Controls** - Route queries exclusively to your own hardware
- **Earn Rewards** - Contribute GPU resources to the network and get compensated

> **Early Access:** Tensorlink is under active development. APIs and internals may evolve.
> [Join our Discord](https://discord.gg/aCW2kTNzJ2) for updates, support, and roadmap discussions.

---

## Quick Start

There are three ways to interact with Tensorlink. Choose the path that fits your use case:

- **[Distributed Models in Python](#distributed-models-in-python)** - run PyTorch/Hugging Face models directly from Python
- **[HTTP API](#http-api)** - OpenAI-style REST endpoints for distributed inference
- **[Run a Node](#run-a-node)** - contribute GPU compute or host your own private cluster

### Distributed Models in Python

**Installation**

```bash
pip install tensorlink
```

**Requirements:** Python 3.10+, UNIX/macOS (Windows: use WSL). No GPU required to use the public network.

**Inference**

```python
from tensorlink.ml import DistributedModel
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-14B"

model = DistributedModel(model=MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
inputs = tokenizer("Explain the theory of relativity.", return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Distributed Training**

```python
from tensorlink.ml import DistributedModel

model = DistributedModel(model="Qwen/Qwen3-14B", training=True)
optimizer = model.create_optimizer(optimizer_type="adamw", lr=1e-4, weight_decay=0.01)

# Training loop works like standard PyTorch
outputs = model(**inputs, labels=inputs["input_ids"])
outputs.loss.backward()
optimizer.step()
optimizer.zero_grad()
```

> For private clusters, custom architectures, and full parameter reference, see [**docs/distributed-models.md**](https://github.com/tensorlink-lab/tensorlink/docs/distributed-models.md).

---

### HTTP API

Access models via HTTP - either through the public network or your own private node. The API is OpenAI-compatible and requires no GPU or Python on the client side.

**Simple generation**

```python
import requests

response = requests.post(
    "http://smartnodes.ddns.net/tensorlink-api/v1/generate",
    json={
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "message": "Explain quantum computing in one sentence.",
        "max_new_tokens": 50,
        "stream": False,
    }
)
print(response.json()["generated_text"])
```

**OpenAI-compatible chat**

```python
import requests

response = requests.post(
    "http://localhost:64747/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the benefits of distributed computing?"}
        ],
        "max_tokens": 150,
        "stream": False,
    }
)
print(response.json()["choices"][0]["message"]["content"])
```

> For all endpoints, streaming, the responses API, and model preloading, see [**docs/api.md**](https://github.com/tensorlink-lab/tensorlink/docs/api.md).

---

### Run a Node

Run worker or validator nodes to contribute compute to the public network, host a private cluster, or expose models as API endpoints.

1. Download the latest `tensorlink-node` from [Releases](https://github.com/mattjhawken/tensorlink/releases)
2. Edit `config.json` to configure your node
3. Run `./run-node.sh`

The default config runs a public worker node, where your GPU will process network jobs and earn rewards on the public
network via [Smartnodes](https://smartnodes.ca).

> For configuration reference, private cluster setup, and network architecture patterns, see [**docs/nodes.md**](https://github.com/tensorlink-lab/tensorlink/docs/nodes.md).  
> To contribute your GPU in the fastest way possible, see [**docs/worker-guide.md**](https://github.com/tensorlink-lab/tensorlink/docs/worker-guide.md).

---

## Learn More

| Resource                                                                                      | Description                                                  |
|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| [Getting Started](https://github.com/tensorlink-lab/tensorlink/docs/getting-started.md)       | Installation, requirements, and first steps                  |
| [Distributed Models](https://github.com/tensorlink-lab/tensorlink/docs/distributed-models.md) | `DistributedModel`, `DistributedOptimizer`, private clusters |
| [API Reference](https://github.com/tensorlink-lab/tensorlink/docs/api.md)                     | HTTP endpoints, parameters, and examples                     |
| [Node Setup](https://github.com/tensorlink-lab/tensorlink/docs/nodes.md)                      | Workers, validators, config reference, network topologies    |
| [Worker Quick Start](https://github.com/tensorlink-lab/tensorlink/docs/worker-guide.md)       | Contribute GPU compute in minutes                            |
| [Discord Community](https://discord.gg/aCW2kTNzJ2)                                            | Get help and connect with developers                         |
| [Live Demo](https://tensorlink.io)                                                            | Try a chatbot powered by Tensorlink                          |
| [Litepaper](https://github.com/tensorlink-lab/tensorlink/docs/LITEPAPER.md)                   | Technical overview and architecture                          |

---

## Contributing

Read our [contribution guide](https://github.com/mattjhawken/tensorlink/blob/main/.github/CONTRIBUTING.md).

Tensorlink is released under the [MIT License](LICENSE).