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

## What is Tensorlink?

Tensorlink is a Python library and decentralized compute platform for running PyTorch and Hugging Face models across 
peer-to-peer networks. It enables you to run, train, and serve large models securely across distributed hardware without relying on 
centralized cloud inference providers.

### Key Features
- **Native PyTorch & REST API Access** – Use models directly in Python or via HTTP endpoints  
- **Run Large Models Without Local VRAM** – Execute models that exceed your GPU capacity  
- **Remote Access to Your Own Hardware** – Securely host and access models on your devices via API  
- **Plug-and-Play Distributed Execution** – Automatic model sharding across multiple GPUs  
- **Training & Inference Support** – Train models with distributed optimizers or run inference across the network  
- **Streaming Generation** – Token-by-token streaming for real-time responses   
- **Privacy Controls** – Route queries exclusively to your own hardware for private usage  
- **Earn Rewards for Idle Compute** – Contribute GPU resources to the network and get compensated  

> **Early Access:** Tensorlink is under active development. APIs and internals may evolve.  
> [Join our Discord](https://discord.gg/aCW2kTNzJ2) for updates, support, and roadmap discussions.
> Learn more in the [**Litepaper**](docs/LITEPAPER.md)

## Quick Start

### Option 1: Distributed Models in Python

#### Installation

```bash
pip install tensorlink
```
**Requirements:** Python 3.10+, PyTorch 2.3+, UNIX/MacOS (Windows: use WSL)

#### Basic Usage

Execute Hugging Face models distributed across the network.

```python
from tensorlink.ml import DistributedModel

MODEL_NAME = "Qwen/Qwen3-8B"

# Connect to a pre-trained model on the network
model = DistributedModel(
    model=MODEL_NAME,
    training=True,
    device="cuda"
)
optimizer = model.create_optimizer(lr=0.001)
```
> See [Examples](docs/examples) for streaming generation, distributed training, custom models, 
> and network configurations.

### Option 2: Accessing Models via HTTP

Access models via HTTP on the public network, or configure your own hardware for private API access. 
Tensorlink exposes OpenAI-style endpoints for distributed inference:
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

print(response.json())
```
>Access the public network or configure your own hardware for private API access. See [Examples](docs/examples) for 
>streaming, chat completions, and API reference.

### Option 3: Run a Node

Run Tensorlink nodes to host models, shard workloads across GPUs, and expose them via Python and HTTP APIs.
Nodes can act as workers (run models), validators (route requests + expose API), or both. This allows you to 
build private clusters, public compute providers, or local development environments.

1. Download the latest `tensorlink-node` from [Releases](...)
2. Edit `config.json` to configure your nodes.
3. Run: `./run-node.sh`

> By default, the config is set for running a public worker node. Your GPU will process network workloads and earn 
> rewards via the networking layer ([Smartnodes](https://smartnodes.ca)). See [Examples](docs/examples) for different 
> device and network configurations.

---

## Configuration Reference

Your `config.json` controls networking, rewards, and model execution behavior.

### Node

| Field            | Type                     | Description                                                                  |
|------------------|--------------------------|------------------------------------------------------------------------------|
| `type`           | `worker\|validator`      | Node type                                                            |
| `mode`           | `public\|private\|local` | Network type: public (earn rewards), private (your devices), local (testing) |
| `endpoint`       | `bool`                   | Enables REST API server on this node (validator role)                        |
| `endpoint_url`   | `str`                    | Address the API binds to. Use `0.0.0.0` to expose on LAN                     |
| `endpoint_port`  | `int`                    | Port for the HTTP API (default: `64747`)                                     |
| `priority_nodes` | `[[ip, port]]`           | Bootstrap peers to connect to first (for private clusters)                   |

### Crypto

| Field           | Type   | Description                                              |
|-----------------|--------|----------------------------------------------------------|
| `address`       | `str`  | Wallet address used for identity and rewards             |
| `mining`        | `bool` | Contribute GPU compute to the public network for rewards |
| `mining_script` | `str`  | Path to mining / GPU workload executable                 |

### ML

| Field | Type | Description |
|-------|------|-------------|
| `trusted` | `bool` | Allows execution of custom user-supplied models |
| `max_vram_gb` | `int` | Limits VRAM usage per node to prevent overload |

> For common configuration recipes and examples, see [**Examples: Node Configuration**](docs/EXAMPLES.md#node-configuration-examples)

---

## API Reference

Tensorlink exposes **OpenAI-compatible HTTP endpoints** for distributed inference.

### Endpoints

- `POST /v1/generate` – Simple text generation
- `POST /v1/chat/completions` – OpenAI-compatible chat interface
- `POST /request-model` – Preload models across the network

### Example: `/v1/generate`

```python
import requests

r = requests.post(
    "http://localhost:64747/v1/generate",
    json={
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "message": "Explain quantum computing in one sentence.",
        "max_new_tokens": 64,
        "temperature": 0.7,
        "stream": False,
    }
)

print(r.json()["generated_text"])
```

### Example: `/v1/chat/completions` (OpenAI-compatible)

```python
r = requests.post(
    "http://localhost:64747/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Explain quantum computing."}
        ],
        "max_tokens": 128,
        "stream": True
    },
    stream=True,
)

for line in r.iter_lines():
    if line:
        print(line.decode(), end="")
```

> For complete API documentation, streaming examples, and parameters, see [**Examples: HTTP API**](docs/EXAMPLES.md#http-api-examples)

---

## Learn More

- 📚 **[Documentation](https://smartnodes.ca/tensorlink/docs)** – Full API reference and guides
- 🎯 **[Examples](docs/EXAMPLES.md)** – Comprehensive usage patterns and recipes
- 💬 **[Discord Community](https://discord.gg/aCW2kTNzJ2)** – Get help and connect with developers
- 🎮 **[Live Demo](https://smartnodes.ca/localhostGPT)** – Try localhostGPT powered by Tensorlink
- 📘 **[Litepaper](docs/LITEPAPER.md)** – Technical overview and architecture

## Contributing

We welcome contributions! Here's how to get involved:

- 🐛 **Report bugs** via [GitHub Issues](https://github.com/mattjhawken/tensorlink/issues)
- 💡 **Suggest features** on our [Discord](https://discord.gg/aCW2kTNzJ2)
- 🔧 **Submit PRs** to improve code or documentation
- ☕ **Support the project** via [Buy Me a Coffee](https://www.buymeacoffee.com/smartnodes)

Tensorlink is released under the [MIT License](LICENSE).
