# Tensorlink

Peer-to-peer AI Inference & Distributed Execution with PyTorch

## What is Tensorlink?

Tensorlink is a Python library and decentralized compute platform for running PyTorch and Hugging Face models across peer-to-peer networks. It enables you to run, train, and serve large models securely across distributed hardware without relying on centralized cloud inference providers.

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

## Installation

```bash
pip install tensorlink
```

**Requirements:** Python 3.10+, PyTorch 2.3+, UNIX/MacOS (Windows: use WSL)

## Quick Start

### Distributed Models in Python

Execute Hugging Face models distributed across the network:

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

### Accessing Models via HTTP

Access models via HTTP on the public network, or configure your own hardware for private API access:

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

## API Endpoints

Tensorlink exposes OpenAI-compatible HTTP endpoints for distributed inference:

- `POST /v1/generate` – Simple text generation
- `POST /v1/chat/completions` – OpenAI-compatible chat interface
- `POST /request-model` – Preload models across the network

### Example: Chat Completions

```python
import requests

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

## Learn More

- **Documentation:** https://smartnodes.ca/tensorlink/docs
- **GitHub Repository:** https://github.com/mattjhawken/tensorlink
- **Discord Community:** https://discord.gg/aCW2kTNzJ2
- **Live Demo:** https://smartnodes.ca/tensorlink

## Contributing

We welcome contributions! Report bugs via GitHub Issues, suggest features on Discord, or submit pull requests to improve the project.

## License

Tensorlink is released under the MIT License.