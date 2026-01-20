# Tensorlink

**Peer-to-peer AI Inference & Distributed Execution with PyTorch**

## What is Tensorlink?

Tensorlink is a Python library and decentralized compute platform for running PyTorch and Hugging Face 
models across peer-to-peer networks. It provides a compelling alternative to centralized cloud providers,
allowing you to run, train, and serve large models across distributed hardware.

## Key Features

- **Native PyTorch & REST API Access** – Use models directly in Python or via HTTP endpoints
- **Run Large Models Without Local VRAM** – Execute models that exceed your GPU capacity
- **Remote Access to Your Own Hardware** – Securely host and access models on your devices
- **Plug-and-Play Distributed Execution** – Automatic model sharding across multiple GPUs
- **Training & Inference Support** – Train models with distributed optimizers or run inference
- **Streaming Generation** – Token-by-token streaming for real-time responses
- **Privacy Controls** – Route queries exclusively to your own hardware
- **Earn Rewards for Idle Compute** – Contribute GPU resources and get compensated

> **Note:** Tensorlink is under active development. APIs may evolve.

## Installation

```bash
pip install tensorlink
```

**Requirements:** Python 3.10+, PyTorch 2.3+, UNIX/MacOS (Windows: use WSL)

## Quick Start

### Option 1: Distributed Models in Python

Execute Hugging Face models distributed across the network:

```python
from tensorlink.ml import DistributedModel

# Connect to a model on the network
model = DistributedModel(
    model="Qwen/Qwen2.5-7B-Instruct",
    training=True,
    device="cuda"
)

# Optimizer instantiation method
optimizer = model.create_optimizer(optimizer_type="Adam", lr=0.001)

# Use like any PyTorch model and optimizer
```

### Option 2: HTTP API Access

Access models via OpenAI-compatible HTTP endpoints:

```python
import requests

# Simple generation
response = requests.post(
    "http://smartnodes.ddns.net/tensorlink-api/v1/generate",
    json={
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "message": "Explain quantum computing in one sentence.",
        "max_new_tokens": 50,
    }
)

print(response.json()["generated_text"])
```

### Option 3: Chat Completions (OpenAI-Compatible)

```python
import requests

response = requests.post(
    "http://localhost:64747/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing."}
        ],
        "max_tokens": 128,
        "temperature": 0.7,
        "stream": False
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Streaming Responses

```python
response = requests.post(
    "http://localhost:64747/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Write a story about AI"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line and line.startswith(b"data: "):
        chunk = line.decode()[6:]
        if chunk != "[DONE]":
            import json
            data = json.loads(chunk)
            content = data["choices"][0]["delta"].get("content", "")
            print(content, end="", flush=True)
```

## API Endpoints

- `POST /v1/generate` – Simple text generation
- `POST /v1/chat/completions` – OpenAI-compatible chat interface
- `POST /request-model` – Preload models across the network

## Running Your Own Node

Host models on your own hardware and expose them via API:

1. Download the latest release from [GitHub](https://github.com/mattjhawken/tensorlink/releases)
2. Configure `config.json` for your setup
3. Run: `./run-node.sh`

By default, nodes contribute to the public network and earn rewards. Configure private mode to use only your own devices.

## Documentation & Resources

- **Full Documentation:** [smartnodes.ca/tensorlink/docs](https://smartnodes.ca/tensorlink/docs)
- **Examples & Guides:** [docs/examples](https://github.com/mattjhawken/tensorlink/tree/main/docs/examples)
- **GitHub Repository:** [github.com/mattjhawken/tensorlink](https://github.com/mattjhawken/tensorlink)
- **Discord Community:** [discord.gg/aCW2kTNzJ2](https://discord.gg/aCW2kTNzJ2)
- **Live Demo:** [smartnodes.ca/localhostGPT](https://smartnodes.ca/localhostGPT)
- **Litepaper:** [Technical Overview](https://github.com/mattjhawken/tensorlink/blob/main/docs/LITEPAPER.md)

## Use Cases

- **Researchers:** Run large models without expensive cloud compute
- **Developers:** Build AI applications with distributed inference
- **Organizations:** Deploy private AI infrastructure across your devices
- **GPU Owners:** Monetize idle compute resources
- **Startups:** Scale AI services without infrastructure costs

## Contributing

We welcome contributions! 

- Report bugs via [GitHub Issues](https://github.com/mattjhawken/tensorlink/issues)
- Suggest features on [Discord](https://discord.gg/aCW2kTNzJ2)
- Submit pull requests to improve code or documentation
- Support the project via [Buy Me a Coffee](https://www.buymeacoffee.com/smartnodes)

## License

Tensorlink is released under the [MIT License](https://github.com/mattjhawken/tensorlink/blob/main/LICENSE).
