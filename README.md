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
- [Node Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Learn More](#learn-more)
- [Contributing](#contributing)

## What is Tensorlink?

Tensorlink is a Python library and decentralized compute platform for running PyTorch and Hugging Face models across 
peer-to-peer networks. It lets you run, train, and serve large models securely on distributed hardware without relying 
on centralized cloud inference providers.

With Tensorlink, models can be automatically sharded across multiple GPUs, enabling execution beyond local VRAM limits. 
You can host models on your own devices, expose them through a REST API, stream tokens in real time, and optionally 
route requests only to your own hardware for private usage. Tensorlink supports both distributed training with 
optimizers and low-latency inference across the network.

### Key Features

- **Native PyTorch & REST API Access** — Use models directly in Python or via HTTP endpoints.
- **Run Large Models** — Automatic offloading and model sharding across peers.
- **Plug-and-Play Distributed Execution** — No manual cluster setup required.
- **Streaming Generation** — Token-by-token responses for real-time apps.
- **Privacy Controls** — Route traffic exclusively to your own machines, or leverage hybrid models privacy enhanced model workflows.

> **Early Access:** Tensorlink is under active development. APIs and internals may evolve.  
> [Join our Discord](https://discord.gg/aCW2kTNzJ2) for updates, support, and roadmap discussions.

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
> See [Examples](https://github.com/mattjhawken/tensorlink/blob/main/docs/examples) for streaming generation, distributed training, custom models, 
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
>Access the public network or configure your own hardware for private API access. See [Examples](https://github.com/mattjhawken/tensorlink/blob/main/docs/examples) for 
>streaming, chat completions, and API reference.

### Option 3: Run a Node

Run Tensorlink nodes to host models, shard workloads across GPUs, and expose them via Python and HTTP APIs.
Nodes can act as workers (run models), validators (route requests + expose API), or both. This allows you to 
build private clusters, public compute providers, or local development environments.

1. Download the latest `tensorlink-node` from [Releases](https://github.com/mattjhawken/tensorlink/releases)
2. Edit `config.json` to configure your nodes.
3. Run: `./run-node.sh`

> By default, the config is set for running a public worker node. Your GPU will process network workloads and earn 
> rewards via the networking layer ([Smartnodes](https://smartnodes.ca)). See [Examples](https://github.com/mattjhawken/tensorlink/blob/main/docs/examples) 
> for different device and network configurations.

---

## Configuration Reference

Your `config.json` controls networking, rewards, and model execution behavior. By default, the `config.json` is set for 
running a public worker node. 

### Node

| Field            | Type                   | Description                                                                                             |
|------------------|------------------------|---------------------------------------------------------------------------------------------------------|
| `type`           | `str`                  | Node Type (`worker\|validator\|both`): validator accepts job & api requests, workers run models         |
| `mode`           | `str`                  | Network Type (`public\|private\|local`): public (earn rewards), private (your devices), local (testing) |
| `endpoint`       | `bool`                 | Endpoint Toggle: Enables REST API server on this node (validator role)                                  |
| `endpoint_url`   | `str`                  | Endpoint URL: Address the API binds to. Use `0.0.0.0` to expose on LAN                                  |
| `endpoint_port`  | `int`                  | Endpoint Port: Port for the HTTP API (default: `64747`)                                                 |
| `priority_nodes` | `List[List[str, int]]` | Nodes to Connect: Bootstrap trusted peers to connect to first (e.g.,`[["192.168.2.42", 38751]]`)        |
| `logging`        | `int`                  | Console logging mode (e.g., `DEBUG\|INFO\|WARNING`)                                                     |

### ML

| Field | Type | Description |
|-------|------|-------------|
| `trusted` | `bool` | Allows execution of custom user-supplied models |
| `max_vram_gb` | `int` | Limits VRAM usage per node to prevent overload |

### Crypto

| Field             | Type                        | Description                                              |
|-------------------|-----------------------------|----------------------------------------------------------|
| `address`         | `str`                       | Wallet address used for identity and rewards             |
| `mining`          | `bool`                      | Contribute GPU compute to the public network for rewards |
| `mining_script`   | `str`                       | Path to mining / GPU workload executable                 |
| `seed_validators` | `List[List[str, int, str]]` | Path to mining / GPU workload executable                 |

> For common configuration recipes and examples, see [**Examples: Node Configuration**](https://github.com/mattjhawken/tensorlink/blob/main/docs/examples/EXAMPLES.md#node-configuration-examples)

---

## API Reference

Tensorlink exposes **OpenAI-compatible HTTP endpoints** for distributed inference.

### Endpoints

- `POST /v1/generate` – Simple text generation
- `POST /v1/chat/completions` – OpenAI-compatible chat interface
- `POST /request-model` – Preload models across the network

---

### `/v1/generate`

Simple generation endpoint with flexible output formats.

#### Request Parameters

| Parameter            | Type   | Default    | Description                               |
|----------------------|--------|------------|-------------------------------------------|
| `hf_name`            | string | *required* | Hugging Face model identifier             |
| `message`            | string | *required* | Input text to generate from               |
| `prompt`             | string | `null`     | Alternative to `message`                  |
| `model_type`         | string | `"auto"`   | Model architecture hint                   |
| `max_length`         | int    | `2048`     | Maximum total sequence length             |
| `max_new_tokens`     | int    | `2048`     | Maximum tokens to generate                |
| `temperature`        | float  | `0.7`      | Sampling temperature (0.01-2.0)           |
| `do_sample`          | bool   | `true`     | Enable sampling vs greedy decode          |
| `num_beams`          | int    | `1`        | Beam search width                         |
| `stream`             | bool   | `false`    | Enable streaming responses                |
| `input_format`       | string | `"raw"`    | `"chat"` or `"raw"`                       |
| `output_format`      | string | `"simple"` | `"simple"`, `"openai"`, or `"raw"`        |
| `history`            | array  | `null`     | Chat history for multi-turn conversations |
| `is_chat_completion` | bool   | `false`    | Determines whether to format chat output  |

#### Example: Basic Generation

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

#### Example: Chat Format with History

```python
r = requests.post(
    "http://localhost:64747/v1/generate",
    json={
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "message": "What about entanglement?",
        "input_format": "chat",
        "output_format": "openai",
        "history": [
            {"role": "user", "content": "Explain quantum computing."},
            {"role": "assistant", "content": "Quantum computing uses..."}
        ],
        "max_new_tokens": 128,
    }
)

print(r.json())
```

---

### `/v1/chat/completions`

OpenAI-compatible chat completions endpoint with full streaming support.

#### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | *required* | Hugging Face model identifier |
| `messages` | array | *required* | Array of chat messages |
| `temperature` | float | `0.7` | Sampling temperature (0.01-2.0) |
| `top_p` | float | `1.0` | Nucleus sampling threshold |
| `n` | int | `1` | Number of completions to generate |
| `stream` | bool | `false` | Enable SSE streaming |
| `stop` | string/array | `null` | Stop sequences |
| `max_tokens` | int | `1024` | Maximum tokens to generate |
| `presence_penalty` | float | `0.0` | Presence penalty (-2.0 to 2.0) |
| `frequency_penalty` | float | `0.0` | Frequency penalty (-2.0 to 2.0) |
| `user` | string | `null` | User identifier for tracking |

#### Message Format

```python
{
    "role": "system" | "user" | "assistant",
    "content": "message text"
}
```

#### Example: Non-Streaming

```python
import requests

r = requests.post(
    "http://localhost:64747/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        "max_tokens": 128,
        "temperature": 0.7,
    }
)

response = r.json()
print(response["choices"][0]["message"]["content"])
```

#### Example: Streaming

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
        if line.decode().startswith("data: "):
            data = line.decode()[6:]  # Remove "data: " prefix
            if data != "[DONE]":
                import json
                chunk = json.loads(data)
                if chunk["choices"][0]["delta"].get("content"):
                    print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
```

#### Response Format (Non-Streaming)

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing harnesses quantum mechanics..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 50,
    "total_tokens": 70
  }
}
```

#### Response Format (Streaming)

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Qwen/Qwen2.5-7B-Instruct","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Qwen/Qwen2.5-7B-Instruct","choices":[{"index":0,"delta":{"content":"Quantum"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Qwen/Qwen2.5-7B-Instruct","choices":[{"index":0,"delta":{"content":" computing"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1234567890,"model":"Qwen/Qwen2.5-7B-Instruct","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

---

### `/request-model`

Preload a model across the distributed network before making generation requests.

#### Request Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `hf_name` | string | Hugging Face model identifier |

#### Example

```python
import requests

r = requests.post(
    "http://localhost:64747/request-model",
    json={"hf_name": "Qwen/Qwen3-8B"}
)

print(r.json())
# {"status": "success", "message": "Model loading initiated"}
```

### Notes

Tensorlink is designed to support any Hugging Face model, however errors with certain 
models may appear. Please report any bugs via [Issues](https://github.com/mattjhawken/tensorlink/issues)

- **Temperature**: Values below `0.01` automatically disable sampling to prevent numerical instability
- **Streaming**: Both endpoints support Server-Sent Events (SSE) streaming via `stream: true`
- **Token IDs**: Automatically handles missing pad/eos tokens with safe fallbacks
- **Format Control**: Use `input_format="chat"` and `output_format="openai"` for seamless integration

> For complete examples, error handling, and advanced usage, see [**Examples: HTTP API**](https://github.com/mattjhawken/tensorlink/blob/main/docs/examples/EXAMPLES.md#http-api-examples)

---

## Learn More

- 📚 **[Documentation](https://smartnodes.ca/tensorlink/docs)** – Full API reference and guides
- 🎯 **[Examples](https://github.com/mattjhawken/tensorlink/blob/main/docs/examples/EXAMPLES.md)** – Comprehensive usage patterns and recipes
- 💬 **[Discord Community](https://discord.gg/aCW2kTNzJ2)** – Get help and connect with developers
- 🎮 **[Live Demo](https://smartnodes.ca/tensorlink)** – Try the chatbot demo powered by a model on Tensorlink
- 📘 **[Litepaper](https://github.com/mattjhawken/tensorlink/blob/main/docs/LITEPAPER.md)** – Technical overview and architecture

## Contributing

Read our [contirbution guide.](https://github.com/mattjhawken/tensorlink/blob/main/.github/CONTRIBUTING.md)

Tensorlink is released under the [MIT License](LICENSE).
