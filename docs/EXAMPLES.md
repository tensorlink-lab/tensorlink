# Examples

Comprehensive examples for using Tensorlink across different interfaces and deployment scenarios.

---

## 📚 Table of Contents

- [Python Examples](#python-examples)
  - [Basic Model Usage](#basic-model-usage)
  - [Distributed Training](#distributed-training)
  - [Private Clusters & Custom Models](#private-clusters-and-custom-models)
- [HTTP API Examples](#http-api-examples)
  - [Simple Generation](#simple-generation)
  - [Streaming Response](#streaming-response)
  - [Chat Completions (OpenAI-Compatible)](#chat-completions-openai-compatible)
  - [Model Requesting](#model-preloading)
- [Node Configuration Examples](#node-configuration-examples)
  - [Public Compute Provider](#public-compute-provider)
  - [Private LAN Cluster](#private-lan-cluster)
  - [Local Development](#local-development)

---

## Python Examples
 
### Basic Model Usage

Run a distributed model on the public network with minimal setup:

```python
from tensorlink.ml import DistributedModel
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Initialize distributed model
model = DistributedModel(
    model=MODEL_NAME,
    training=False
)

# Tokenize input
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
inputs = tokenizer("Explain the theory of relativity.", return_tensors="pt")

# Generate response
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

**What's happening:**
- Tensorlink finds available workers with capacity for this model
- Automatically shards the model across GPUs if needed
- Executes forward passes across the network
- Returns outputs as standard PyTorch tensors

---

### Distributed Training

Train models across multiple GPUs on the network:

```python
from tensorlink.ml import DistributedModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

MODEL_NAME = "gpt2"

# Initialize for training
model = DistributedModel(
    model=MODEL_NAME,
    training=True,
    device="cuda"
)

model.train()

# Setup distributed optimizer
optimizer = model.create_optimizer(
    lr=0.001, 
    weight_decay=0.01,
    optimizer_type="adamw"
)

# Training loop
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_loader = DataLoader(your_dataset, batch_size=8)

for epoch in range(3):
    for batch in train_loader:
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True)
        
        # Forward pass (distributed)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        # Backward pass (gradients aggregated across workers)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Loss: {loss.item():.4f}")

# Save trained model
model.save_pretrained("./fine-tuned-model")
```

**What's happening:**
- Model requested on initialization and sent to worker(s)
- Gradients automatically synchronized
- Works with any PyTorch optimizer
- Checkpointing supported

---

### Private Clusters and Custom Models

Run your own models on personal hardware. Currently custom models can only be run on "trusted"
mode which excludes public jobs:

```python
from custom_model import CustomModel
import torch

from tensorlink.ml import DistributedModel
from tensorlink.nodes import User, UserConfig

# Must explicitly create a node to connect to validator node
node = User(
  UserConfig(
    priority_nodes=[["192.168.2.42", 38751]]
  )
)

# Upload and distribute your model
model = DistributedModel(
    model=CustomModel(),  # Can also specify a path to model weights
    training=False,
    trusted=True,
    node=node
)

# Use like any PyTorch model
input_ids = torch.randint(0, 50000, (1, 128))
outputs = model(input_ids)
```

**Requirements:**
- Validator & Worker nodes must be running with `"trusted": true` in config
- Model must be serializable
- Forward method defines compute graph

---

## HTTP API Examples

### Simple Generation

Basic text generation via HTTP:

```python
import requests

response = requests.post(
    "http://localhost:64747/v1/generate",
    json={
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "message": "Explain quantum computing in one sentence.",
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": False,
    }
)

result = response.json()
print(result["generated_text"])
```

**Response Format:**
```json
{
  "generated_text": "Quantum computing uses quantum bits...",
  "tokens_generated": 42,
  "execution_time_ms": 1847
}
```

---

### Streaming Response

Stream tokens incrementally for real-time output:

```python
import requests

response = requests.post(
    "http://localhost:64747/v1/generate",
    json={
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "message": "Write a haiku about distributed computing.",
        "max_new_tokens": 100,
        "stream": True,
    },
    stream=True,
)

print("Response: ", end="", flush=True)
for line in response.iter_lines():
    if line:
        decoded = line.decode()
        if decoded.strip() == "data: [DONE]":
            break
        if decoded.startswith("data: "):
            token = decoded[6:]  # Remove "data: " prefix
            print(token, end="", flush=True)
print()
```

**Stream Format (SSE):**
```
data: Bits
data:  and
data:  qubits
data:  dance
data: ,
data: [DONE]
```

---

### Chat Completions (OpenAI-Compatible)

Use the OpenAI-compatible endpoint for easy integration:

```python
import requests

response = requests.post(
    "http://localhost:64747/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What are the benefits of distributed computing?"}
        ],
        "max_tokens": 150,
        "temperature": 0.8,
        "stream": False
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

**OpenAI Format Response:**
```json
{
  "id": "chatcmpl-xyz123",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Distributed computing offers several advantages..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 94,
    "total_tokens": 122
  }
}
```

**Streaming Chat:**
```python
response = requests.post(
    "http://localhost:64747/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [
            {"role": "user", "content": "Tell me a joke"}
        ],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        decoded = line.decode()
        if "data: [DONE]" in decoded:
            break
        # Process SSE chunk
        print(decoded)
```

---

### Model Preloading

Preload models for faster first-request latency:

```python
import requests

# Request model be loaded across the network
requests.post(
    "http://localhost:64747/request-model",
    json={
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_type": "causal"
    }
)

# Model is now sharded and ready
# Subsequent requests will be faster
response = requests.post(
    "http://localhost:64747/v1/generate",
    json={
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "message": "Hello!",
        "max_new_tokens": 50
    }
)
```

**Use cases:**
- Warm up models before traffic
- Prepare multiple models in parallel
- Reduce cold-start latency

---

## Node Configuration Examples

### Public Compute Provider

**Scenario:** Contribute your GPU to the public network and earn rewards.

**config.json:**
```json
{
  "node": {
    "type": "worker",
    "mode": "public",
    "endpoint": false,
    "priority_nodes": []
  },
  "crypto": {
    "address": "0x1Bc3a15dfFa205AA24F6386D959334ac1BF27336",
    "mining": false,
    "mining_script": "path/to/mining.executable",
    "seed_validators": [
      ["smartnodes.ddns.net", 38751, "58ef79797cd451e19df4a73fbd9871797f9c6a2995783c7f6fd2406978a2ba2e"]
    ]
  },
  "ml": {
    "trusted": false,
    "max_vram_gb": 24
  }
}
```

**What happens:**
- Node connects to Tensorlink's public network
- Accepts inference jobs from anyone
- Earns rewards based on compute contributed
- Only runs verified/safe models (`trusted: false`)
- Mining can be set to true if you have a GPU-heavy workload to run when your device has no active jobs.

**Start node:**
```bash
./run-node.sh
```

**Monitor earnings:**
Check your wallet address on the Smartnodes network dashboard.

---

### Private LAN Cluster

**Scenario:** You have 3 machines on your local network and want to run models privately.

**Worker Node 1** (`config.json` on `192.168.1.101`):
```json
{
  "node": {
    "type": "worker",
    "mode": "private"
  },
  "ml": {
    "trusted": true,
    "max_vram_gb": 24
  }
}
```

**Worker Node 2** (`config.json` on `192.168.1.102`):
```json
{
  "node": {
    "type": "worker",
    "mode": "private"
  },
  "ml": {
    "trusted": true,
    "max_vram_gb": 24
  }
}
```

**Validator Node** (`config.json` on `192.168.1.100`):
```json
{
  "node": {
    "type": "validator",
    "mode": "private",
    "endpoint": true,
    "endpoint_url": "0.0.0.0",
    "endpoint_port": 64747,
    "priority_nodes": [
      ["192.168.1.101", 38752],
      ["192.168.2.102", 38753]
    ]
  },
  "ml": {
    "trusted": true
  }
}
```

**Architecture:**
```
Client → http://192.168.1.100:64747
              ↓
        Validator (100)
           ↙        ↘
    Worker (101)  Worker (102)
    24GB VRAM     12GB VRAM
```

**API Usage:**
```python
import requests

# All traffic goes through validator
response = requests.post(
    "http://192.168.1.100:64747/v1/generate",
    json={
        "hf_name": "Qwen/Qwen3-14B",
        "message": "Hello from my private cluster!",
        "max_new_tokens": 100
    }
)
```

or see [Private Clusters & Custom Models](#private-clusters-and-custom-models) for connecting
within Python.

---

### Local Development

**Scenario:** Test Tensorlink locally without any network connectivity.

**config.json:**
```json
{
  "node": {
    "type": "worker",
    "mode": "local",
    "endpoint": true,
    "endpoint_url": "127.0.0.1",
    "endpoint_port": 64747,
    "priority_nodes": []
  },
  "ml": {
    "trusted": true,
    "max_vram_gb": 24
  }
}
```

**Use case:**
- Offline development
- Testing custom models
- Debugging before deploying to cluster

**Example:**
```python
from tensorlink.ml import DistributedModel
from tensorlink.nodes import User, UserConfig

node = User(
  UserConfig(
    upnp=False,
    local_test=True,
    priority_nodes=[["127.0.0.1", 38752]]
  )
)
# Runs entirely on local node
model = DistributedModel(
    model="gpt2",
    training=False
)
```

---

## Security Considerations

**Private networks:**
- Use firewall rules to restrict access to validator endpoints
- Enable `trusted: true` only on nodes you control
- Consider VPN for remote access to private clusters

**Public networks:**
- Never set `trusted: true` on public nodes
- Don't send sensitive data through public validators
- Validate model outputs (public nodes are untrusted)

## Need Help?

- 💬 **[Join our Discord](https://discord.gg/aCW2kTNzJ2)** for community support
- 📚 **[Read the docs](https://smartnodes.ca/tensorlink/docs)** for API reference
- 🐛 **[Report issues](https://github.com/mattjhawken/tensorlink/issues)** on GitHub
- 📧 **Email:** support@smartnodes.ca