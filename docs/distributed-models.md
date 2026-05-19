# Distributed Models & Optimizers

*Core Tensorlink primitives for running and training PyTorch models across distributed GPU infrastructure.*

`DistributedModel` and `DistributedOptimizer` behave like native PyTorch components while transparently executing across 
remote GPU workers. The system handles worker discovery, model sharding, forward/backward pass propagation, and pipeline
parallelism across multiple workers. From your code's perspective, you're working with a standard PyTorch model.

---

## DistributedModel

This is the primary interface for running PyTorch and Hugging Face models on the Tensorlink network. Distributed models 
support Hugging Face model names, custom PyTorch modules, or paths to weight files that exceed local VRAM as model 
input.

By default, a distributed model will spawn a node in the background and automatically connect to the tensorlink public 
network. If you wish to connect to private infrastructure, you will manually have to set up a node (see 
[Node Setup Guide](nodes.md)). 

| Parameter | Type | Default | Description                                                                                             |
|-----------|------|---------|---------------------------------------------------------------------------------------------------------|
| `model` | `str \| nn.Module` | — | Hugging Face model name, a PyTorch module instance, or a path to model weights                          |
| `training` | `bool` | `False` | When `True`, enables gradient synchronization and distributed optimizer creation.                       |
| `device` | `str` | `"cuda"` | Target device for worker execution. Workers without compatible hardware are filtered automatically      |
| `trusted` | `bool` | `False` | Required for custom user-supplied architectures. Workers must also have `trusted: true` in their config |
| `node` | `User \| None` | `None` | Explicit define User node for private cluster connections. When `None`, connects to the public network  |

The following example demonstrates the base case model usage and covers: requesting a model via the public network, 
connecting to assigned worker nodes, executing forward passes across workers with automatic tensor routing.

```python
from tensorlink.ml import DistributedModel
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-14B"

model = DistributedModel(
    model=MODEL_NAME
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
inputs = tokenizer("Explain the theory of relativity.", return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Generation

`model.generate()` supports all standard Hugging Face generation parameters:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    do_sample=True,
    num_return_sequences=1
)
```

### Saving & Checkpointing

Trained models can be saved using standard Hugging Face methods:

```python
# Save locally
model.save_pretrained("./my-fine-tuned-model")

# Push to Hugging Face Hub
model.push_to_hub("username/model-name")
```

### Model Sharding

For models that exceed a single GPU's VRAM, layers are automatically distributed across multiple workers using pipeline 
parallelism. Tensorlink determines the optimal sharding strategy based on model architecture and available worker 
resources. Manual sharding control and data parallelism are planned for future releases.

---

## DistributedOptimizer 

When `training=True`, gradient synchronization is enabled and a `DistributedOptimizer` is available via `model.create_optimizer()`. This coordinates parameter updates across all participating workers and mirrors the interface of standard PyTorch optimizers.

```python
from tensorlink.ml import DistributedModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

model = DistributedModel(model=MODEL_NAME, training=True)

optimizer = model.create_optimizer(
    optimizer_type="adamw",
    lr=1e-4,
    weight_decay=0.01
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_loader = DataLoader(your_dataset, batch_size=8)

for epoch in range(3):
    for batch in train_loader:
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True)

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

model.save_pretrained("./fine-tuned-model")
```

### Parameters (`DistributedModel.create_optimizer` )

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optimizer_type` | `str` | `"adamw"` | Optimizer algorithm: `"adam"`, `"adamw"`, or `"sgd"`. AdamW is recommended for most LLM fine-tuning |
| `lr` | `float` | — | Learning rate |
| `weight_decay` | `float` | — | Weight decay (L2 regularization) |
| `**kwargs` | | | Any additional arguments are passed through to the underlying PyTorch optimizer |

> **Performance tip:** Low-shot fine-tuning (few epochs, small datasets) is significantly more practical than 
> large-scale pre-training over internet connections. For best throughput, use local or LAN-based clusters. Connection 
> quality has a substantial impact on distributed performance.

---

## Private Clusters & Custom Models

For custom PyTorch architectures or sensitive workloads, connect to a private cluster using `User` and `UserConfig`:

```python
from tensorlink.ml import DistributedModel
from tensorlink.nodes import User, UserConfig
from custom_model import CustomModel

node = User(
    UserConfig(
        priority_nodes=[["192.168.2.42", 38752]]
    )
)

model = DistributedModel(
    model=CustomModel(),   # or a path to model weights
    training=False,
    trusted=True,          # required for custom architectures
    node=node
)

import torch
outputs = model(torch.randint(0, 50000, (1, 128)))
```

**Custom model requirements:**

- All worker nodes must have `"trusted": true` in `config.json`
- Models must be serializable (no complex external dependencies)
- The `forward()` method defines the distributed compute graph

Custom models cannot run on the public network for security reasons, this will be addressed in a future release. For 
node topology, configuration examples, and network architecture patterns, see the [Node Setup Guide](nodes.md).
