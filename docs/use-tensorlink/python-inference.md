# Running Models in Python

Use `DistributedModel` to run Hugging Face or PyTorch models across Tensorlink workers from Python. It behaves like a standard `torch.nn.Module`, while worker discovery, sharding and tensor routing happen behind the scenes.

This page covers inference. For fine-tuning, see [Training and fine-tuning](python-training.md). For custom models and private clusters, see [Private clusters](private-clusters.md).

---

## Requirements

- Linux or macOS (Windows through WSL)
- Python 3.10+ and PyTorch 2.3+
- No local GPU is needed to run as a client. A modern NVIDIA GPU or Apple Silicon is needed only to [run your own node](worker-quickstart.md). AMD, Intel and multi-GPU setups haven't been thoroughly tested.

## Install

Use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install tensorlink
```

If PyTorch isn't already installed, pip may install a CPU-only build. For GPU workers, install PyTorch first with the right CUDA version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

See the [PyTorch install guide](https://pytorch.org/get-started/locally/) for other configurations. If installation fails, upgrade pip and retry.

---

## Basic usage

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

By default, creating a `DistributedModel` starts a node in the background and connects to the public Tensorlink network. To use your own infrastructure instead, see [Private clusters](private-clusters.md).

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` \| `nn.Module` | required | Hugging Face model name, a PyTorch module, or a path to model weights |
| `training` | `bool` | `False` | Enables gradient synchronization and distributed optimizer creation. See [Training](python-training.md) |
| `device` | `str` | `"cuda"` | Target device for worker execution. Workers without compatible hardware are filtered out automatically |
| `trusted` | `bool` | `False` | Required for custom user-supplied architectures. Workers must also set `trusted: true` |
| `node` | `User` \| `None` | `None` | The user node to connect through. `None` uses the public network |

Passing a path to weights lets you run models that exceed your local VRAM.

---

## Generation

`model.generate()` accepts the standard Hugging Face generation parameters:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    do_sample=True,
    num_return_sequences=1,
)
```

---

## How models are sharded

When a model exceeds one GPU's VRAM, its layers are distributed across several workers with pipeline parallelism. Tensorlink picks the sharding strategy from the model architecture and the resources workers have available. Manual sharding control and data parallelism are planned.

For how requests reach workers, see [Architecture](../architecture.md).

---

## Saving models

Use the standard Hugging Face methods:

```python
model.save_pretrained("./my-model")
model.push_to_hub("username/model-name")
```

---

## Prefer HTTP?

If you only need text generation or embeddings and don't need PyTorch objects, the [Inference API](api.md) is simpler and needs no local install.
