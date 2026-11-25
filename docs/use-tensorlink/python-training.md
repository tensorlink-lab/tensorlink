# Training and Fine-Tuning

Train or fine-tune models across Tensorlink workers with `DistributedModel` and `DistributedOptimizer`. They mirror the standard PyTorch training loop, while gradients are synchronized across workers for you.

Read [Running models in Python](python-inference.md) first for installation and `DistributedModel` basics.

> **Set expectations.** Over internet or LAN links, low-shot fine-tuning (few epochs, small datasets) is far more practical than large-scale pre-training. For the best throughput, train on a local or LAN cluster. Connection quality has a large effect on performance.

---

## Enable training

Pass `training=True`. This turns on gradient synchronization and makes `model.create_optimizer()` available.

```python
from tensorlink.ml import DistributedModel

MODEL_NAME = "Qwen/Qwen3-14B"

model = DistributedModel(model=MODEL_NAME, training=True)
```

## Create the optimizer

`create_optimizer()` returns a `DistributedOptimizer` that coordinates parameter updates across all participating workers. It works like a standard PyTorch optimizer.

```python
optimizer = model.create_optimizer(
    optimizer_type="adamw",
    lr=1e-4,
    weight_decay=0.01,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `optimizer_type` | `str` | `"adamw"` | `"adam"`, `"adamw"` or `"sgd"`. AdamW is recommended for most LLM fine-tuning |
| `lr` | `float` | none | Learning rate |
| `weight_decay` | `float` | none | Weight decay (L2 regularization) |
| `**kwargs` | | | Passed through to the underlying PyTorch optimizer |

---

## Training loop

```python
from tensorlink.ml import DistributedModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

MODEL_NAME = "Qwen/Qwen3-14B"

model = DistributedModel(model=MODEL_NAME, training=True)
optimizer = model.create_optimizer(optimizer_type="adamw", lr=1e-4, weight_decay=0.01)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_loader = DataLoader(your_dataset, batch_size=8)  # your_dataset: yields dicts with a "text" field

for epoch in range(3):
    for batch in train_loader:
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True)

        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch}, loss: {loss.item():.4f}")

model.save_pretrained("./fine-tuned-model")
```

---

## Saving and sharing

```python
model.save_pretrained("./fine-tuned-model")
model.push_to_hub("username/model-name")
```

---

## Notes

- Training is supported on public-network Hugging Face models. Custom architectures require a [private cluster](private-clusters.md) in trusted mode.
- Data parallelism is planned. Today, layers are split across workers with pipeline parallelism.
