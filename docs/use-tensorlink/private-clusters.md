# Private Clusters and Custom Models

Use a private cluster when you need to run custom PyTorch architectures, keep data on infrastructure you control, or pick exactly which machines do the work.

This page covers connecting to a cluster from Python. To set up the cluster itself, see [Run a validator](validator-endpoint.md), [Run a worker](worker-quickstart.md) and [Configuration](configuration.md).

---

## Connect to a private cluster

Create a `User` node that points at your validator, then pass it to `DistributedModel`:

```python
from tensorlink.ml import DistributedModel
from tensorlink.nodes import User, UserConfig

node = User(
    UserConfig(
        priority_nodes=[["192.168.1.100", 38752]]  # your validator's IP and port
    )
)

model = DistributedModel(
    model="Qwen/Qwen2.5-7B-Instruct",
    training=False,
    node=node,
)
```

If you omit `node`, `DistributedModel` starts its own user node and connects to the public network.

### `UserConfig` parameters

| Parameter | Description |
|---|---|
| `priority_nodes` | List of `[IP, port]` pairs for explicit validator connections |
| `upnp` | Enable UPnP automatic port forwarding. Turn it off for local and private setups |
| `local_test` | Restrict connections to localhost, for testing |

---

## Local testing

Run everything on one machine without touching the network:

```python
from tensorlink.ml import DistributedModel
from tensorlink.nodes import User, UserConfig

node = User(
    UserConfig(
        upnp=False,
        local_test=True,
        priority_nodes=[["127.0.0.1", 38752]],
    )
)

model = DistributedModel(model="gpt2", training=False, node=node)
```

---

## Custom models

Custom PyTorch architectures run only on private clusters, in trusted mode:

```python
import torch
from tensorlink.ml import DistributedModel
from tensorlink.nodes import User, UserConfig
from custom_model import CustomModel

node = User(UserConfig(priority_nodes=[["192.168.2.42", 38752]]))

model = DistributedModel(
    model=CustomModel(),   # or a path to model weights
    training=False,
    trusted=True,          # required for custom architectures
    node=node,
)

outputs = model(torch.randint(0, 50000, (1, 128)))
```

**Requirements**

- Every worker in the cluster must set `"trusted": true` in its `config.json`.
- The model must be serializable, with no complex external dependencies.
- The model's `forward()` method defines the distributed compute graph.

> **Why not on the public network?** Trusted mode allows arbitrary code execution on workers, so custom models can't run on the public network. Secure public support is planned. **Never enable `trusted: true` on public nodes.** See [Configuration → Security](configuration.md#security).

---

## Using the HTTP API with a private cluster

A private validator can also expose the [Inference API](api.md) to your own apps. See [Run a validator](validator-endpoint.md).
