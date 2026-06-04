# Examples

## Public Network Chatbot

**[`examples/public_model.py`](examples/public_model.py)**

Runs a multi-turn chatbot on the public Tensorlink network with no node setup required. `DistributedModel` automatically finds available workers and shards the model across them.

- Uses a rolling `deque` to cap context at a fixed number of turns
- Drives generation directly with `model.generate()` and a manually built prompt
- No `User` node needed, the default public validator handles routing

---

### Private Cluster - Python API

**[`examples/private_model.py`](examples/private_model.py)**

Runs the same chatbot pattern against a private cluster of devices you control. Useful when you want full PyTorch access (custom sampling, logit processing, etc.) rather than going through the HTTP API.

**How it differs from the public example:**

- Requires an explicit `User` node connected to your validator
- Uses proper chat-template formatting (`<|im_start|>` / `<|im_end|>`) rather than plain-text turns, important for instruction-tuned models like Qwen3
- Validator and Worker nodes are spun up via `launch_nodes()` / `connect_nodes()` helpers, or you can start them separately with the node binary

**Node config for this setup**: see [Private LAN Cluster](#private-lan-cluster) below.

---

### Private Cluster - HTTP API

**[`examples/private_api.py`](examples/private_api.py)**

Same private cluster, but accessed through the HTTP endpoint instead of directly from Python. No `User` node is needed on the client side, just point your HTTP client at the validator.

- Uses `POST /v1/models/request` to preload the model before inference
- Drives multi-turn conversation by appending assistant replies to the `messages` list and re-posting to `/v1/chat/completions`
- Works from any language or tool that can make HTTP requests

---

## Node Configuration Examples

### Public Compute Provider

Contribute your GPU to the public network and earn rewards.

**`config.json`:**
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
    "seed_validators": [
      ["smartnodes.ddns.net", 38752, "58ef79797cd451e19df4a73fbd9871797f9c6a2995783c7f6fd2406978a2ba2e"]
    ]
  },
  "ml": {
    "trusted": false,
    "max_vram_gb": 24
  }
}
```

- Accepts inference jobs from anyone on the public network
- `trusted: false` means only verified safe models are executed, never set this to `true` on a public node
- Set `mining: true` to run a mining workload when the node has no active jobs
- Monitor earnings for your wallet address on the Smartnodes network dashboard

**Start the node:**
```bash
./run-node.sh
```

---

### Private LAN Cluster

Run models across multiple machines on your local network without touching the public network.

**Validator** (`192.168.1.100`, `config.json`):
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
      ["192.168.1.102", 38753]
    ]
  },
  "ml": {
    "trusted": true
  }
}
```

**Worker 1** (`192.168.1.101`, `config.json`):
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

**Worker 2** (`192.168.1.102`, `config.json`):
```json
{
  "node": {
    "type": "worker",
    "mode": "private"
  },
  "ml": {
    "trusted": true,
    "max_vram_gb": 12
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

Two ways to connect devices: add the validator's IP:PORT to each worker's `priority_nodes`, or add all worker IP:PORT pairs to the validator's `priority_nodes`. Either direction works.

Once running, hit the validator endpoint from any client:

```python
import requests

response = requests.post(
    "http://192.168.1.100:64747/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-14B",
        "messages": [{"role": "user", "content": "Hello from my private cluster!"}],
        "max_tokens": 100
    }
)
```

Or connect a `User` node from Python: see [`examples/private_model.py`](examples/private_model.py).

---

### Local Development

Test Tensorlink entirely on one machine without any network connectivity.

**`config.json`:**
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

```python
from tensorlink.ml import DistributedModel
from tensorlink.nodes import User, UserConfig

node = User(UserConfig(upnp=False, local_test=True, priority_nodes=[["127.0.0.1", 38752]]))
model = DistributedModel(model="gpt2", training=False)
```

Good for offline development, testing custom models, and debugging before deploying to a cluster.

---

## Security Considerations

**Private networks:**
- Use firewall rules to restrict access to the validator endpoint
- Enable `trusted: true` only on nodes you personally control
- May require port forwarding for access outside your LAN

**Public networks:**
- Never set `trusted: true` on public nodes
- Don't send sensitive data through public validators
- Validate model outputs, deterministic output verification is coming soon
