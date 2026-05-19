# Node Setup & Network Configuration

Deploy workers, validators, and private clusters for distributed ML workflows.

Tensorlink's node system supports public, private, and local deployment. 
Nodes are for when you want to **contribute compute or control your own infrastructure**:

- Contribute GPU resources to the public network and earn token rewards
- Deploy private clusters for sensitive workloads with complete data privacy
- Expose your hardware as a private API endpoint for external applications
- Run custom models that require trusted execution environments

> **Just want to contribute your GPU to the public network?** Skip ahead to the [Worker Quick Start Guide](worker-quickstart.md) — you'll be up and running in minutes.

Public nodes communicate through a peer-to-peer network secured by smart contracts, enabling rewards and payments for GPU resources.

---

## Node Types and Roles

Tensorlink separates computational responsibilities into three distinct node types.

**Workers**

Worker nodes execute the actual model operations on GPU hardware. They handle tensor computations, gradient calculations, and model weight storage, receiving job assignments from validators.

- Execute forward and backward passes for distributed models
- Store and manage model weights across the network
- Compute gradients during distributed training
- Process inference requests from user nodes

**Validators**

Validator nodes coordinate distributed jobs, route API requests, and manage network topology. They determine optimal model sharding strategies and worker assignments, and optionally expose HTTP endpoints for external access.

- Route inference and training requests to appropriate workers
- Determine model sharding strategies based on available workers
- Expose HTTP API endpoints (optional)
- Validate job authenticity and manage network security

**Users**

User nodes request jobs and communicate with workers via `DistributedModel`. They are spawned automatically inside `DistributedModel.__init__` — no manual setup required for client use.

---

## Installing Node Software

To run dedicated worker or validator nodes, download the latest node software from the [GitHub releases page](https://github.com). The binary includes everything needed to participate in the network and requires minimal configuration.

1. Visit GitHub Releases and download `tensorlink-node` for your platform (Linux/macOS)
2. Extract the archive and navigate to the directory
3. Edit `config.json` to configure your node
4. Run `./run-node.sh` to start the node

**System Requirements:**

- UNIX-based OS (Linux, macOS) — Windows via WSL
- Python 3.10+ (required for worker nodes)
- CUDA-compatible GPU (recommended for workers)
- Stable internet connection

---

## Configuration Examples

`config.json` controls all aspects of node behaviour, from network connectivity to security settings.

### Public Worker Node (Earn Rewards)

Contribute your GPU to the public network and earn token rewards for processing inference and training jobs.

```json
{
  "node": {
    "type": "worker",
    "mode": "public",
    "endpoint": false,
    "priority_nodes": []
  },
  "crypto": {
    "address": "0xYourWalletAddress",
    "mining": false,
    "mining_script": "path/to/mining.executable",
    "seed_validators": [
      ["tensorlink.ddns.net", 38752, "58ef79797cd451e19df4a73fbd9871797f9c6a2e"]
    ]
  },
  "ml": {
    "trusted": false,
    "max_vram_gb": 24
  }
}
```

- Set `trusted: false` to only accept verified Hugging Face models
- Add your wallet address to receive rewards
- Optionally enable `mining` to run an idle script between jobs

### Private Validator Node (API Endpoint)

Coordinate a private cluster and expose HTTP endpoints for external application access.

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

- Use `0.0.0.0` to expose the API on all LAN interfaces
- List your worker nodes in `priority_nodes`
- Set `trusted: true` to allow custom model execution

### Private Worker Node

Minimal configuration for workers in a private cluster.

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

- Workers automatically discover validators on private networks
- Set `max_vram_gb` to cap GPU usage per node

---

## Connecting from Python

Once your nodes are running, connect to them from Python using `User` and `UserConfig` to target specific infrastructure.

### Private Cluster

```python
from tensorlink.ml import DistributedModel
from tensorlink.nodes import User, UserConfig

node = User(
    UserConfig(
        priority_nodes=[["192.168.1.100", 38752]]  # Your validator IP/port
    )
)

model = DistributedModel(
    model="Qwen/Qwen2.5-7B-Instruct",
    training=False,
    node=node
)
```

### Local Testing

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

model = DistributedModel(
    model="gpt2",
    training=False,
    node=node
)
```

**`UserConfig` Parameters:**

| Parameter | Description |
|-----------|-------------|
| `priority_nodes` | List of `[IP, port]` pairs for explicit validator connections |
| `upnp` | Enable UPnP for automatic port forwarding (disable for local/private) |
| `local_test` | Force localhost-only connections for testing |

---

## Common Network Architectures

**Single Device (Worker + Validator)**

```
Device 1 (192.168.1.100)
  ├── Validator Node :38752  (API endpoint)
  └── Worker Node   :38752  (24GB VRAM)
```

Simplest setup. One device runs both roles, exposing an API endpoint while executing models locally.

**Private LAN Cluster**

```
Validator (192.168.1.100:38752)
    ├── Worker 1 (192.168.1.100:38752)  24GB VRAM
    ├── Worker 2 (192.168.1.101:38753)  12GB VRAM
    └── Worker 3 (192.168.1.102:38754)  24GB VRAM
```

Dedicated validator coordinates multiple workers. Models are sharded across available VRAM.

**Hybrid Public-Private Network**

```
Private Cluster
  └── Validator (192.168.1.100) → Public Network
        ├── Local Worker 1 (192.168.1.101)
        ├── Local Worker 2 (192.168.1.102)
        └── Public Workers (global network)
```

Validator connects to both private local workers and the public network, allowing workloads to overflow to community resources when local capacity is exceeded.

---

## Security Best Practices

🔒 **Never enable `trusted: true` on public nodes.** This allows arbitrary code execution. Only enable it on infrastructure you fully control. Public workers should always use `trusted: false`.

🛡️ **Firewall private endpoints.** Binding to `0.0.0.0` exposes the API on all network interfaces — use firewall rules to restrict access to trusted IP ranges.

🔐 **Protect your wallet keys.** `config.json` only requires your public wallet address for reward distribution. Never share your private keys.

📊 **Set `max_vram_gb` appropriately.** This prevents workers from accepting jobs that exceed available GPU memory, avoiding OOM errors during peak usage.

---

## Next Steps

- [Worker Quick Start Guide](worker-quickstart.md) — contribute GPU resources in minutes
- Download the latest node software from GitHub releases
- Join the Discord community for help with node setup and configuration