# Run a Worker

Contribute your GPU to the Tensorlink public network and earn rewards. This gets a public worker running as quickly as possible.

For private clusters or custom models, see [Private clusters](private-clusters.md) and [Configuration](configuration.md).

---

## Requirements

- **OS:** Linux (Ubuntu 20.04+ recommended) or macOS. Windows users can use WSL.
- **GPU:** A modern NVIDIA GPU with CUDA support (RTX 30xx or newer), or Apple Silicon (M3 or newer).
- **RAM:** 16 GB minimum, 32 GB+ recommended.
- **Python:** 3.10 or higher.
- **Wallet:** An Ethereum/Base wallet address to receive rewards.
- **Network:** A stable wired connection. Outbound traffic on port `38752` must not be blocked.

---

## Step 1: Install dependencies

If Python or the CUDA drivers aren't installed:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Verify CUDA is available (NVIDIA GPUs)
nvidia-smi
```

If `nvidia-smi` fails, install the drivers from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) before continuing.

## Step 2: Download the node package

Download the latest `tensorlink-node` archive for your platform from the [GitHub releases page](https://github.com/tensorlink-lab/tensorlink/releases). Once downloaded, 
extract the file. If you're using the command line, you can use the following command (make sure to replace `v0.X.X` 
with the actual version number of the file you downloaded):

```bash
tar -xvf tensorlink-node-linux.v0.X.X.tar.gz && cd tensorlink-node
```

## Step 3: Configure your node

Open `config.json` and set your wallet address. The rest can stay as is for a standard public worker:

```json
{
  "node": {
    "type": "worker",
    "mode": "public",
    "endpoint": false,
    "priority_nodes": [],
    "logging": "INFO"
  },
  "crypto": {
    "address": "0xYourWalletAddress",
    "mining": false,
    "mining_script": "path/to/mining.executable",
    "seed_validators": [
      ["tensorlink.ddns.net", 38752, "58ef79797cd451e19df4a73fbd9871797f9c6a2995783c7f6fd2406978a2ba2e"]
    ]
  },
  "ml": {
    "trusted": false,
    "max_vram_gb": 24
  }
}
```

| Field | What to set |
|---|---|
| `crypto.address` | Your public wallet address, where rewards are sent. Never put a private key in this file |
| `ml.max_vram_gb` | Cap on VRAM the worker may use. Match your GPU's capacity |
| `ml.trusted` | Keep `false` on public nodes so only verified Hugging Face models run |
| `crypto.mining` | Set `true` to run a mining script during idle periods (optional) |

Every field is described in the [Configuration reference](configuration.md).

## Step 4: Start the node

```bash
./run-node.sh
```

The first start can take a few minutes while Python dependencies install. Once running, your node connects to the public network, receives job assignments and accrues rewards.

---

## Rewards

Rewards accrue to the wallet address in your config. Enter it in the [Smartnodes dashboard](https://smartnodes.ca/app) to view and claim them.

---

## Troubleshooting

If you hit a problem, open a [GitHub issue](https://github.com/tensorlink-lab/tensorlink/issues) with steps to reproduce, relevant node logs and any error messages.

| Problem | Fix |
|---|---|
| `nvidia-smi` not found | The CUDA drivers aren't installed. Follow the [NVIDIA CUDA installation guide](https://developer.nvidia.com/cuda-downloads) |
| Node fails to connect | Check your internet connection and make sure no firewall blocks outbound traffic on port `38752` |
| Out-of-memory errors | Lower `max_vram_gb` to leave headroom for system processes |
| Anything else | Join the [Discord](https://discord.gg) or open a GitHub issue |
