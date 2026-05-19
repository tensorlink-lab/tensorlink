# Worker Quick Start Guide

Contribute your GPU to the Tensorlink public network and start earning rewards.

This guide gets a public worker node running as quickly as possible. If you need more control for private clusters, 
custom models, or connecting from Python, see the full [Node Setup Guide](nodes.md).

---

## Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS. Windows users can use WSL.
- **GPU**: Modern NVIDIA GPU with CUDA support (RTX 30xx, M3 Apple Silicon, or Better) 
- **RAM**: 16 GB minimum (32 GB+ recommended)
- **Python**: 3.10 or higher
- **Wallet**: An Ethereum/Base wallet address to receive rewards

---

## Step 1: Install Dependencies

If Python or CUDA aren't already installed:

```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Verify CUDA is available (NVIDIA GPU required)
nvidia-smi
```

If `nvidia-smi` fails, install the CUDA drivers from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) before continuing.

---

## Step 2: Download the Node Software

Download and extract the latest `tensorlink-node` binary from the [GitHub releases page](https://github.com):

```bash
tar -xvf tensorlink-node-linux.v0.X.X.tar.gz && cd tensorlink-node
```

---

## Step 3: Configure Your Node

Open `config.json` and set your wallet address. Everything else can stay as-is for a standard public worker:

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
      ["tensorlink.ddns.net", 38751, "58ef79797cd451e19df4a73fbd9871797f9c6a2995783c7f6fd2406978a2ba2e"]
    ]
  },
  "ml": {
    "trusted": false,
    "max_vram_gb": 24
  }
}
```

**Key fields:**

| Field | Description                                                            |
|-------|------------------------------------------------------------------------|
| `address` | Your public Ethereum/Base wallet address where rewards are sent        |
| `max_vram_gb` | Cap how much VRAM the worker can use. Set to match your GPU's capacity |
| `trusted` | Keep this `false` so you can only run verified Hugging Face models     |
| `mining` | Set to `true` to run a mining script during idle periods (optional)    |

---

## Step 4: Start the Node

```bash
./run-node.sh
```

The first start may take a few minutes while Python dependencies are installed. Once running, your node will connect to the public network, start receiving job assignments, and accumulate rewards.

---

## That's It

Your worker is now live and will automatically process jobs and earn rewards.

Rewards are accrued to the wallet address defined in your configuration. Enter your address in the [Smartnodes dashboard](https://smartnodes.ca/app) to view and claim rewards.

## Troubleshooting

If you encounter issues, please open a [GitHub Issue](https://github.com/tensorlink-lab/tensorlink/issues) and include:

* Steps to reproduce the problem
* Relevant node logs
* Any error messages

Below are some common issues:

**`nvidia-smi` not found**
CUDA drivers are not installed. Follow the [NVIDIA CUDA installation guide](https://developer.nvidia.com/cuda-downloads) for your operating system.

**Node fails to connect**
Check that your internet connection is stable and ensure no firewall is blocking outbound traffic on port `38751`.

**Out of memory errors**
Reduce `max_vram_gb` in `config.json` to leave additional headroom for system processes.

**Other issues**
Join the [Discord](https://discord.gg) for support or open a [GitHub Issue](https://github.com/tensorlink-lab/tensorlink/issues).
