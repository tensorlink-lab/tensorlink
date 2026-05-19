# Getting Started
Install Tensorlink and set up your environment for distributed AI workloads.

Tensorlink lets you run large Hugging Face models across distributed GPU infrastructure without needing local hardware. 
If you're looking to access compute, the quickest path is through the [API](api.md) (no GPU required). If you're 
just looking to contribute compute to the public network, see the [Worker Setup Guide](worker-guide.md).

---

## Requirements 
- **OS**: Linux or macOS. Windows users can use WSL.
- **Python**: 3.10 or higher. Check with `python --version`.
- **GPU**: Not required to use Tensorlink as a client. A modern NVIDIA GPU or Apple Silicon is needed if you plan to
[run your own node](nodes.md).
  - AMD, Intel, Multi-GPU configurations have not been thoroughly test and some issues may arise. 

---

## Installation
 
We recommend using a virtual environment to keep dependencies isolated.
 
### Step 1: Create a virtual environment
 
```bash
python -m venv .venv
source .venv/bin/activate
```
 
Your terminal prompt should show `(.venv)` once active.
 
### Step 2: Upgrade pip
 
```bash
pip install --upgrade pip setuptools wheel
```
 
### Step 3: Install Tensorlink
 
```bash
pip install tensorlink
```
 
This installs Tensorlink and its core dependencies. If PyTorch is not already present, pip may install a CPU-only version automatically. For GPU-accelerated workers, install PyTorch separately with the correct CUDA version first:
 
```bash
# Example for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
 
See the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) for other configurations. If you hit installation errors, make sure pip is up to date:
 
```bash
pip install --upgrade pip
```
 
For other issues, check the [GitHub Issues](https://github.com) page or ask in the Discord community.

---
 
## What's Next?

- To set up distributed models across multiple personal computers over LAN or WAN, see the [Worker and Validator Node Setup](nodes.md).
- To run Hugging Face models using public compute or create a private GPU network for remote inference in Python, see [Distributed Models](distributed-models.md).
- To quickly get started with hosted inference using an OpenAI-compatible interface, check out the [API Guide](api.md).