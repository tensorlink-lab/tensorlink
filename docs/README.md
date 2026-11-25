# Overview

**Tensorlink: Distributed model and networking infrastructure for PyTorch and Hugging Face**

Tensorlink is a Python library and computational platform for remote access to PyTorch models, allowing you to connect 
to models hosted on machines through native PyTorch integrations or an OpenAI-compatible API. It also enables seamless 
distributed execution of models across pooled computers. Nodes can combine their memory to host models larger than 
any single computer can hold across any platform (CUDA, Apple Silicon, AMD, etc.), and Tensorlink handles partitioning, 
routing, and execution behind the scenes.

Tensorlink supports three main deployment modes:
- **Private:** your own machines, isolated from the public network.
- **Public:** shared community GPUs, with rewards for the people who contribute them.
- **Hybrid:** a private cluster that can overflow to public workers when it needs more capacity. Input pre-processing and output post-processing stay on your private nodes, so public workers only handle intermediate computation and never see the raw inputs or outputs.
  - Combine with custom adapters on your private nodes for stronger privacy in hybrid public/private setups.

---

## Quickstart

Tensorlink offers four ways to get started. Pick the one that matches your use case.

| Goal                                                       | Start here                                          |
|------------------------------------------------------------|-----------------------------------------------------|
| Access distributed models through an OpenAI-compatible API | [API reference](use-tensorlink/api.md)              |
| Access distributed models directly through PyTorch         | [Distributed Models](distributed-models.md)         |
| Node setup & network configurations                        | [Run a worker](use-tensorlink/worker-quickstart.md) |

---

## What you can do with it

- **Host a private API on your own hardware.** Serve models from one or more personal computers to your own apps through an OpenAI-compatible endpoint, with your data staying on your infrastructure.
- **Run models beyond your local VRAM.** Distribute larger models across several GPUs on a network.
- **Build apps without per-token cloud bills.** Use the public API or low-cost private infrastructure for chatbots, services, and agent workflows.
- **Fine-tune across machines.** Train with gradient synchronization across workers (see [Training](use-tensorlink/python-training.md)).
- **Earn rewards for idle compute.** Contribute a gaming rig or workstation to the public network and earn rewards through a blockchain-based rewards and payments system (see [Smartnodes](https://github.com/tensorlink-lab/smartnodes-core)).

### Main Components

| Component | What it is | Docs |
|---|---|---|
| **HTTP API** | OpenAI-compatible chat completions and embeddings, plus model management and network endpoints | [API reference](use-tensorlink/api.md), [Network endpoints](use-tensorlink/network-api.md) |
| **`DistributedModel`** | A wrapper that makes a remote, sharded model look like a normal `torch.nn.Module` | [Running models](use-tensorlink/python-inference.md) |
| **`DistributedOptimizer`** | Gradient aggregation and synchronized parameter updates across workers | [Training](use-tensorlink/python-training.md) |
| **Nodes** | Workers run the model on GPUs, validators coordinate jobs and serve the API, and user nodes request jobs | [Architecture](architecture.md) |

### Current limitations

> **Early release.** Tensorlink is in active development. Expect bugs, performance variation, and limited public capacity until the 1.0 release.

- **Models:** The public network supports open-source Hugging Face models that don't require API keys or gated access.
- **Custom models:** Custom PyTorch architectures run only in trusted mode on private networks. Public support depends on a secure serialization mechanism that isn't built yet. See [Private cluster](run-a-node/private-cluster.md).
- **Capacity:** With few public workers, models can fail to find enough compute. Running a private worker or [contributing your GPU](run-a-node/worker.md) helps.
- **Latency:** Internet latency and connection quality matter a lot for distributed execution over P2P. Ethernet and a fiber optic service (when running over WAN) are optimal for running a node.
- **Platforms:** Linux and macOS, with Windows through WSL. Python 3.11+ and PyTorch 2.3+.
