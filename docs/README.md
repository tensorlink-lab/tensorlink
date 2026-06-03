This README is to guide you through the docs and examples.

## Table of Contents

- [Tensorlink Overview](#overview)
- [Quick Start](getting-started.md)
  - If you don't know where to begin, start here!
- [Distributed Models (Python)](distributed-models.md)
- [API](api.md)
- [Running Nodes](nodes.md)
- [Examples](#python-examples)

--- 

# Overview


Tensorlink is a Python library that enables distributed inference and training of PyTorch models across 
peer-to-peer networks, allowing users to aggregate computing power for private use or access publicly 
donated resources. Through Tensorlink’s native PyTorch object wrappers and HTTP endpoint, bootstrapping 
peers and deploying models is seamless.

On the public network, Tensorlink forms a distributed mesh of compute nodes that provides low-cost access 
to pooled GPUs. Workers automatically combine their computational power to collectively host models larger 
than what any single device can support, with partitioning and execution handled behind the scenes.

Users can also contribute compute to the public network to earn rewards, or combine personal GPUs and devices 
into private AI clusters with resources exposed through the same PyTorch wrappers and APIs.

## Key Features

Tensorlink serves diverse computational scenarios, from prototyping large models to building production-grade AI 
services with cost-effective inference. Secure sensitive workloads on your own infrastructure, experiment with 
distributed fine-tuning on models too large for a single device, or contribute idle compute to earn rewards while 
supporting the network.

>#### OpenAI-Compatible REST API
>Access Hugging Face models through familiar OpenAI-style endpoints including /v1/responses and /v1/chat/completions. 
> Developers can maintain existing workflows while operating models across a distributed compute network. APIs support 
> both streaming and non-streaming responses, with backend execution transparently handled by worker nodes. Available 
> in free public tier and private deployment modes for production workloads.

>#### DistributedModel
>A drop-in wrapper around torch.nn.Module objects that transparently distributes model execution across one or many 
> nodes. The system automatically parses model architectures, intelligently shards layers across available compute 
> resources, and orchestrates forward and backward passes while preserving the standard PyTorch interface (.forward(), 
> .backward(), .parameters()). Works with both pre-trained Hugging Face models and custom PyTorch architectures.

>#### DistributedOptimizer
>Complements DistributedModel with automatic gradient aggregation and synchronized parameter updates across distributed 
> workers. Fully compatible with standard PyTorch optimizers (Adam, AdamW, SGD) and Hugging Face's optimization 
> libraries, ensuring seamless integration into existing training pipelines. Supports adaptive learning rates, weight 
> decay, and momentum across the network.

>#### Flexible Network Deployment
>Deploy nodes in three distinct modes tailored to your use case. Public mode connects to the global Tensorlink network 
> where users contribute GPU resources and earn token rewards through the Smartnodes blockchain layer. Private mode 
> creates isolated clusters on your own infrastructure, ideal for sensitive workloads that require complete data 
> privacy. Local mode enables offline development and testing without any network connectivity, perfect for prototyping 
> before deployment.

>#### Worker and Validator Nodes
>Tensorlink's architecture separates compute execution from request routing. Worker nodes execute model operations on 
> GPU hardware, handling tensor operations, gradient computations, and model weight storage. Validator nodes coordinate 
> distributed jobs, route API requests, manage model sharding strategies, and expose HTTP endpoints for external access. 
> A single physical device can run both roles simultaneously, or you can build specialized clusters with dedicated 
> workers and validators for optimal resource allocation.

>#### Incentivized Compute Network
>Public network participation is incentivized through a blockchain-based token reward system built on the Smartnodes 
> network. GPU providers earn rewards proportional to compute contributed, similar to distributed computing projects 
> like Folding@Home or Gridcoin. This economic layer bootstraps GPU availability and sustains free-tier API access for 
> the community. Nodes can optionally enable mining during idle periods to maximize rewards when not processing 
> inference requests, creating a sustainable ecosystem for decentralized AI infrastructure.


>#### Security and Privacy Controls
>Tensorlink provides granular control over model execution and data privacy. For production deployments, you can expose 
> your own hardware as private API endpoints with custom authentication, allowing secure access from any application via 
> API keys. Hybrid workflows can also be built by combining personal hardware with the public network for distributed 
> execution. See below:

 
````mermaid
flowchart LR

A[Tensorlink API & Model Wrappers\n(Routing Layer)]

B[Public Devices / Nodes]
C[Private Nodes]
D[Local Pre/Post Processing + Adapters]
E[Private AI Cluster]
F[Hybrid Distributed Inference Model]

%% Inference routing
A --> C
A --> B

%% Private execution path
C --> D
C --> E
E --> F
D --> F

%% Hybrid output routing
F --> B
````

## Current Limitations
>**⚠️ Early Release Notice**
>Tensorlink is in active development and early access. Users may encounter bugs, performance inconsistencies, and limited network availability. These limitations will be progressively addressed as the network matures and the community grows. We appreciate your patience and feedback as we work toward a stable 1.0 release.

Currently, support focuses on open-source Hugging Face models that do not require API keys or restricted access. Custom model distribution is supported exclusively in trusted mode on private networks, as we develop secure model serialization mechanisms for public custom model deployment. Future updates will introduce custom models and fault-tolerant execution environments on the public network.

Due to limited availability of public worker nodes during early access, models may experience longer queue times or fail to find sufficient compute resources. The network's capacity will scale as more GPU providers join and contribute resources. Consider running private worker nodes or contributing your own GPU to help expand network capacity.

Internet latency and connection quality significantly impact performance for distributed training and inference when using peer-to-peer connections. HTTP API calls are generally unaffected, but Python-based distributed execution may encounter challenges in latency-sensitive or high-throughput scenarios. Fiber internet and wired Ethernet connections are strongly recommended for optimal node performance.

Local and LAN-based clusters achieve the best performance, making them ideal for production training workloads. Likewise, due to LAN speeds, large-scale model training is not optimal, while low-shot fine-tuning would be more practical.

Tensorlink currently supports UNIX-based systems (Linux, macOS) and requires Python 3.10+ with PyTorch 2.3+. Windows users can run Tensorlink via WSL (Windows Subsystem for Linux).

---

## Need Help?

- 💬 **[Join our Discord](https://discord.gg/aCW2kTNzJ2)** for community support
- 📚 **[Read the docs](https://smartnodes.ca/tensorlink/docs)** for API reference
- 🐛 **[Report issues](https://github.com/mattjhawken/tensorlink/issues)** on GitHub