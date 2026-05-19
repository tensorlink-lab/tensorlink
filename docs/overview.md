# Overview
## Tensorlink: Distributed AI Infrastructure for PyTorch and Hugging Face Models

Tensorlink is a Python library and computational platform that enables distributed inference 
and training of PyTorch models across peer-to-peer networks. It allows users to pool GPU power 
to run larger models and remotely access machine learning resources in Python or via HTTP. 
Tensorlink's native PyTorch object wrappers and http endpoint simplify bootstrapping peers and 
configuring models for distributed execution, making large-scale inference and training easy to deploy.

Unlike centralized cloud infrastructure, Tensorlink forms a distributed mesh of compute nodes that provides 
low-cost access to pooled public GPUs. Workers automatically combine their computational power to collectively 
host models larger than what any single GPU can support, with partitioning and execution handled transparently 
behind the scenes. Tensorlink connects devices and exposes them through PyTorch object wrappers or a flexible 
HTTP API, enabling both public and private GPU networks to serve Python applications, services, and web apps.

Whether you're running large language models without local GPU capacity, deploying private inference infrastructure 
accessible from any device, building agentic workflows with on-demand LLM APIs, or conducting distributed training with 
automatic model and data parallelism, Tensorlink provides the infrastructure to make it happen. Users can also expose 
their personal GPUs as private API endpoints, allowing secure access to AI resources from external applications while 
maintaining complete data privacy and control.

## Common Use Cases
- **Running Models Beyond Local GPU Capacity:** Execute large language models (14B+ parameters) that exceed your available VRAM by distributing layers across multiple GPUs on the network. Ideal for researchers and developers who need access to state-of-the-art models without investing in expensive hardware.

- **Private API Endpoints for Personal Hardware:** Host models on your home PC or workstation and access them securely from mobile apps, web services, or third-party applications via private API keys. Your data never leaves your infrastructure, ensuring complete privacy and control while maintaining the convenience of cloud-style API access.

- **Cost-Effective Production AI Services:** Build web services, chatbots, and agentic applications with access to free-tier public APIs or low-cost private infrastructure. Eliminate recurring cloud inference costs while maintaining the scalability and reliability needed for production deployments.

- **Distributed Model Training and Fine-Tuning:** Train and fine-tune models with automatic gradient synchronization across distributed workers. Leverage model parallelism to train architectures too large for a single GPU, or use data parallelism to accelerate training across multiple devices on your local network.

- **Earning Rewards for Idle Compute:** Monetize unused GPU capacity by contributing to the public network. Perfect for gaming rigs, workstations, or mining hardware that sits idle for significant portions of the day. Earn blockchain-based rewards while supporting the decentralized AI ecosystem.

## Key Features

Tensorlink serves diverse computational scenarios, from prototyping large models to building production-grade AI 
services with cost-effective inference, the platform adapts to your needs. Secure sensitive workloads on your own 
infrastructure, experiment with distributed fine-tuning on models too large for a single GPU, or contribute idle compute 
to earn rewards while supporting the network.

>#### OpenAI-Compatible REST API
>Access Hugging Face models through familiar OpenAI-style endpoints including /v1/generate and /v1/chat/completions. Developers can maintain existing workflows while operating models across a distributed compute network. APIs support both streaming and non-streaming responses, with backend execution transparently handled by worker nodes. Available in free public tier and private deployment modes for production workloads.

>#### DistributedModel
>A drop-in wrapper around torch.nn.Module objects that transparently distributes model execution across one or many nodes. The system automatically parses model architectures, intelligently shards layers across available compute resources, and orchestrates forward and backward passes while preserving the standard PyTorch interface (.forward(),.backward(),.parameters()). Works with both pre-trained Hugging Face models and custom PyTorch architectures.

>#### DistributedOptimizer
>Complements DistributedModel with automatic gradient aggregation and synchronized parameter updates across distributed workers. Fully compatible with standard PyTorch optimizers (Adam, AdamW, SGD) and Hugging Face's optimization libraries, ensuring seamless integration into existing training pipelines. Supports adaptive learning rates, weight decay, and momentum across the network.

>#### Flexible Network Deployment
>Deploy nodes in three distinct modes tailored to your use case. Public mode connects to the global Tensorlink network where users contribute GPU resources and earn token rewards through the Smartnodes blockchain layer. Private mode creates isolated clusters on your own infrastructure, ideal for sensitive workloads that require complete data privacy. Local mode enables offline development and testing without any network connectivity, perfect for prototyping before deployment.

>#### Worker and Validator Nodes
>Tensorlink's architecture separates compute execution from request routing. Worker nodes execute model operations on GPU hardware, handling tensor operations, gradient computations, and model weight storage. Validator nodes coordinate distributed jobs, route API requests, manage model sharding strategies, and expose HTTP endpoints for external access. A single physical device can run both roles simultaneously, or you can build specialized clusters with dedicated workers and validators for optimal resource allocation.

>#### Security and Privacy Controls
>Tensorlink provides granular control over model execution and data privacy. The trusted mode flag determines whether nodes accept custom user-supplied models or restrict execution to verified Hugging Face checkpoints only. Private networks ensure sensitive data never leaves your infrastructure, while priority node configurations let you explicitly define which devices handle your workloads. For production deployments, you can expose your own hardware as private API endpoints with custom authentication, ensuring your data flows exclusively through controlled infrastructure while remaining accessible from any application via secure API keys.

>#### Incentivized Compute Network
>Public network participation is incentivized through a blockchain-based token reward system built on the Smartnodes network. GPU providers earn rewards proportional to compute contributed, similar to distributed computing projects like Folding@Home or Gridcoin. This economic layer bootstraps GPU availability and sustains free-tier API access for the community. Nodes can optionally enable mining during idle periods to maximize rewards when not processing inference requests, creating a sustainable ecosystem for decentralized AI infrastructure.

## Current Limitations
>**⚠️ Early Release Notice**
>Tensorlink is in active development and early access. Users may encounter bugs, performance inconsistencies, and limited network availability. These limitations will be progressively addressed as the network matures and the community grows. We appreciate your patience and feedback as we work toward a stable 1.0 release.

Currently, support focuses on open-source Hugging Face models that do not require API keys or restricted access. Custom model distribution is supported exclusively in trusted mode on private networks, as we develop secure model serialization mechanisms for public custom model deployment. Future updates will introduce custom models and fault-tolerant execution environments on the public network.

Due to limited availability of public worker nodes during early access, models may experience longer queue times or fail to find sufficient compute resources. The network's capacity will scale as more GPU providers join and contribute resources. Consider running private worker nodes or contributing your own GPU to help expand network capacity.

Internet latency and connection quality significantly impact performance for distributed training and inference when using peer-to-peer connections. HTTP API calls are generally unaffected, but Python-based distributed execution may encounter challenges in latency-sensitive or high-throughput scenarios. Fiber internet and wired Ethernet connections are strongly recommended for optimal node performance.

Local and LAN-based clusters achieve the best performance, making them ideal for production training workloads. Likewise, due to LAN speeds, large-scale model training is not optimal, while low-shot fine-tuning would be more practical.

Tensorlink currently supports UNIX-based systems (Linux, macOS) and requires Python 3.10+ with PyTorch 2.3+. Windows users can run Tensorlink via WSL (Windows Subsystem for Linux).
