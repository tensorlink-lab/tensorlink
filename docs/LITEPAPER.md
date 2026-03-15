# Tensorlink Whitepaper

**Version 0.2 - Pre-Release**  
*Decentralized AI Inference & Training over Peer-to-Peer Networks*

---

## Abstract

Tensorlink is a decentralized computational platform and Python library that democratizes access to large-scale AI model inference and training. By enabling distributed execution of PyTorch models across consumer hardware through a peer-to-peer network, Tensorlink eliminates the need for costly centralized infrastructure while preserving user privacy and promoting sustainable computing practices. The platform features automated model parsing and distribution, Hugging Face integration, REST APIs for inference, and a built-in incentive mechanism powered by the Smartnodes (SNO) token that rewards contributors for sharing computational resources.

---

## 1. Introduction

### 1.1 The Scale Problem

The prevalence of large language models has fueled an unprecedented demand for computational power. State-of-the-art models like GPT-4, with hundreds of billions of parameters, require large arrays of graphics or tensor processing units for training and operation. Depending on whether we use single or half-precision floating-point arithmetic, such models would require 300-600 gigabytes of memory just to store their weights, let alone train them. 

The current most powerful accelerator, NVIDIA's A100, has 80GB of memory and costs up to $20,000. Even for basic inference, one would require at least five of these high-end GPUs or a multi-node cluster, making it inaccessible to researchers with limited budgets and most individuals with consumer hardware.

Since models of this scale exceed the capacity of a single device, distributed training has become essential. However, setting up distributed models on a cluster of computers requires extensive customization to model workflows for distributing workloads and synchronization. Traditional deployment methods such as cloud services and supercomputer rentals come with substantial costs and centralization risks, making efficient and affordable scaling a significant challenge.

### 1.2 The Centralization Problem

Beyond cost, existing cloud-based AI services present several critical barriers:

**Privacy Concerns:** Cloud AI services store user data on external servers, creating risks of data breaches and unauthorized data harvesting. All major "free" commercial LLM services log and analyze user inputs, fundamentally compromising privacy. Users surrender control over their sensitive data and intellectual property.

**Environmental Impact:** Centralized data centers place growing strain on power grids, contributing to unsustainable energy consumption patterns. The concentration of compute power in massive facilities exacerbates regional power demands.

**Accessibility Gap:** Individual developers, small teams, and researchers lack the resources to deploy cutting-edge AI models, creating a widening divide between AI capabilities and accessibility. Innovation becomes restricted to well-funded organizations.

**Vendor Lock-in:** Dependence on specific cloud providers creates technical and economic constraints that limit flexibility and increase long-term costs.

### 1.3 Leveraging Distributed Consumer Hardware

We can leverage concepts of volunteer computing to distribute the required work among a network of regular PCs. A single PC may be slow and unreliable, but the combined performance of a fleet of PCs can match that of the largest supercomputers. 

The key to obtaining such a network is to incentivize the gaming demographic to offer up their idle compute for rewards. It is estimated that there are approximately 1.855 billion PC gaming users worldwide. According to several PC building websites, most popular desktops are equipped with RTX 2080/2080 Ti or GTX 1080Ti GPUs—devices that are 50–80% as fast as Tesla V100 for deep learning, a popular GPU among researchers. 

As a rough estimate, the combined throughput of 10,000 desktops is 8-15 times that of server pods with 512 V100 GPUs. With the profitability of cryptocurrency mining being much lower than in the past, running Tensorlink presents a promising alternative for those who wish to monetize their compute. At the highest level, running Tensorlink and crypto mining can essentially be seen as the same thing: both are means to earn rewards for computational work.

### 1.4 The Tensorlink Solution

Tensorlink introduces a versatile framework designed to facilitate neural network offloading and acceleration within PyTorch, a leading machine learning framework in Python. Tensorlink offers a straightforward, plug-and-play solution that seamlessly parses and distributes any PyTorch-based neural network model, including transformers utilized in libraries such as Hugging Face.

Through this approach, Tensorlink aims to overcome existing economic and technical barriers by offering an affordable, plug-and-play, decentralized solution that enhances accessibility and ease of use for researchers, businesses, and individuals alike.

**Core Objectives:**
- **Distribute workloads** across consumer hardware via direct peer-to-peer connections
- **Preserve privacy** by processing data locally without mandatory centralized storage
- **Reduce costs** by leveraging idle computational resources
- **Promote sustainability** through distributed, grid-friendly computing
- **Maintain familiarity** through seamless PyTorch integration

Users are provided with a model wrapper that handles model distribution and connects them directly to a network of computing power, while maintaining their existing workflows on their own device. This wrapper gives the perception of operating models on a user's own hardware, which reduces complexity, especially for larger models that require hosting across multiple devices.

By incentivizing the contribution of idle computing power through smart contract-based payments and rewards (via Smartnodes), Tensorlink aspires to cultivate a distributed network that competes with the capabilities of supercomputers. This strategy democratizes access to computational resources and serves as a viable alternative to traditional cryptocurrency mining.

### 1.5 Use Cases & Applications
#### Large Model Inference Without Local Hardware
- **Scenario:** Developers building applications that require AI capabilities but lack local GPU resources.
- **Solution:** Run models with zero VRAM requirements through Tensorlink's distributed network. Useful for web services, mobile apps, and server workflows that need on-demand AI inference.

#### Private Inference Setups
- **Scenario:** Organizations or individuals requiring AI capabilities with strict privacy requirements.
- **Solution:** Host an LLM on a home PC or private cluster, access it securely from mobile/web apps via API. Tensorlink enables semi-private or fully private AI usage—not entirely private if using public nodes, but still not centralized in one database. Users can specify their own Tensorlink API key from their Worker node, allowing fully private chat with models via their own hardware.

#### Research & Development
- **Scenario:** Academic researchers training or fine-tuning large language models without institutional GPU clusters.
- **Solution:** Tensorlink enables researchers to pool resources or access public compute, reducing barriers to cutting-edge research. Particularly useful for low-shot fine-tuning on very large models that exceed single-device capacity.

#### Agentic Workflows
- **Scenario:** AI-controlled pipelines and multi-step reasoning tasks requiring quick access to various models.
- **Solution:** Quick API access to models for AI agents, enabling complex workflows without managing infrastructure. Supports chaining multiple model calls and dynamic model selection.

#### Edge AI Deployment
- **Scenario:** IoT networks requiring distributed inference across edge devices without cloud dependencies.
- **Solution:** Tensorlink enables coordinated inference across device clusters, keeping data local while leveraging distributed compute power.

#### Monetizing Idle Hardware
- **Scenario:** GPU owners with idle compute capacity—gamers when not gaming, former mining rigs, or workstations during off-hours.
- **Solution:** Running Tensorlink workers generates passive income by processing distributed AI workloads. Provides an alternative to cryptocurrency mining with potentially better returns.

---

## 2. Architecture Overview

### 2.1 Core Components

Tensorlink's architecture consists of three primary node types and two developer-facing interfaces:

#### Developer Interfaces

**DistributedModel:** A wrapper around `torch.nn.Module` that automatically parses and distributes models across the 
network. Preserves the standard PyTorch interface including `forward()`, `backward()`, and `parameters()`, enabling 
drop-in replacement in existing codebases. Complimented by the `DistributedOptimizer` which provides synchronized 
parameter updates across nodes.

**Rest API:** Tensorlink provides access to free and paid tier inference on open-source models through HTTP.

#### Node Types

**User Nodes:** Initiate training or inference jobs and coordinate distributed execution. User nodes submit model definitions and manage the workflow while the actual computation happens on worker nodes. Users can operate locally or connect to the public network seamlessly.

**Worker Nodes:** Execute computational tasks assigned by users. Workers can be any device with computational resources (GPUs, CPUs) and earn SNO token-based rewards for processing jobs. Workers store critical training data including model updates, input-output tensors, and gradients for verification purposes.

**Validator Nodes:** Coordinate job distribution, verify computational integrity, and facilitate secure peer-to-peer connections. Validators maintain network stability, ensure fair reward distribution, and periodically request stored data from workers to verify validity and ensure consistency. Validators must collateralize their nodes with SNO tokens to ensure commitment and accountability.


### 2.2 Network Architecture

Tensorlink operates on a flexible hybrid network model:

**Public Network:** By default, nodes connect through a smart contract-secured peer-to-peer mesh powered by Smartnodes, enabling global resource sharing with token-based incentives. This decentralized architecture allows anyone to contribute or consume compute resources.

**Private Networks:** Organizations can deploy isolated Tensorlink clusters for proprietary workloads while maintaining the same developer interface. Private deployments offer maximum control over data and resources for sensitive applications.

---

## 3. Technical Design

### 3.1 Scaling Models Over the Internet

To achieve efficient scaling of neural networks over the internet, Tensorlink leverages several training parallelism techniques:

**Data Parallelism:** A technique that involves splitting the input data into smaller subsets that can be processed simultaneously across multiple devices. Each device receives a copy of the entire model and processes a different subset of the data. Once all devices have computed their results for their respective data shards, the gradients from each node are averaged and synchronized to update the model parameters. This approach is widely used in large-scale training because it is straightforward to implement and scales well across multiple machines.

**Model Parallelism:** Divides a neural network model itself across multiple devices, with each device holding only a portion of the model layers. When data is processed, it moves sequentially through each device as it progresses through different model segments. This approach is beneficial when models are too large to fit entirely on a single device. For example, with a very large neural network, the first few layers might be on one GPU, the next layers on another, and so forth.

**Pipeline Parallelism:** A hybrid approach that combines aspects of model parallelism to reduce latency by dividing the model across devices but also overlaps the execution of multiple data batches. Instead of processing one data batch through all model layers sequentially, the model is split into stages or "pipelines," each handling a portion of the model's layers. Data batches are fed into the pipeline in a staggered fashion, so while one batch is being processed by one stage, the next batch can start processing in a different stage. This overlapping reduces idle time and can significantly speed up training by maximizing throughput.

<p align="center">
  <img src="images/pipeline.png" alt="Pipeline parallelism micro-batching." width="520"/>
</p>
<p align="center"><strong>Figure 1:</strong> <em>Pipeline parallelism micro-batching to mitigate latency in distributed learning environments.</em></p>

### 3.2 Model Parsing and Distribution

Tensorlink provides algorithms for analyzing and parsing neural networks in the context of distributed training. These algorithms take factors such as memory usage, the number of workers/intermediates required in the workflow, and available network resources to optimize the distribution of models across the network.

**Distribution Process:**

1. **Parsing:** The system analyzes the model's computational graph to identify natural partition points (typically between layers or modules). The parser considers memory requirements, computational complexity, and data dependencies.

2. **Allocation:** Based on worker capabilities (memory, compute power) and network topology, the system assigns model segments to optimize performance. The allocation algorithm balances load across workers while minimizing inter-node communication.

3. **Communication:** During forward/backward passes, intermediate activations and gradients are transmitted between workers using optimized serialization protocols. Tensorlink employs efficient tensor compression and batching to minimize network overhead.

**Privacy-Oriented Distribution:**

For users comfortable sharing their data, Tensorlink allows the distribution of full models and datasets to validators. This method minimizes the impact of internet latency, as it involves a limited number of offloaded modules in the pipeline.

Additionally, models can be distributed in a privacy-oriented approach for those prioritizing data security. By further fragmenting model structures, only a small portion of the model is accessible to any single node. Users can further safeguard their training data by employing **hybrid distributed models**, where a small portion of the model is processed on the user side before the data continues through the pipeline. This effectively obfuscates sensitive model and training data, ensuring a more secure training environment, potentially at the sacrifice of additional training time or computational cost.

<p align="center">
  <img src="images/ML Flow Chart.png" alt="Distributed model architecture." width="520"/>
</p>
<p align="center"><strong>Figure 2:</strong> <em>An example illustrating model distribution and parallelization among workers, as well as intermediate computation by the user.</em></p>

### 3.3 Proof of Learning (Under Development)

Proof of Learning (PoL) is a foundational concept of Tensorlink, designed to enhance the integrity and reliability of distributed learning. During training processes, worker nodes are responsible for storing critical data related to model updates, input-output tensors, and gradients. Validators are tasked with periodically requesting this data throughout the training process to verify its validity and ensure consistency among worker nodes.

In Tensorlink's PoL process, efficient, low-cost checks occur within workers handling replica model sections, allowing for rapid validation among workers without incurring significant computational costs. For more comprehensive verification, collateralized validators perform in-depth checks across individual workers and larger model sections to ensure reliability and accuracy of training.

**Verification Mechanisms:**

**Gradient Validation:** Validator nodes compare stored gradient values across worker nodes to ensure alignment and accuracy. Any gradient discrepancies may indicate potential malicious activity, signaling the need for further inspection.

**Forward Pass Validation:** Validators assess the output tensors generated during forward passes by comparing them to expected outputs based on provided input tensors. This step is essential to ensure each worker node processes data accurately and produces reliable predictions.

**Cross-Validation:** Validator nodes cross-check data between worker nodes by taking data from one node and using it as input in another. This independent validation ensures consistency across nodes, helping to identify any discrepancies that might impact model accuracy.

### 3.4 Security & Trust Model

**Trusted Mode:** For private networks and known workers, full model weights and data can be shared directly for maximum performance. This mode is suitable for organizational deployments where all participants are authenticated and trusted.

**Untrusted Mode (Under Development):** For public networks, Tensorlink will implement:
- **Encrypted Computation:** Workers process encrypted model segments without accessing plaintext weights or activations
- **Zero-Knowledge Proofs:** Verify computational correctness without revealing intermediate results
- **Differential Privacy:** Protect training data privacy through noise injection mechanisms
- **Trusted Execution Environments:** Leverage hardware enclaves (Intel SGX, ARM TrustZone) for sensitive computation

---

## 4. Economic Model & Incentives

### 4.1 The Smartnodes Token (SNO)

Tensorlink is powered by Smartnodes and its native token, SNO. SNO functions as a vital incentive for encouraging network participation while promoting growth and development.

**Token Utility:**

**Payment Medium:** Users pay SNO or ETH to access premium features, guaranteed resources, and custom model deployments. SNO serves as the primary means of payment for Tensorlink and Smartnodes services.

**Reward Mechanism:** Workers and validators earn SNO and ETH proportional to their contributions. Distribution is determined by factors such as computational power provided, job duration, and the worker's reputation within the network.

**Collateralization:** Validators must collateralize their nodes with a specified amount of SNO tokens, adding an extra layer of security and trust to the ecosystem. This ensures validator commitment and accountability.

### 4.2 Reward Distribution

**Workers earn SNO tokens based on:**
- Computational intensity (FLOPs, memory usage)
- Job completion time
- Quality of service (uptime, latency)
- Reputation score within the network

**Validators earn fees for:**
- Job coordination and routing
- Verification of computational correctness through Proof of Learning
- Network infrastructure maintenance
- Facilitating secure peer connections

**Users can:**
- Access free inference on select models (subject to availability and rate limits)
- Pay SNO tokens for guaranteed resources and custom models
- Earn SNO tokens by contributing unused resources as workers

### 4.3 Pricing Structure & Network Maturity

To foster user engagement and research, jobs within the Tensorlink network will initially commence without requiring payment as part of an onboarding and incentivization strategy to attract users. This free tier provides access to a rotating selection of popular models for public inference with no cost.

However, as the network matures and scales, a gradual transition will occur, necessitating payments for certain types of jobs to sustain network operation. The paid tier will offer:
- Guaranteed resource allocation
- Access to larger models
- Custom deployments
- Priority job scheduling

**Dynamic Pricing:** Resource pricing will be based on:
- Model size and computational requirements
- Requested latency guarantees
- Network utilization and demand
- Worker availability and reputation
