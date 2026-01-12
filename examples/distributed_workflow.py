"""
Tensorlink Distributed Model Workflow Example

This script demonstrates a distributed machine learning workflow by simulating a network with a user, worker, and
validator node collaborating to coordinate the request of, and train a simple model over P2P.

Overview:
---------
1. **Node Initialization**:
   - Launches `UserNode`, `WorkerNode`, and `ValidatorNode` to simulate distributed roles.
   - `UserNode`: Manages job coordination, should be spawned in the main train or inference script.
   - `WorkerNode`: Handles data processing, can be run on local group of computers and bootstrapped to a validator
    manually, as done in this example, or through the tensorlink-miner to donate public compute.
   - `ValidatorNode`: Validates and approves tasks, coordinates resources and connections between worker and user node.
    Mainly ensures operation of the public P2P net, but is still required for local jobs.

2. **Node Connectivity**:
   - Establishes P2P connections among nodes (this process is only required for connecting local or a closed
   group of devices, network bootstrapping is handled automatically for public jobs).
   - Simulates job request through a validator and coordinates the worker with the user to create the distributed model.

3. **Model and Training**:
   - Uses `BertForSequenceClassification` for testing distributed training.
   - Demonstrates model distribution, gradient updates, and optimization across nodes.

"""

from tensorlink.ml import DistributedModel
from tensorlink.nodes import (
    User,
    UserConfig,
    Validator,
    ValidatorConfig,
    Worker,
    WorkerConfig,
)

from transformers import AutoTokenizer
import torch
import logging
import time


LOCAL = True  # Network operations are on localhost when true (i.e. 127.0.0.1)
UPNP = (
    True if not LOCAL else False
)  # Must be activated for public network use. Upgrade to STUN or other in the future.
OFFCHAIN = LOCAL  # Can be used to deactivate on-chain features (for private jobs)

# Parameters for distributed model request.
BATCH_SIZE = 16
PIPELINES = 1
DP_FACTOR = 1
TRAINING = False  # Set true to request train job and get a distributed optimizer

# Chatbot parameters
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
MAX_HISTORY_TURNS = 6
MAX_TOKENS = 2048
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7


if __name__ == "__main__":
    # Launches a node of each type in their own process (Not necessary if just accessing a DistributedModel
    # as a user, it will do this in the background...
    validator = Validator(
        config=ValidatorConfig(
            upnp=UPNP,
            off_chain_test=OFFCHAIN,
            local_test=LOCAL,
            print_level=logging.DEBUG,
        )
    )
    user = User(
        config=UserConfig(
            upnp=UPNP,
            off_chain_test=OFFCHAIN,
            local_test=LOCAL,
            print_level=logging.DEBUG,
        )
    )
    worker = Worker(
        config=WorkerConfig(
            upnp=UPNP,
            off_chain_test=OFFCHAIN,
            local_test=LOCAL,
            print_level=logging.DEBUG,
        )
    )

    # Get validator node information for connecting
    val_key, val_host, val_port = validator.send_request("info", None)
    time.sleep(1)

    # Connect worker node and user node to the validator node.
    # This would only have to be done for local jobs.
    # For public jobs, the user will automatically be bootstrapped to the network.
    worker.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    time.sleep(1)
    user.connect_node(val_host, val_port, node_id=val_key, timeout=5)
    time.sleep(1)

    # Get distributed model directly from HuggingFace without loading
    distributed_model = DistributedModel(model_name, training=False, node=user)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize distributed optimizer
    if TRAINING:
        distributed_optimizer = distributed_model.create_optimizer(
            lr=0.001, weight_decay=0.01
        )
        distributed_model.train()

    # Run a dummy training loop to showcase functionality
    for _ in range(5):
        # Tokenize input
        input_text = "You: Hello Bot."
        inputs = tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        )

        # Generate response
        with torch.no_grad():
            # Then during generation:
            outputs = distributed_model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,  # still valid, optional if eos_token_id is used
                eos_token_id=tokenizer.eos_token_id,  # explicitly recommended
                do_sample=True,  # temperature only has effect if sampling is on
            )

        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Bot: {response}\n")

    # Gracefully shut down nodes
    user.cleanup()
    worker.cleanup()
    validator.cleanup()
