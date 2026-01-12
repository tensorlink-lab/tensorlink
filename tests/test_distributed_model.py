"""
test_local_dist_train.py

This script tests distributed machine learning using Tensorlink's P2P network using local nodes on different
processes. It simulates a local environment with a user, worker, and validator node collaborating to train a
simple model.
"""

from tensorlink.ml import DistributedModel
import torch.nn as nn
import torch.optim as optim
import torch

# Variables for nodes and distributed models
LOCAL = True
UPNP = False
BATCH_SIZE = 16
PIPELINES = 1
DP_FACTOR = 1


def test_model_inference(connected_nodes):
    """Test distributed training with a simple model."""
    validator, user, worker, _ = connected_nodes

    model = "sshleifer/tiny-gpt2"

    distributed_model = DistributedModel(model=model, training=False, node=user)

    with torch.no_grad():
        _ = distributed_model(torch.randint(0, 100, (1, 1)))


def test_model_training(connected_nodes):
    """Test distributed training setup with a tiny encoder model."""
    validator, user, worker, _ = connected_nodes

    model_name = "sshleifer/tiny-gpt2"

    distributed_model = DistributedModel(
        model=model_name,
        training=True,
        optimizer=optim.Adam,
        node=user,
    )

    assert distributed_model is not None

    optimizer = distributed_model.create_optimizer(lr=0.001, weight_decay=0.01)

    assert optimizer is not None

    distributed_model.train()
    optimizer.zero_grad()
    dummy_input = torch.randint(0, 100, (2, 8))
    outputs = distributed_model(dummy_input, labels=dummy_input)

    logits = outputs.logits  # (B, T, V)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = dummy_input[:, 1:].contiguous()

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    loss.backward()
