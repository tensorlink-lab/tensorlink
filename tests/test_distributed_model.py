"""
test_distributed_model.py

This script tests distributed machine learning in PyTorch using Tensorlink's P2P network on
local nodes. It simulates a local environment with a user, worker, and validator node collaborating
to run a tiny Hugging Face model.

Furthermore, two types of models are tested to ensure full coverage of possible workflows: one tiny model 
that can be loaded on a single worker, and a slightly larger model that will require model sharding.
"""

from tensorlink.ml import DistributedModel
import torch.optim as optim
import torch

# Variables for nodes and distributed models
LOCAL = True
UPNP = False
BATCH_SIZE = 16
PIPELINES = 1
DP_FACTOR = 1


MODEL_NAME = "sshleifer/tiny-gpt2"


def test_model_inference(connected_uwv_nodes):
    """
    Test distributed inference with a simple model, ensures distributed forward and
    generate functions work from torch requests.
    Using UWV nodes (User-Worker-Validator) for distributed model tests.
    """
    user, worker, validator, _ = connected_uwv_nodes

    distributed_model = DistributedModel(model=MODEL_NAME, training=False, node=user)

    with torch.no_grad():
        _ = distributed_model(torch.randint(0, 100, (1, 1)))


def test_model_training(connected_uwv_nodes):
    """
    Test distributed training setup with a tiny encoder model. Ensures backward pass
    and distributed optimizer functions work.
    Using UWV nodes (User-Worker-Validator) for distributed model tests.
    """
    user, worker, validator, _ = connected_uwv_nodes

    distributed_model = DistributedModel(
        model=MODEL_NAME,
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


def test_multiple_models(connected_uwv_nodes):
    """
    Test a few tiny models that will require different distributed configs (i.e., single offloaded model and
    multiple offloaded modules)
    Using UWV nodes (User-Worker-Validator) for distributed model tests.
    """
    user, worker, validator, _ = connected_uwv_nodes
    # TODO: Implement multiple model testing
    pass
