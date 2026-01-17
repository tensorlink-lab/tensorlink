"""
test_model_api.py

This script tests distributed machine learning requests via node API on local nodes.
It simulates an endpoint where model requests, generations, and streamed generations can
be tested on a tiny Hugging Face model.

Furthermore, two types of models are tested to ensure full coverage of possible workflows: one tiny model
that can be loaded on a single worker, and a slightly larger model that will require model sharding.
"""

import requests
import pytest
import time
import json


OFFCHAIN = True
LOCAL = True
UPNP = False

SERVER_URL = "http://127.0.0.1:64747"
MODELS = [
    pytest.param(
        {
            "name": "sshleifer/tiny-gpt2",
            "timeout": 60,
            "sleep": 5,
            "parsed": False,
        },
        id="tiny-gpt2",
    ),
    pytest.param(
        {
            "name": "HuggingFaceTB/SmolLM-135M",
            "timeout": 120,
            "sleep": 10,
            "parsed": True,
        },
        id="smollm-135m",
    ),
]


@pytest.fixture(params=MODELS)
def model_cfg(request):
    return request.param


@pytest.fixture
def requested_model(connected_wwv_nodes, model_cfg):
    """
    Request a model on the Tensorlink network before tests run.
    Using WWV nodes (Worker-Worker-Validator) for API tests.
    """
    worker, worker2, validator, _ = connected_wwv_nodes

    payload = {
        "hf_name": model_cfg["name"],
        "model_type": "causal",
    }

    response = requests.post(
        url=f"{SERVER_URL}/request-model",
        json=payload,
        timeout=model_cfg["timeout"],
    )

    assert response.status_code == 200

    return model_cfg


def test_generate(connected_wwv_nodes, requested_model):
    """
    Test generate request via API
    """
    cfg = requested_model
    worker, worker2, validator, _ = connected_wwv_nodes

    time.sleep(cfg["sleep"])

    generate_payload = {
        "hf_name": cfg["name"],
        "message": "Hi.",
        "max_new_tokens": 10,
        "do_sample": True,
        "num_beams": 2,
    }

    response = requests.post(
        f"{SERVER_URL}/v1/generate",
        json=generate_payload,
        timeout=100,
    )

    assert response.status_code == 200


def test_streaming_generation(connected_wwv_nodes, requested_model):
    """
    Test generate request with token-by-token streaming via API
    """
    cfg = requested_model
    worker, worker2, validator, _ = connected_wwv_nodes

    time.sleep(cfg["sleep"])

    generate_payload = {
        "hf_name": cfg["name"],
        "message": "Hi.",
        "max_new_tokens": 10,
        "stream": True,
        "do_sample": False,
        "num_beams": 1,
    }

    response = requests.post(
        f"{SERVER_URL}/v1/generate",
        json=generate_payload,
        stream=True,
        timeout=120,
    )

    assert response.status_code == 200

    full_text = ""
    received_tokens = 0
    done_received = False

    for line in response.iter_lines():
        if not line:
            continue

        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue

        data = decoded[6:]

        if data == "[DONE]":
            done_received = True
            break

        chunk = json.loads(data)
        delta = chunk["choices"][0].get("delta", {})
        token = delta.get("content")

        if token:
            received_tokens += 1
            full_text += token

    assert done_received
    assert received_tokens > 0
    assert full_text
