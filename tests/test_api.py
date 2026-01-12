from tensorlink import WorkerNode, ValidatorNode, UserNode

import requests
import logging
import pytest
import time
import json


OFFCHAIN = True
LOCAL = True
UPNP = False

SERVER_URL = "http://127.0.0.1:64747"
MODEL_NAME = "HuggingFaceTB/SmolLM-135M"


def test_streaming_generation(connected_nodes):
    """
    End-to-end test:
    - validator + workers running
    - HTTP streaming endpoint responds
    - tokens arrive incrementally
    """

    validator, user, worker, _ = connected_nodes

    payload = {
        "hf_name": MODEL_NAME,
        "message": "Hi.",
        "max_new_tokens": 10,
        "stream": True,
        "do_sample": False,
        "num_beams": 1,
    }

    response = requests.post(
        f"{SERVER_URL}/generate",
        json=payload,
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

        try:
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content")

            if token:
                received_tokens += 1
                full_text += token

        except json.JSONDecodeError:
            continue

    # Assertions
    assert done_received, "Streaming response never completed"
    assert received_tokens > 0, "No tokens received from stream"
    assert isinstance(full_text, str)
    assert len(full_text) > 0
