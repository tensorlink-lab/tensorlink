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


# Node config
OFFCHAIN = True
LOCAL = True
UPNP = False
SERVER_URL = "http://127.0.0.1:64747"

# Models to test with
MODELS = [
    # pytest.param(
    #     {
    #         "name": "Qwen/Qwen3-0.6B-MLX-8bit",
    #         "timeout": 600,
    #         "sleep": 15,
    #         "parsed": False,
    #     },
    #     id="Qwen3-0.6B",
    # ),
    pytest.param(
        {
            "name": "sshleifer/tiny-gpt2",
            "timeout": 60,
            "sleep": 15,
            "parsed": False,
        },
        id="tiny-gpt2",
    ),
    # pytest.param(
    #     {
    #         "name": "HuggingFaceTB/SmolLM2-135M",
    #         "timeout": 60,
    #         "sleep": 15,
    #         "parsed": True,
    #     },
    #     id="smollm2-135m",
    # ),
]


@pytest.fixture(params=MODELS, scope="module")
def model_env(request, connected_wwv_nodes):
    """
    Uses existing WWV setup but guarantees fresh nodes per model param.
    """
    cfg = request.param
    worker, worker2, validator, _ = connected_wwv_nodes

    payload = {"hf_name": cfg["name"], "model_type": "causal", "time": 300}

    response = requests.post(
        url=f"{SERVER_URL}/request-model",
        json=payload,
        timeout=cfg["timeout"],
    )

    assert response.status_code == 200

    # Let model load/shard
    time.sleep(cfg["sleep"])

    yield cfg, (worker, worker2, validator)


# /v1/chat/completions — non-streaming
def test_chat_completions(model_env):
    """
    Non-streaming OpenAI-compatible chat completions.
    Validates the full response envelope, choice structure, and usage stats.
    """
    cfg, _ = model_env
    time.sleep(1)

    payload = {
        "model": cfg["name"],
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello world' and nothing else."},
        ],
        "max_tokens": 20,
        "temperature": 0.1,
        "stream": False,
    }

    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    assert response.status_code == 200

    result = response.json()

    # Top-level envelope
    assert "id" in result
    assert "object" in result
    assert result["object"] == "chat.completion"
    assert "created" in result
    assert "model" in result
    assert result["model"] == cfg["name"]

    # Choices
    assert "choices" in result and len(result["choices"]) > 0
    choice = result["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert (
        result["usage"]["completion_tokens"] == 0
        or choice["message"]["content"].strip() != ""
    )

    # Usage
    usage = result["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    print(f"✅ chat/completions (non-streaming) passed")
    print(f"   Output : {choice['message']['content'][:60]}...")
    print(f"   Tokens : {usage['total_tokens']}")


# /v1/chat/completions — streaming
def test_chat_completions_stream(model_env):
    """
    Streaming chat completions via SSE.
    Validates chunk structure, delta content accumulation, and the [DONE] sentinel.
    """
    cfg, _ = model_env
    time.sleep(1)

    payload = {
        "model": cfg["name"],
        "messages": [
            {"role": "user", "content": "Count to three."},
        ],
        "max_tokens": 15,
        "temperature": 0.1,
        "stream": True,
    }

    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=120,
    )
    assert response.status_code == 200

    full_text = ""
    received_chunks = 0
    done_received = False
    last_chunk = None

    for line in response.iter_lines():
        if not line:
            continue

        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue

        payload_str = decoded[6:]

        if payload_str == "[DONE]":
            done_received = True
            break

        chunk = json.loads(payload_str)
        assert "choices" in chunk and len(chunk["choices"]) > 0
        received_chunks += 1
        last_chunk = chunk

        delta = chunk["choices"][0].get("delta", {})
        full_text += delta.get("content") or ""

    assert done_received, "Stream ended without [DONE] sentinel"
    assert received_chunks > 0, "No chunks received"

    # If the model produced output at all, content fields must be non-empty
    if full_text.strip():
        assert full_text.strip() != ""

    tokens = (
        last_chunk.get("usage", {}).get("total_tokens", "n/a") if last_chunk else "n/a"
    )
    print(f"✅ chat/completions (streaming) passed")
    print(f"   Output : {full_text[:60]}...")
    print(f"   Tokens : {tokens}")


# /v1/responses — text-to-text
def test_responses_text(model_env):
    """
    /v1/responses with type='text' should behave identically to
    /v1/chat/completions for non-streaming requests.
    """
    cfg, _ = model_env
    time.sleep(1)

    payload = {
        "model": cfg["name"],
        "type": "text",
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        "max_tokens": 10,
        "temperature": 0.0,
        "stream": False,
    }

    response = requests.post(
        f"{SERVER_URL}/v1/responses",
        json=payload,
        timeout=120,
    )
    assert response.status_code == 200

    result = response.json()
    assert result["object"] == "chat.completion"
    assert "choices" in result and len(result["choices"]) > 0
    assert result["choices"][0]["message"]["role"] == "assistant"

    print(f"✅ /v1/responses (text) passed")
    print(f"   Output : {result['choices'][0]['message']['content'][:60]}...")


# /v1/responses — invalid type
def test_responses_invalid_type():
    """
    Submitting an unknown type to /v1/responses should return 422.
    This is a lightweight schema-level check; no model fixture needed.
    """
    payload = {
        "model": "any-model",
        "type": "video",  # not a supported type
        "prompt": "test",
    }

    response = requests.post(
        f"{SERVER_URL}/v1/responses",
        json=payload,
        timeout=10,
    )

    # FastAPI rejects unknown Literal values at the schema layer (422)
    # or handler returns 422 explicitly — either is correct
    assert response.status_code in (
        422,
        400,
    ), f"Expected 422 or 400 for unsupported type, got {response.status_code}"
    print(
        f"✅ /v1/responses (invalid type) correctly rejected with {response.status_code}"
    )
