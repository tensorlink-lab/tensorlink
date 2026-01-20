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


@pytest.fixture(params=MODELS, scope="module")
def model_env(request, connected_wwv_nodes):
    """
    Uses existing WWV setup but guarantees fresh nodes per model param.
    """
    cfg = request.param
    worker, worker2, validator, _ = connected_wwv_nodes

    payload = {
        "hf_name": cfg["name"],
        "model_type": "causal",
    }

    response = requests.post(
        url=f"{SERVER_URL}/request-model",
        json=payload,
        timeout=cfg["timeout"],
    )

    assert response.status_code == 200

    # Let model load/shard
    time.sleep(cfg["sleep"])

    yield cfg, (worker, worker2, validator)


def test_generate_simple(model_env):
    """
    Test generate request with simple format (default).
    Validates structured response with metadata and usage stats.
    """
    cfg, (worker, worker2, validator) = model_env
    time.sleep(1)

    generate_payload = {
        "hf_name": cfg["name"],
        "message": "Hi.",
        "max_new_tokens": 10,
        "do_sample": True,
        "num_beams": 2,
        "output_format": "simple",  # Explicitly set to simple
    }

    response = requests.post(
        f"{SERVER_URL}/v1/generate",
        json=generate_payload,
        timeout=100,
    )

    assert response.status_code == 200

    result = response.json()

    # Validate simple format structure
    assert "id" in result, "Response missing 'id'"
    assert "model" in result, "Response missing 'model'"
    assert "text" in result, "Response missing 'text'"
    assert "usage" in result, "Response missing 'usage'"
    assert "processing_time" in result, "Response missing 'processing_time'"
    assert "finish_reason" in result, "Response missing 'finish_reason'"

    # Validate model matches request
    assert result["model"] == cfg["name"]

    # Validate usage statistics
    usage = result["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage
    assert usage["prompt_tokens"] > 0, "prompt_tokens should be > 0"
    assert usage["completion_tokens"] > 0, "completion_tokens should be > 0"
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    # Validate text output
    assert isinstance(result["text"], str)
    assert len(result["text"]) > 0, "Generated text should not be empty"

    # Validate processing time
    assert isinstance(result["processing_time"], (int, float))
    assert result["processing_time"] >= 0

    # Validate finish reason
    assert result["finish_reason"] == "stop"

    print(f"✅ Simple format test passed")
    print(f"   Generated: {result['text'][:50]}...")
    print(f"   Usage: {usage}")
    print(f"   Processing time: {result['processing_time']:.2f}s")


def test_generate_openai(model_env):
    """
    Test generate request with simple format (default).
    Validates structured response with metadata and usage stats.
    """
    cfg, (worker, worker2, validator) = model_env
    time.sleep(1)

    generate_payload = {
        "hf_name": cfg["name"],
        "message": "Hi.",
        "max_new_tokens": 10,
        "do_sample": True,
        "num_beams": 2,
        "output_format": "openai",
    }

    response = requests.post(
        f"{SERVER_URL}/v1/generate",
        json=generate_payload,
        timeout=100,
    )

    assert response.status_code == 200

    result = response.json()

    # Validate OpenAI format structure
    assert "id" in result, "Response missing 'id'"
    assert "object" in result, "Response missing 'object'"
    assert "created" in result, "Response missing 'created'"
    assert "model" in result, "Response missing 'model'"
    assert "choices" in result, "Response missing 'choices'"
    assert "usage" in result, "Response missing 'usage'"

    # Validate object type
    assert result["object"] == "chat.completion"

    # Validate model
    assert result["model"] == cfg["name"]

    # Validate choices
    assert len(result["choices"]) > 0
    choice = result["choices"][0]
    assert "index" in choice
    assert "message" in choice
    assert "finish_reason" in choice

    # Validate message structure
    message = choice["message"]
    assert "role" in message
    assert "content" in message
    assert message["role"] == "assistant"
    assert isinstance(message["content"], str)
    assert result["usage"]["completion_tokens"] == 0 or message["content"].strip() != ""

    # Validate usage
    usage = result["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    print(f"✅ OpenAI format test passed")
    print(f"   Generated: {message['content'][:50]}...")
    print(f"   Usage: {usage}")


def test_streaming_generation_openai(model_env):
    """
    Test generate request with token-by-token streaming via API
    """
    cfg, (worker, worker2, validator) = model_env
    time.sleep(1)

    generate_payload = {
        "hf_name": cfg["name"],
        "message": "Hi.",
        "max_new_tokens": 10,
        "stream": True,
        "do_sample": False,
        "num_beams": 1,
        "output_format": "openai",
    }

    response = requests.post(
        f"{SERVER_URL}/v1/generate",
        json=generate_payload,
        stream=True,
        timeout=120,
    )

    assert response.status_code == 200

    full_text = ""
    received_chunks = 0
    received_content_fields = 0
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

        assert "choices" in chunk
        assert len(chunk["choices"]) > 0

        received_chunks += 1

        delta = chunk["choices"][0].get("delta", {})

        if "content" in delta:
            received_content_fields += 1
            full_text += delta.get("content") or ""

    assert done_received
    assert received_chunks > 0

    # content is optional if model stops immediately
    if full_text.strip() != "":
        assert received_content_fields > 0


def test_streaming_generation_simple(model_env):
    """
    Test generate request with token-by-token streaming via API (simple format)
    """
    cfg, (worker, worker2, validator) = model_env
    time.sleep(1)

    generate_payload = {
        "hf_name": cfg["name"],
        "message": "Hi.",
        "max_new_tokens": 10,
        "stream": True,
        "do_sample": False,
        "num_beams": 1,
        "output_format": "simple",
    }

    response = requests.post(
        f"{SERVER_URL}/v1/generate",
        json=generate_payload,
        stream=True,
        timeout=120,
    )

    assert response.status_code == 200

    full_text = ""
    received_chunks = 0
    received_token_fields = 0
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

            received_chunks += 1

            if chunk.get("done") is True:
                full_text = chunk.get("full_text", full_text)
                done_received = True
                break
            else:
                if "token" in chunk:
                    received_token_fields += 1
                    full_text += chunk.get("token") or ""

        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {data}")
            raise

    assert done_received, "Never received done=true or [DONE] marker"
    assert received_chunks > 0, "No stream chunks received"
    assert isinstance(full_text, str)

    print(f"Received {received_chunks} tokens")
    print(f"Full text: {full_text}")


def test_chat_completions(model_env):
    """
    Test OpenAI-compatible chat completions endpoint (non-streaming)
    """
    cfg, (worker, worker2, validator) = model_env
    time.sleep(1)

    chat_payload = {
        "model": cfg["name"],
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello world' and nothing else."},
        ],
        "max_tokens": 10,
        "temperature": 0.7,
        "stream": False,
    }

    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=chat_payload,
        timeout=120,
    )

    assert response.status_code == 200

    result = response.json()

    # Validate OpenAI response format
    assert "id" in result
    assert "object" in result
    assert result["object"] == "chat.completion"
    assert "created" in result
    assert "model" in result
    assert result["model"] == cfg["name"]
    assert "choices" in result
    assert len(result["choices"]) > 0

    # Check the first choice
    choice = result["choices"][0]
    assert "index" in choice
    assert choice["index"] == 0
    assert "message" in choice
    assert "role" in choice["message"]
    assert choice["message"]["role"] == "assistant"
    assert "content" in choice["message"]
    assert (
        result["usage"]["completion_tokens"] == 0
        or choice["message"]["content"].strip() != ""
    )

    # Check usage stats
    assert "usage" in result
    assert "prompt_tokens" in result["usage"]
    assert "completion_tokens" in result["usage"]
    assert "total_tokens" in result["usage"]
    assert result["usage"]["total_tokens"] == (
        result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"]
    )
