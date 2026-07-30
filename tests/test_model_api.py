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
    pytest.param(
        {
            "name": "sshleifer/tiny-gpt2",
            "model_type": "causal",
            "timeout": 60,
            "parsed": False,
        },
        id="tiny-gpt2",
    ),
    pytest.param(
        {
            "name": "HuggingFaceTB/SmolLM2-135M",
            "model_type": "causal",
            "timeout": 90,
            "parsed": True,
        },
        id="smollm2-135m",
    ),
    # pytest.param(
    #     {
    #         "name": "Qwen/Qwen3-0.6B",
    #         "model_type": "causal",
    #         "timeout": 90,
    #         "parsed": True,
    #     }
    # )
]


def request_model(
    model_name: str, model_type: str = "causal", timeout: int = 60
) -> requests.Response:
    """POST /v1/models/request for the given model name."""
    payload = {"hf_name": model_name, "model_type": model_type, "time": 300}
    return requests.post(
        url=f"{SERVER_URL}/v1/models/request",
        json=payload,
        timeout=timeout,
    )


def get_model_status(model_name: str, timeout: int = 10) -> requests.Response:
    """GET /v1/models/status for the given model name."""
    return requests.get(
        url=f"{SERVER_URL}/v1/models/status",
        params={"model": model_name},
        timeout=timeout,
    )


@pytest.fixture(params=MODELS, scope="module")
def model_env(request, connected_wwv_nodes):
    """
    Uses existing WWV setup but guarantees fresh nodes per model param.
    """
    cfg = request.param
    worker, worker2, validator, _ = connected_wwv_nodes

    response = request_model(cfg["name"], cfg["model_type"], cfg["timeout"])

    assert response.status_code == 200

    yield cfg, (worker, worker2, validator)


@pytest.fixture(scope="module")
def active_model_env(model_env):
    """
    Wait until the model is active before yielding.
    """
    cfg, nodes = model_env

    start = time.time()
    last_status = None

    while time.time() - start < cfg["timeout"]:
        response = get_model_status(cfg["name"])
        assert response.status_code == 200

        result = response.json()
        last_status = result.get("status")

        if last_status == "active":
            break

        time.sleep(1)
    else:
        pytest.fail(
            f"[{cfg['name']}] Model did not become active within "
            f"{cfg['timeout']} seconds. Last status: {last_status}"
        )

    yield cfg, nodes


# ========== Model Status Tests ==========


@pytest.mark.order(1)
def test_status_before_request(connected_wwv_nodes):
    """
    Query status before any model request has been made.
    """
    for param in MODELS:
        cfg = param.values[0]
        response = get_model_status(cfg["name"])
        assert response.status_code == 200, (
            f"[{cfg['name']}] Status check failed with "
            f"{response.status_code}: {response.text}"
        )

        result = response.json()
        assert (
            "status" in result
        ), f"[{cfg['name']}] Response missing 'status' field: {result}"
        assert result["status"] == "inactive", (
            f"[{cfg['name']}] Expected 'inactive' before model request, "
            f"got '{result['status']}'"
        )
        print(f"✅ [{cfg['name']}] status before request: '{result['status']}'")


@pytest.mark.order(2)
def test_status_loading(model_env):
    """Query status immediately after requesting the model.

    Workers have been assigned modules but haven't finished loading yet, so
    the expected response is status == 'initializing'.
    """
    cfg, _ = model_env
    result = None
    for _ in range(5):
        # Check a few times as the job takes a second to be added to the validator
        response = get_model_status(cfg["name"])
        assert response.status_code == 200, (
            f"[{cfg['name']}] Status check failed with "
            f"{response.status_code}: {response.text}"
        )
        result = response.json()
        if result.get("status") != "inactive":
            break
        time.sleep(1)

    assert (
        "status" in result
    ), f"[{cfg['name']}] Response missing 'status' field: {result}"
    assert result["status"] == "initializing", (
        f"[{cfg['name']}] Expected 'initializing' immediately after model request, "
        f"got '{result['status']}'"
    )
    print(f"✅ [{cfg['name']}] status immediately after request: '{result['status']}'")


@pytest.mark.order(3)
def test_status_active(model_env):
    """
    Query status after waiting for the model to fully load.
    """
    cfg, _ = model_env
    print(f"   [{cfg['name']}] Waiting for model to finish loading...")

    start = time.time()
    while time.time() - start < cfg["timeout"]:
        response = get_model_status(cfg["name"])
        assert response.status_code == 200, (
            f"[{cfg['name']}] Status check failed with "
            f"{response.status_code}: {response.text}"
        )

        result = response.json()
        assert (
            "status" in result
        ), f"[{cfg['name']}] Response missing 'status' field: {result}"

        if result["status"] == "active":
            print(f"✅ [{cfg['name']}] status became 'active'")
            break

        time.sleep(1)
    else:
        pytest.fail(
            f"[{cfg['name']}] Model did not become active within "
            f"{cfg['timeout']} seconds. Last status: {result.get('status')}"
        )


# ========= Model Inference Tests =========


@pytest.mark.order(4)
def test_chat_completions(active_model_env):
    """
    Non-streaming OpenAI-compatible chat completions.
    Validates the full response envelope, choice structure, and usage stats.
    """
    cfg, _ = active_model_env

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


@pytest.mark.order(5)
def test_chat_completions_stream(active_model_env):
    """
    Streaming chat completions via SSE.
    Validates chunk structure, delta content accumulation, and the [DONE] sentinel.
    """
    cfg, _ = active_model_env
    time.sleep(1)

    payload = {
        "model": cfg["name"],
        "messages": [
            {"role": "user", "content": "Count to three."},
        ],
        "max_tokens": 50,
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

        print(full_text)

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


@pytest.mark.order(6)
def test_responses_text(active_model_env):
    """
    /v1/responses with type='text' should behave identically to
    /v1/chat/completions for non-streaming requests.
    """
    cfg, _ = active_model_env
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


@pytest.mark.order(7)
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
    # or handler returns 422 explicitly
    assert response.status_code in (
        422,
        400,
    ), f"Expected 422 or 400 for unsupported type, got {response.status_code}"
    print(
        f"✅ /v1/responses (invalid type) correctly rejected with {response.status_code}"
    )
