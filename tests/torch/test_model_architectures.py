"""
test_model_architectures.py

Smoke-tests a range of model architectures through the node API on local nodes.
Each model is requested, waited on until active, and then hit with a single
/v1/chat/completions call to make sure that architecture works end to end.

Two node topologies are used:
  - Worker-Worker-Validator (default): small models fit on one worker.
  - Worker-Validator (`single_worker: True`): the larger model is divided between
    one worker and the validator's hosted modules, by raising the validator's memory
    and max module size instead of adding a second worker.

Validator hosting is controlled per param:
  - `hosting` (default True): the validator may host modules itself. With False it
    offers no host memory, so the worker(s) must hold the whole model.
  - `host_fill` (default None = config default, off): when hosting is on, the validator
    fills the front/back of the model up to its memory budget. Without it the validator
    only hosts what a job explicitly asks it to (nothing, for these tests), so a model
    that does not fit on the workers is rejected.
"""

import requests
import pytest
import time


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
            "name": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "model_type": "causal",
            "timeout": 90,
            "parsed": True,
        },
        id="tiny-random-llama",
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
    pytest.param(
        # Single worker case: worker memory too small for the whole model,
        # validator takes the rest.
        {
            "name": "Qwen/Qwen3-0.6B",
            "model_type": "causal",
            "timeout": 180,
            "parsed": True,
            "single_worker": True,
            "worker_memory_gb": 1.1,
            "validator_memory_gb": 1.8,
            "validator_module_gb": 1.6,
            "hosting": True,
            "host_fill": True,
        },
        id="qwen3-0.6b-1-worker+host",
    ),
    pytest.param(
        # Two workers: workers host entire model, validator takes nothing
        {
            "name": "Qwen/Qwen3-0.6B",
            "model_type": "causal",
            "timeout": 180,
            "parsed": True,
            "worker_memory_gb": 1.3,
        },
        id="qwen3-0.6b-2-workers",
    ),
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


@pytest.fixture(scope="module", params=MODELS)
def model_cfg(request):
    return request.param


@pytest.fixture(scope="module")
def worker_memory_gb(model_cfg, max_memory_gb):
    return model_cfg.get("worker_memory_gb", max_memory_gb)


@pytest.fixture(scope="module")
def validator_memory_gb(model_cfg, max_memory_gb):
    return model_cfg.get("validator_memory_gb", max_memory_gb)


@pytest.fixture(scope="module")
def validator_module_gb(model_cfg, max_module_gb):
    return model_cfg.get("validator_module_gb", max_module_gb)


@pytest.fixture(scope="module")
def enable_hosting(model_cfg):
    """Whether the validator may host modules itself (default True)."""
    return model_cfg.get("hosting", True)


@pytest.fixture(scope="module")
def host_fill(model_cfg):
    """Validator host fill for this param (None = config default, which is off)."""
    return model_cfg.get("host_fill")


@pytest.fixture(scope="module")
def nodes(model_cfg, request):
    """
    Pick the node topology for this model: Worker-Validator for models flagged
    `single_worker`, otherwise Worker-Worker-Validator.
    """
    fixture = (
        "connected_wv_nodes"
        if model_cfg.get("single_worker")
        else "connected_wwv_nodes"
    )
    return request.getfixturevalue(fixture)


@pytest.fixture(scope="module")
def model_env(model_cfg, nodes):
    """
    Requests the model on nodes built for this param.
    """
    cfg = model_cfg

    response = request_model(cfg["name"], cfg["model_type"], cfg["timeout"])
    assert response.status_code == 200
    time.sleep(1)

    yield cfg

    print(f"Waiting before switching away from {cfg['name']}...")
    time.sleep(3)


@pytest.fixture(scope="module")
def active_model_env(model_env):
    """
    Wait until the model is active before yielding.
    """
    cfg = model_env

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

    yield cfg


def test_chat_completion(active_model_env):
    """
    One non-streaming OpenAI-compatible chat completion per architecture.
    """
    cfg = active_model_env

    payload = {
        "model": cfg["name"],
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello world' and nothing else."},
        ],
        "max_tokens": 25,
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
    assert result["object"] == "chat.completion"
    assert result["model"] == cfg["name"]

    choice = result["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)

    usage = result["usage"]
    assert usage["completion_tokens"] == 0 or choice["message"]["content"].strip() != ""
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    print(f"✅ [{cfg['name']}] chat/completions passed")
    print(f"   Output : {choice['message']['content'][:60]}...")
    print(f"   Tokens : {usage['total_tokens']}")
