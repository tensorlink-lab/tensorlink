"""
test_embeddings_api.py

Requests an embedding model through /v1/models/request, waits for it to become
active, then exercises /v1/embeddings. Uses the same WWV fixtures as
test_model_api.py. With the default conftest caps the tiny model is small enough
to be hosted on the validator itself (no worker involved).

ASSUMED API SHAPE (the FastAPI layer wasn't in the files I had; the three
helpers under "API helpers" are the only places to adjust):
    POST /v1/models/request  {"hf_name", "model_type": "embedding", "time"}
    GET  /v1/models/status   ?model=<name> -> {"status": "active", ...}
    POST /v1/embeddings      {"model", "input": [str], "dimensions"?}
        -> {"object": "list", "data": [{"index", "embedding"}], "model",
            "usage": {"prompt_tokens", "total_tokens"}}
    errors: 400/422 bad input
"""

import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import requests

SERVER_URL = "http://127.0.0.1:64747"
MODEL_NAME = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
TIMEOUT = 90

# Must be set before the node fixtures start their processes (they inherit it).
os.environ["TENSORLINK_EMBEDDING_MODELS"] = MODEL_NAME

TEXTS = ["the quick brown fox", "jumps over the lazy dog"]


# ------------------------------------------------------------------ API helpers
def request_embedding_model(name=MODEL_NAME, timeout=TIMEOUT):
    return requests.post(
        f"{SERVER_URL}/v1/models/request",
        json={"hf_name": name, "model_type": "embedding", "time": 300},
        timeout=timeout,
    )


def get_model_status(name=MODEL_NAME):
    return requests.get(
        f"{SERVER_URL}/v1/models/status", params={"model": name}, timeout=10
    )


def embed(model_input, *, dimensions=None, model=MODEL_NAME, timeout=60):
    payload = {"model": model, "input": model_input}
    if dimensions is not None:
        payload["dimensions"] = dimensions
    return requests.post(f"{SERVER_URL}/v1/embeddings", json=payload, timeout=timeout)


def parse_embeddings(body) -> np.ndarray:
    """Response JSON -> (n, dim) float32. Accepts float lists or base64 float32."""
    rows = []
    for item in sorted(body["data"], key=lambda d: d["index"]):
        emb = item["embedding"]
        if isinstance(emb, str):
            emb = np.frombuffer(base64.b64decode(emb), dtype="<f4")
        rows.append(np.asarray(emb, dtype=np.float32))
    return np.stack(rows)


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def embedding_env(connected_wwv_nodes):
    resp = request_embedding_model()
    assert resp.status_code == 200, resp.text
    time.sleep(1)
    yield MODEL_NAME
    time.sleep(5)


@pytest.fixture(scope="module")
def active_embedding(embedding_env):
    start, status = time.time(), None
    while time.time() - start < TIMEOUT:
        r = get_model_status(embedding_env)
        assert r.status_code == 200
        status = r.json().get("status")
        if status == "active":
            yield embedding_env
            return
        time.sleep(1)
    pytest.fail(f"[{embedding_env}] not active in {TIMEOUT}s (last: {status})")


# ------------------------------------------------------------------------ tests
@pytest.mark.order(1)
def test_status_before_request(connected_wwv_nodes):
    r = get_model_status()
    assert r.status_code == 200
    assert r.json()["status"] == "inactive"


@pytest.mark.order(2)
def test_status_after_request(embedding_env):
    result = None
    for _ in range(10):  # the job takes a moment to reach the validator
        result = get_model_status(embedding_env).json()
        if result.get("status") != "inactive":
            break
        time.sleep(1)
    assert result["status"] in ("initializing", "active")


@pytest.mark.order(3)
def test_embeddings_envelope(active_embedding):
    resp = embed(TEXTS)
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["object"] == "list"
    assert body["model"] == MODEL_NAME
    assert [d["index"] for d in body["data"]] == list(range(len(TEXTS)))

    usage = body["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"]


def test_embeddings_vectors_are_unit_length(active_embedding):
    vectors = parse_embeddings(embed(TEXTS).json())

    assert vectors.shape[0] == len(TEXTS) and vectors.shape[1] > 0
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


def test_embeddings_batch_consistency(active_embedding):
    batch = parse_embeddings(embed(TEXTS).json())
    single = parse_embeddings(embed([TEXTS[0]]).json())
    np.testing.assert_allclose(batch[0], single[0], atol=1e-4)


def test_embeddings_dimensions(active_embedding):
    full = parse_embeddings(embed(TEXTS).json())

    resp = embed(TEXTS, dimensions=8)
    assert resp.status_code == 200, resp.text
    truncated = parse_embeddings(resp.json())
    assert truncated.shape == (len(TEXTS), 8)
    np.testing.assert_allclose(np.linalg.norm(truncated, axis=1), 1.0, atol=1e-4)

    for bad in (full.shape[1] + 1, 0):  # 400 from validator, 422 from schema
        assert embed(TEXTS, dimensions=bad).status_code in (400, 422)


@pytest.mark.parametrize("bad_input", [[], [""], ["   "], ["ok", ""]])
def test_embeddings_reject_bad_input(active_embedding, bad_input):
    assert embed(bad_input).status_code in (400, 422)


def test_embeddings_reject_too_many_inputs(active_embedding):
    from tensorlink.ml.embedding import MAX_INPUTS_PER_REQUEST

    resp = embed(["x"] * (MAX_INPUTS_PER_REQUEST + 1), timeout=120)
    assert resp.status_code in (400, 422)


def test_embeddings_unknown_model_is_an_error(active_embedding):
    resp = embed(TEXTS, model="sentence-transformers-testing/never-requested")
    assert resp.status_code >= 400


def test_embeddings_concurrent_calls(active_embedding):
    expected = {t: parse_embeddings(embed([t]).json())[0] for t in TEXTS}

    def run(text):
        resp = embed([text])
        assert resp.status_code == 200, resp.text
        return text, parse_embeddings(resp.json())[0]

    with ThreadPoolExecutor(max_workers=4) as pool:
        for text, vec in pool.map(run, TEXTS * 3):
            np.testing.assert_allclose(vec, expected[text], atol=1e-4)


def test_embeddings_match_local_model(active_embedding):
    st = pytest.importorskip("sentence_transformers")
    local = st.SentenceTransformer(MODEL_NAME, device="cpu").encode(
        TEXTS, normalize_embeddings=True, convert_to_numpy=True
    )
    remote = parse_embeddings(embed(TEXTS).json())

    assert remote.shape == local.shape
    assert ((remote * local).sum(axis=1) > 0.99).all()
