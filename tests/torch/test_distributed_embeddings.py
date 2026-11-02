"""
test_distributed_embeddings.py

Tests DistributedEmbedding on local nodes: a user, worker, and validator collaborate
to run a tiny embedding model. A User-requested job never loads on the host, so the
whole model is offloaded to the single worker and every encode() is an RPC that looks
like a local call.

Uses the same UWV fixtures as test_distributed_model.py.
"""

from tensorlink.ml.module import DistributedEmbedding, OffloadedModule
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import os
import pytest


MODEL_NAME = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
TEXTS = ["the quick brown fox", "jumps over the lazy dog"]

os.environ["TENSORLINK_EMBEDDING_MODELS"] = MODEL_NAME


def to_array(result) -> np.ndarray:
    """EmbeddingResult -> (count, dim) float32 array."""
    assert len(result.buffer) == result.count * result.dim * 4
    return np.frombuffer(result.buffer, dtype="<f4").reshape(result.count, result.dim)


@pytest.fixture(scope="module")
def embedding_env(connected_uwv_nodes):
    user, worker, validator, _ = connected_uwv_nodes

    return DistributedEmbedding(model=MODEL_NAME, node=user)


def test_embedding_is_offloaded_to_single_worker(embedding_env):
    """The whole model must sit on exactly one worker behind an OffloadedModule."""
    model = embedding_env

    assert isinstance(model.model, OffloadedModule)
    assert model.model.entire_model

    assert len(model.distributed_graph) == 1
    (module_info,) = model.distributed_graph.values()
    assert module_info["module_path"] == "model"
    assert len(module_info["assigned_workers"]) == 1


def test_embedding_inference(embedding_env):
    """Basic encode: shape, token count and L2 normalisation."""
    model = embedding_env

    result = model.encode(TEXTS)
    vectors = to_array(result)

    assert result.count == len(TEXTS)
    assert result.dim > 0
    assert result.prompt_tokens > 0
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


def test_embedding_batch_consistency(embedding_env):
    """An input embeds to the same vector alone or inside a batch."""
    model = embedding_env

    batch = to_array(model.encode(TEXTS))
    single = to_array(model.encode([TEXTS[0]]))

    np.testing.assert_allclose(batch[0], single[0], atol=1e-4)


def test_embedding_dimensions(embedding_env):
    """Matryoshka-style truncation returns unit vectors; impossible sizes raise."""
    model = embedding_env
    full = model.encode(TEXTS)

    truncated = model.encode(TEXTS, dimensions=8)
    assert truncated.dim == 8
    np.testing.assert_allclose(
        np.linalg.norm(to_array(truncated), axis=1), 1.0, atol=1e-4
    )

    with pytest.raises(ValueError):
        model.encode(TEXTS, dimensions=full.dim + 1)

    with pytest.raises(ValueError):
        model.encode(TEXTS, dimensions=0)


@pytest.mark.parametrize("bad_input", [[], [""], ["   "], ["ok", ""], "not a list"])
def test_embedding_rejects_bad_input(embedding_env, bad_input):
    with pytest.raises(ValueError):
        embedding_env.encode(bad_input)


def test_embedding_repeated_calls(embedding_env):
    """Forward keys are (n_batch, micro, module_id); each call must get a fresh one."""
    model = embedding_env
    first = to_array(model.encode(TEXTS))

    for _ in range(3):
        again = to_array(model.encode(TEXTS))
        np.testing.assert_allclose(first, again, atol=1e-4)

    # Outputs stashed for a (never-run) backward pass must not pile up.
    assert all(len(v) == 0 for v in model.model.intermediates.values())


def test_embedding_concurrent_calls(embedding_env):
    """Simultaneous callers are serialized and each gets its own result back."""
    model = embedding_env
    expected = {t: to_array(model.encode([t]))[0] for t in TEXTS}

    def run(text):
        return text, to_array(model.encode([text]))[0]

    with ThreadPoolExecutor(max_workers=4) as pool:
        for text, vec in pool.map(run, TEXTS * 3):
            np.testing.assert_allclose(vec, expected[text], atol=1e-4)


def test_embedding_matches_local_model(embedding_env):
    """Worker-hosted vectors match sentence-transformers run directly."""
    st = pytest.importorskip("sentence_transformers")

    local = st.SentenceTransformer(MODEL_NAME, device="cpu").encode(
        TEXTS, normalize_embeddings=True, convert_to_numpy=True
    )
    remote = to_array(embedding_env.encode(TEXTS))

    assert remote.shape == local.shape
    cosine = (remote * local).sum(axis=1)
    assert (cosine > 0.999).all()


def test_embedding_is_inference_only(embedding_env):
    with pytest.raises(NotImplementedError):
        embedding_env.generate(None)

    with pytest.raises(RuntimeError):
        embedding_env.train()

    embedding_env.eval()  # allowed, and must not raise


# def test_embedding_host_loaded
# def test_embedding_worker_declines_then_replacement
