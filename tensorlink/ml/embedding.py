"""
tensorlink/ml/embedding.py

Embedding-model lifecycle for hosted validators (runs in the ML process).

  * Models are loaded on the validator host the first time they are requested
    and kept warm; ``EmbeddingManager.reap()`` unloads any that have been idle
    longer than ``idle_ttl``. Nothing is sharded across workers.
  * Vectors come back L2-normalised, optionally truncated to ``dimensions``
    (Matryoshka-style) and re-normalised.
  * Only the standard library is imported at module level so the API process
    can import the constants below without pulling in numpy/torch.
    sentence-transformers is imported lazily and is an optional dependency:
        pip install sentence-transformers      (``tensorlink[embeddings]``)

The manager takes an injectable ``loader`` and clock so its behaviour (lazy
load, idle eviction, validation) can be unit-tested without any model weights.
"""

from __future__ import annotations

import gc
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "TENSORLINK_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"
)

# Models a public validator is willing to load for /v1/embeddings. Extend via
# TENSORLINK_EMBEDDING_MODELS="org/model-a,org/model-b".
_BUILTIN_SUPPORTED = (
    "Qwen/Qwen3-Embedding-0.6B",
    "BAAI/bge-m3",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
)

DEFAULT_IDLE_SECONDS = 30 * 60
MAX_INPUTS_PER_REQUEST = 2048


def supported_embedding_models() -> set:
    extra = os.environ.get("TENSORLINK_EMBEDDING_MODELS", "")
    models = set(_BUILTIN_SUPPORTED) | {DEFAULT_EMBEDDING_MODEL}
    models |= {m.strip() for m in extra.split(",") if m.strip()}
    return models


def idle_seconds_from_env() -> float:
    try:
        return float(os.environ.get("TENSORLINK_EMBEDDING_IDLE_SECONDS", ""))
    except ValueError:
        return DEFAULT_IDLE_SECONDS


class EmbeddingUnavailable(RuntimeError):
    """The embedding backend (or the requested job's model) isn't available."""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
@dataclass
class EmbeddingResult:
    """Vectors as one little-endian float32 buffer, shape (count, dim)."""

    buffer: bytes
    count: int
    dim: int
    prompt_tokens: int


def finalize_vectors(vectors, dimensions: Optional[int] = None):
    """
    Return float32 vectors of shape (n, d). If ``dimensions`` is smaller than
    the native size, truncate and re-normalize to unit length. Raises
    ValueError for an impossible ``dimensions``.
    """
    import numpy as np

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {arr.shape}")

    native = arr.shape[1]
    if dimensions is not None:
        if dimensions < 1 or dimensions > native:
            raise ValueError(
                f"dimensions must be between 1 and {native} for this model "
                f"(got {dimensions})"
            )
        if dimensions < native:
            arr = arr[:, :dimensions]
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.maximum(norms, 1e-12)
    return np.ascontiguousarray(arr, dtype="<f4")


def validate_inputs(texts: Sequence[str]) -> List[str]:
    if not isinstance(texts, (list, tuple)) or not texts:
        raise ValueError("input must be a non-empty list of strings")
    if len(texts) > MAX_INPUTS_PER_REQUEST:
        raise ValueError(
            f"too many inputs ({len(texts)}); the limit is {MAX_INPUTS_PER_REQUEST}"
        )
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("every input must be a non-empty string")
    return list(texts)


# ---------------------------------------------------------------------------
# sentence-transformers backend
# ---------------------------------------------------------------------------
class SentenceTransformerEmbedder:
    """
    Thin adapter over sentence-transformers. The library reads each model's
    pooling / normalisation / prompt configuration from its Hugging Face repo,
    so one adapter covers Qwen3-Embedding, BGE, MiniLM, etc.
    """

    def __init__(
        self, model_name: str, device: Optional[str] = None, batch_size: int = 32
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise EmbeddingUnavailable(
                "sentence-transformers is not installed on this validator "
                "(pip install sentence-transformers)"
            ) from e

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device or None)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.memory_bytes = int(
            sum(p.numel() * p.element_size() for p in self._model.parameters())
        )

    def encode(self, texts: Sequence[str]):
        return self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def count_tokens(self, texts: Sequence[str]) -> int:
        try:
            enc = self._model.tokenizer(
                list(texts),
                add_special_tokens=True,
                truncation=True,
                max_length=getattr(self._model, "max_seq_length", None),
            )
            return int(sum(len(ids) for ids in enc["input_ids"]))
        except Exception:
            return int(sum(len(t.split()) for t in texts))  # rough fallback

    def close(self):
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Lifecycle manager
# ---------------------------------------------------------------------------
@dataclass
class _Entry:
    model_name: str
    embedder: object
    last_used: float


class EmbeddingManager:
    """
    Owns loaded embedding models, keyed by job id.

    load()   blocking; returns the model's memory footprint in bytes
    embed()  validates, encodes, truncates + normalises; refreshes last-used
    reap()   unloads models idle longer than ``idle_ttl``; returns
             [(job_id, model_name), ...] so the caller can retire the jobs
    """

    def __init__(
        self,
        loader: Optional[Callable[[str], object]] = None,
        idle_ttl: Optional[float] = None,
        time_fn: Callable[[], float] = time.monotonic,
        device: Optional[str] = None,
    ):
        self._loader = loader or (
            lambda name: SentenceTransformerEmbedder(name, device=device)
        )
        self.idle_ttl = idle_seconds_from_env() if idle_ttl is None else idle_ttl
        self._time = time_fn
        self._entries: Dict[str, _Entry] = {}
        self._lock = threading.RLock()

    # -- introspection ------------------------------------------------------
    def has(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._entries

    def model_name(self, job_id: str) -> Optional[str]:
        with self._lock:
            entry = self._entries.get(job_id)
            return entry.model_name if entry else None

    def job_ids(self) -> List[str]:
        with self._lock:
            return list(self._entries)

    # -- lifecycle ------------------------------------------------------------
    def load(self, job_id: str, model_name: str) -> int:
        with self._lock:
            existing = self._entries.get(job_id)
            if existing:
                existing.last_used = self._time()
                return int(getattr(existing.embedder, "memory_bytes", 0))

        embedder = self._loader(model_name)  # slow; don't hold the lock
        with self._lock:
            self._entries[job_id] = _Entry(model_name, embedder, self._time())
        return int(getattr(embedder, "memory_bytes", 0))

    def unload(self, job_id: str) -> Optional[str]:
        with self._lock:
            entry = self._entries.pop(job_id, None)
        if entry is None:
            return None
        close = getattr(entry.embedder, "close", None)
        if callable(close):
            close()
        return entry.model_name

    def reap(self, now: Optional[float] = None) -> List[Tuple[str, str]]:
        now = self._time() if now is None else now
        with self._lock:
            idle = [
                (job_id, e.model_name)
                for job_id, e in self._entries.items()
                if now - e.last_used > self.idle_ttl
            ]
        for job_id, _ in idle:
            self.unload(job_id)
        return idle

    # -- inference ------------------------------------------------------------
    def embed(
        self, job_id: str, texts: Sequence[str], dimensions: Optional[int] = None
    ) -> EmbeddingResult:
        texts = validate_inputs(texts)
        with self._lock:
            entry = self._entries.get(job_id)
            if entry is None:
                raise EmbeddingUnavailable(
                    f"no embedding model loaded for job {job_id}"
                )
            embedder = entry.embedder

        vectors = finalize_vectors(embedder.encode(texts), dimensions)
        if vectors.shape[0] != len(texts):
            raise RuntimeError(
                f"embedder returned {vectors.shape[0]} vectors for {len(texts)} inputs"
            )
        tokens = int(embedder.count_tokens(texts))

        with self._lock:
            if job_id in self._entries:
                self._entries[job_id].last_used = self._time()

        return EmbeddingResult(
            buffer=vectors.tobytes(),
            count=int(vectors.shape[0]),
            dim=int(vectors.shape[1]),
            prompt_tokens=tokens,
        )
