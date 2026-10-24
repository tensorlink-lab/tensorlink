"""
tensorlink/api/embedding.py

API-process side of POST /v1/embeddings (OpenAI-compatible).

Flow: validate -> make sure the embedding model is loaded (create a hosted job
the first time, then wait for it) -> queue an EmbeddingRequest for the ML
process -> await the result -> format. The ML process half lives in
tensorlink/ml/embedding.py and the DistributedValidator hooks.

This module deliberately does not import torch / sentence-transformers.
"""

from tensorlink.api.models import EmbeddingRequest
from tensorlink.ml.utils.model_types import ModelType
from tensorlink.ml.embedding import (
    DEFAULT_EMBEDDING_MODEL,
    MAX_INPUTS_PER_REQUEST,
    supported_embedding_models,
)
from fastapi import HTTPException
from typing import List, Optional, Union
import asyncio
import array
import base64
import os
import random
import sys
import time

# How long the first request waits for a cold model load before a 503.
LOAD_TIMEOUT = float(os.environ.get("TENSORLINK_EMBEDDING_LOAD_TIMEOUT", 60))
REQUEST_TIMEOUT = 120
# Don't create a second load job for a model whose first one is still starting.
LOAD_REQUEST_GRACE = 120
EMBEDDING_JOB_SECONDS = 1800


class EmbeddingMixin:
    def _init_embedding_state(self):
        self._embedding_locks = {}
        self._embedding_load_requested = {}

    async def _dispatch_embedding(
        self,
        model: Optional[str],
        _input: Union[str, List[str]],
        encoding_format: Optional[str] = "float",
        dimensions: Optional[int] = None,
    ) -> dict:
        model_name = model or DEFAULT_EMBEDDING_MODEL
        if model_name not in supported_embedding_models():
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' is not available for embeddings.",
            )

        texts = self._normalize_embedding_input(_input)
        if dimensions is not None and dimensions < 1:
            raise HTTPException(status_code=400, detail="dimensions must be >= 1")

        self._log_model_request(model_name)
        await self._ensure_embedding_model(model_name)

        request = EmbeddingRequest(
            hf_name=model_name,
            input=texts,
            dimensions=dimensions,
            encoding_format=encoding_format or "float",
            id=hash(f"emb_{random.random()}"),
            start_time=time.time(),
        )
        done = await self._wait_for_result(request, timeout=REQUEST_TIMEOUT)

        if getattr(done, "error", None):
            raise HTTPException(status_code=done.error_status or 500, detail=done.error)
        return self._format_embedding_response(done, model_name)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _embedding_ready(self, model_name: str) -> bool:
        _job_id, job = self._find_job_for_model(model_name)
        return bool(job and job.get("status") == "active")

    def _request_embedding_job(self, model_name: str):
        if not self._submit_model_job(
            model_name, ModelType.EMBEDDING, duration=EMBEDDING_JOB_SECONDS
        ):
            raise HTTPException(
                status_code=503,
                detail=f"Embedding model '{model_name}' request was rejected.",
                headers={"Retry-After": "15"},
            )

    async def _ensure_embedding_model(self, model_name: str):
        """Return once the model is active; load it (once) if it isn't."""
        lock = self._embedding_locks.setdefault(model_name, asyncio.Lock())
        async with lock:
            if self._embedding_ready(model_name):
                return

            _job_id, job = self._find_job_for_model(model_name)
            requested_at = self._embedding_load_requested.get(model_name)
            recently_requested = (
                requested_at is not None
                and time.monotonic() - requested_at < LOAD_REQUEST_GRACE
            )
            if job is None and not recently_requested:
                self._request_embedding_job(model_name)
                self._embedding_load_requested[model_name] = time.monotonic()

        deadline = time.monotonic() + LOAD_TIMEOUT
        while time.monotonic() < deadline:
            if self._embedding_ready(model_name):
                return
            await asyncio.sleep(0.25)

        raise HTTPException(
            status_code=503,
            detail=f"Embedding model '{model_name}' is still loading. Try again shortly.",
            headers={"Retry-After": "10"},
        )

    @staticmethod
    def _normalize_embedding_input(value) -> List[str]:
        texts = [value] if isinstance(value, str) else list(value)
        if not texts:
            raise HTTPException(status_code=400, detail="input must not be empty")
        if len(texts) > MAX_INPUTS_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"input has too many items; the limit is {MAX_INPUTS_PER_REQUEST}",
            )
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise HTTPException(
                status_code=400, detail="every input must be a non-empty string"
            )
        return texts

    @staticmethod
    def _format_embedding_response(done, model_name: str) -> dict:
        out = done.output
        buf, count, dim = out["buffer"], int(out["count"]), int(out["dim"])
        tokens = int(out.get("prompt_tokens", 0))

        data = []
        if done.encoding_format == "base64":
            step = dim * 4
            for i in range(count):
                chunk = buf[i * step : (i + 1) * step]
                data.append(
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": base64.b64encode(chunk).decode("ascii"),
                    }
                )
        else:
            values = array.array("f")
            values.frombytes(buf)
            if sys.byteorder == "big":  # buffer is little-endian float32
                values.byteswap()
            for i in range(count):
                data.append(
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": values[i * dim : (i + 1) * dim].tolist(),
                    }
                )

        return {
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }
