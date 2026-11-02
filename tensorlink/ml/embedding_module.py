"""
tensorlink/ml/embedding_module.py

Torch-side wrapper that lets an embedding model be treated like any other
Tensorlink module: it is an ``nn.Module`` whose ``forward`` takes and returns
tensors, so ``OffloadedModule`` / the worker's ``_handle_forward`` can run it
remotely without knowing anything about embeddings.

Texts are shipped to the module as a uint8 tensor holding UTF-8 JSON. Tokenising,
pooling, prompts and normalisation all happen inside sentence-transformers on
whichever machine holds the weights, so a worker-hosted model returns the same
vectors a validator-hosted one does. ``tensorlink/ml/embedding.py`` stays
torch-free (the API process imports it); this file is only imported by the ML
processes.
"""

from __future__ import annotations

import json
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from tensorlink.ml.embedding import SentenceTransformerEmbedder


def pack_texts(texts: Sequence[str]) -> torch.Tensor:
    """Texts -> 1-D uint8 tensor (UTF-8 JSON), safe to send as a forward arg."""
    raw = json.dumps(list(texts)).encode("utf-8")
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


def unpack_texts(payload: torch.Tensor) -> List[str]:
    raw = payload.detach().cpu().numpy().astype(np.uint8).tobytes()
    return json.loads(raw.decode("utf-8"))


class EmbeddingModule(nn.Module):
    """
    forward(packed_texts) -> {"embeddings": (N, dim) float32 L2-normalised,
                              "prompt_tokens": int64 tensor of shape (1,)}
    """

    def __init__(self, embedder: SentenceTransformerEmbedder):
        super().__init__()
        self.embedder = embedder
        # Register the underlying nn.Module so .to() / .parameters() see it.
        self.st = embedder._model
        self.dim = embedder.dim
        self.memory_bytes = embedder.memory_bytes
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, packed_texts: torch.Tensor):
        texts = unpack_texts(packed_texts)
        vectors = np.ascontiguousarray(self.embedder.encode(texts), dtype=np.float32)
        tokens = int(self.embedder.count_tokens(texts))
        return {
            "embeddings": torch.from_numpy(vectors),
            "prompt_tokens": torch.tensor([tokens], dtype=torch.int64),
        }


def load_embedding_module(
    model_name: str, device: Optional[str] = None
) -> EmbeddingModule:
    """Load ``model_name`` fully onto ``device`` and wrap it."""
    return EmbeddingModule(SentenceTransformerEmbedder(model_name, device=device))
