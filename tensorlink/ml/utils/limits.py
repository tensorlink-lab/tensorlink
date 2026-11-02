"""
tensorlink/ml/utils/limits.py

Token-limit helpers shared by the validator and ``normalize_generate_args``.

Design, in one paragraph: a model's *context window* is read from the tokenizer
(``tokenizer.model_max_length``), which every Hugging Face tokenizer exposes.
Chat history is never trimmed and prompts are never silently truncated: if a
prompt does not fit, the request fails with a `ContextLengthError` that says
so. The generation budget is always bounded by what is left of the window.

Only the standard library is imported at module level; `transformers` is
imported lazily and only to look up a model config.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

# Hugging Face uses int(1e30) as "no limit was set" for tokenizers whose
# model_max_length is unknown. No real context window is anywhere near this.
UNSET_LIMIT = 10**9

# Used only when neither the tokenizer nor the model config reports a limit.
DEFAULT_CONTEXT_WINDOW = 4096

# Reply budget when the client does not ask for one.
DEFAULT_MAX_NEW_TOKENS = 1024

# A prompt must leave at least this much of the window for the reply.
MIN_GENERATION_TOKENS = 10

# Optional operator-side ceiling on the window (e.g. to match the sequence
# length the node reserved memory for).
MAX_CONTEXT_ENV = "TENSORLINK_MAX_CONTEXT_TOKENS"

_CONFIG_FIELDS = (
    "max_position_embeddings",
    "n_positions",
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
)


class ContextLengthError(ValueError):
    """The prompt does not fit in the model's context window."""


class InvalidGenerationParameter(ValueError):
    """A generation parameter is out of range."""


def request_error_fields(exc: Exception) -> Tuple[str, int]:
    """(error_type, http_status) to report for a rejected request."""
    if isinstance(exc, ContextLengthError):
        return "context_length_exceeded", 400
    return "invalid_request_error", 400


@dataclass(frozen=True)
class ContextWindow:
    tokens: int
    source: str

    def describe(self) -> str:
        return f"{self.tokens} tokens ({self.source})"


def _valid_limit(value) -> Optional[int]:
    """A usable token limit, or None for missing/sentinel/nonsense values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value <= 0 or value >= UNSET_LIMIT:  # NaN, <=0, sentinel
        return None
    return int(value)


def operator_cap_from_env() -> Optional[int]:
    return _valid_limit(_to_number(os.environ.get(MAX_CONTEXT_ENV, "")))


def _to_number(text: str):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def config_context_limit(model_name: str) -> Optional[int]:
    """
    The positional capacity declared by the model's own config, or None if it
    cannot be read (offline, remote-code config, unusual architecture).
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name)
    except Exception as e:
        logging.debug(f"No model config for {model_name}: {type(e).__name__}: {e}")
        return None

    for holder in (config, getattr(config, "text_config", None)):
        if holder is None:
            continue
        for name in _CONFIG_FIELDS:
            limit = _valid_limit(getattr(holder, name, None))
            if limit:
                return limit
    return None


def resolve_context_window(
    tokenizer,
    config_limit: Optional[int] = None,
    operator_cap: Optional[int] = None,
) -> ContextWindow:
    """Context window for a model, from `tokenizer.model_max_length`"""
    tok = _valid_limit(getattr(tokenizer, "model_max_length", None))
    cfg = _valid_limit(config_limit)

    if tok and cfg:
        tokens = min(tok, cfg)
        source = (
            "tokenizer and model config agree"
            if tok == cfg
            else f"tokenizer says {tok}, model config says {cfg}; using the smaller"
        )
    elif tok:
        tokens, source = tok, "tokenizer.model_max_length"
    elif cfg:
        tokens, source = cfg, "model config (the tokenizer reports no limit)"
    else:
        tokens = DEFAULT_CONTEXT_WINDOW
        source = "default: neither tokenizer nor model config reports a limit"

    cap = _valid_limit(operator_cap)
    if cap and cap < tokens:
        tokens, source = cap, f"{source}; capped by {MAX_CONTEXT_ENV}={cap}"

    return ContextWindow(tokens, source)
