"""
tensorlink/ml/utils/model_info.py

Per-model metadata that is published on a job when a model goes active and
surfaced to clients by GET /v1/models/available:

    context_length        the context window the validator actually enforces
    generation_defaults   the model's own recommended temperature / top_p

The producer (the validator's ML process) and the consumer (the node thread that
stores it) both run the same ``sanitize_*`` functions, so a malformed value can
never reach clients; it is dropped instead.

Only the standard library is imported at module level. ``huggingface_hub`` is
imported lazily, and only to read one small file.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Optional

# No real context window is anywhere near this; it filters out HF's "unset"
# sentinel (about 1e30) and garbage.
_MAX_CONTEXT_LENGTH = 10**9

GENERATION_CONFIG_FILE = "generation_config.json"


def _finite_number(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def sanitize_context_length(value) -> Optional[int]:
    """A positive, plausible token count, or None."""
    number = _finite_number(value)
    if number is None or number <= 0 or number >= _MAX_CONTEXT_LENGTH:
        return None
    return int(number)


def sanitize_generation_defaults(value) -> Optional[dict]:
    """
    Keep only valid ``temperature`` (0 < t <= 2, the range the API accepts) and
    ``top_p`` (0 < p <= 1), the two sampling knobs the API actually forwards to
    generate(). Anything else (top_k, repetition_penalty, ...) is dropped: it would
    invite settings that cannot be applied. Returns None if nothing usable is left.
    """
    if not isinstance(value, dict):
        return None

    clean = {}
    temperature = _finite_number(value.get("temperature"))
    if temperature is not None and 0 < temperature <= 2:
        clean["temperature"] = round(temperature, 3)
    top_p = _finite_number(value.get("top_p"))
    if top_p is not None and 0 < top_p <= 1:
        clean["top_p"] = round(top_p, 3)
    return clean or None


def _read_generation_config(model_name: str) -> Optional[dict]:
    """The raw generation_config.json of a model (hub repo or local directory)."""
    if os.path.isdir(model_name):
        path = os.path.join(model_name, GENERATION_CONFIG_FILE)
        if not os.path.exists(path):
            return None
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_name, GENERATION_CONFIG_FILE)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def generation_defaults(model_name: str) -> Optional[dict]:
    """
    The model author's recommended sampling settings, or None if there are none
    to advertise (no generation_config.json, nothing valid in it, or the model
    defaults to greedy decoding, in which case a temperature would mislead).

    The raw file is read, not transformers' GenerationConfig, because that object
    fills in defaults (temperature 1.0, top_p 1.0) and so cannot tell "the author
    set this" from "nobody set this".
    """
    try:
        raw = _read_generation_config(model_name)
    except Exception as e:  # noqa: BLE001 - metadata is best effort, never fatal
        logging.debug(f"No generation config for {model_name}: {type(e).__name__}: {e}")
        return None

    if not raw or raw.get("do_sample") is False:
        return None
    return sanitize_generation_defaults(raw)
