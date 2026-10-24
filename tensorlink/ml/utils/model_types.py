"""
tensorlink/ml/model_types.py

The one list of model types a job can be requested as. Standard library only, so
the API process and the ML processes can both import it.

``ModelType`` is a ``str`` enum: members compare equal to their plain-string
values (``ModelType.EMBEDDING == "embedding"``), so existing checks such as
``model_type in ("causal", "chat")`` keep working, and pydantic rejects any other
value with a 422 before a handler runs.

Add a member only once ``ml/utils/loading.py`` (and the validator) know how to
load it; anything unrecognised there falls through to a bare ``AutoModel``.
"""

from enum import Enum


class ModelType(str, Enum):
    CHAT = "chat"
    CAUSAL = "causal"
    SEQ2SEQ = "seq2seq"
    AUDIO2TEXT = "audio2text"
    EMBEDDING = "embedding"

    def __str__(self) -> str:  # f"{ModelType.CHAT}" -> "chat", not "ModelType.CHAT"
        return self.value
