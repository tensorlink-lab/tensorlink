from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated, Optional, List, Literal, Union, Dict, Any


class NodeRequest(BaseModel):
    address: str


class JobRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    hf_name: str
    model_type: Optional[str] = "chat"
    time: int = 1800
    payment: int = 0


# ---------------------------------------------------------------------------
# Internal generation request — used by the ML pipeline only.
# Not exposed directly as an API request body.
# ---------------------------------------------------------------------------
class GenerationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    hf_name: str
    message: str
    cancelled: bool = False

    # Generation params (all optional)
    max_new_tokens: Optional[int] = None
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None
    num_beams: Optional[int] = None
    reasoning: Optional[bool] = False
    stream: bool = False

    prompt: str = None
    model_type: Optional[str] = "auto"

    # Chat/history
    history: Optional[List[dict]] = None

    # Output fields
    output: str = None
    formatted_response: Optional[Dict[str, Any]] = None

    # Processing metadata
    processing: bool = False
    id: int = None
    start_time: float = 0

    # Format control
    input_format: Literal["chat", "raw"] = "raw"
    output_format: Literal["raw", "openai"] = "raw"
    is_chat_completion: bool = False


# ---------------------------------------------------------------------------
# v1/chat/completions — OpenAI-compatible chat completion request
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    messages: List[ChatMessage]

    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 1024

    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0

    # Chat completions always use chat input and openai-style output
    input_format: Literal["chat", "raw"] = "chat"
    output_format: Literal["raw", "openai"] = "openai"

    user: Optional[str] = None


# ---------------------------------------------------------------------------
# v1/responses — Multi-modal response API (layout; handlers TBD)
#
# Modality is declared via `type`. Each type will have its own request/
# response pair below. Add new modalities here as the API grows.
#
# Supported types (planned):
#   "text"        — chat / text-generation  (maps to ChatCompletionRequest)
#   "image"       — text-to-image generation
#   "embedding"   — text embeddings
#   "audio"       — text-to-speech / speech-to-text  (future)
# ---------------------------------------------------------------------------
class _BaseResponseRequest(BaseModel):
    """Shared fields inherited by every modality request"""

    model_config = ConfigDict(protected_namespaces=())

    model: str
    stream: Optional[bool] = False
    user: Optional[str] = None


class TextResponseRequest(_BaseResponseRequest):
    """Text-generation variant of _BaseResponseRequest.
    Functionally equivalent to ChatCompletionRequest but under the new envelope."""

    type: Literal["text"] = "text"
    messages: List[ChatMessage]

    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 1024
    stop: Optional[Union[str, List[str]]] = None


class ImageResponseRequest(_BaseResponseRequest):
    """Text-to-image via /v1/responses. TODO: implement handler."""

    type: Literal["image"] = "image"
    prompt: str

    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    quality: Optional[str] = "standard"  # "standard" | "hd"
    response_format: Optional[Literal["url", "b64_json"]] = "url"


class EmbeddingResponseRequest(_BaseResponseRequest):
    """Text embeddings via /v1/responses. TODO: implement handler."""

    type: Literal["embedding"] = "embedding"
    input: Union[str, List[str]]

    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = None


AnyResponseRequest = Annotated[
    Union[TextResponseRequest, ImageResponseRequest, EmbeddingResponseRequest],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Shared status / info responses
# ---------------------------------------------------------------------------
class ModelStatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    status: str
    message: str
