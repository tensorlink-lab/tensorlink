from pydantic import BaseModel, ConfigDict, Field
from typing import Annotated, Optional, List, Literal, Union, Dict, Any

from tensorlink.ml.utils.model_types import ModelType


class NodeRequest(BaseModel):
    address: str


class JobRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    hf_name: str
    model_type: ModelType = ModelType.CHAT
    time: int = 1800
    payment: int = 0


class GenerationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    hf_name: str
    message: str
    cancelled: bool = False

    # Generation params (all optional)
    max_new_tokens: Optional[int] = None
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
    max_tokens: Optional[int] = Field(default=1024, ge=1)
    # Newer OpenAI SDKs send this instead of max_tokens; it wins when both are set.
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    reasoning: Optional[bool] = False

    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0

    # Chat completions always use chat input and openai-style output
    input_format: Literal["chat", "raw"] = "chat"
    output_format: Literal["raw", "openai"] = "openai"

    user: Optional[str] = None


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
    max_tokens: Optional[int] = Field(default=1024, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    stop: Optional[Union[str, List[str]]] = None
    reasoning: Optional[bool] = False


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


class EmbeddingCreateRequest(BaseModel):
    """OpenAI-compatible body for POST /v1/embeddings."""

    model_config = ConfigDict(protected_namespaces=())

    # Optional here (OpenAI requires it): falls back to the network default.
    model: Optional[str] = None
    input: Union[str, List[str]]
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = Field(default=None, ge=1)
    user: Optional[str] = None


class EmbeddingRequest(BaseModel):
    """Internal request queued from the API process to the ML process.
    Not part of the public API. ``output`` / ``error`` are filled in by the
    ML process before it is returned."""

    model_config = ConfigDict(protected_namespaces=())

    hf_name: str
    input: List[str]
    dimensions: Optional[int] = None
    encoding_format: Literal["float", "base64"] = "float"

    cancelled: bool = False
    processing: bool = False
    stream: bool = False  # always False; the shared request plumbing reads it
    id: int = None
    start_time: float = 0

    output: Optional[Dict[str, Any]] = None  # {buffer, count, dim, prompt_tokens}
    error: Optional[str] = None
    error_status: Optional[int] = None


AnyResponseRequest = Annotated[
    Union[TextResponseRequest, ImageResponseRequest, EmbeddingResponseRequest],
    Field(discriminator="type"),
]


class ModelStatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    status: str
    message: str


class ModelDistributionEntry(BaseModel):
    """Describes a single module's distribution state."""

    model_config = ConfigDict(protected_namespaces=())

    worker_id: Optional[str] = None  # null for validator-hosted modules
    loaded: bool = False
    type: str = ""
    memory: int = 0


class ModelStatusDistributionResponse(BaseModel):
    """Extended model status response including per-module distribution."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    status: str  # "active" | "initializing" | "inactive"
    message: str
    distribution: Optional[Dict[str, ModelDistributionEntry]] = None
