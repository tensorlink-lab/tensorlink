from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Literal, Union, Dict, Any


class NodeRequest(BaseModel):
    address: str


class JobRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    hf_name: str
    model_type: Optional[str] = "chat"
    time: int = 1800
    payment: int = 0


class GenerationRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    # Input fields
    hf_name: str
    message: str
    prompt: str = None
    model_type: Optional[str] = "auto"

    # Generation parameters
    max_length: int = 2048
    max_new_tokens: int = 2048
    temperature: float = 0.7
    do_sample: bool = True
    num_beams: int = 1

    # Chat/history
    history: Optional[List[dict]] = None

    # Output fields
    output: str = None
    formatted_response: Optional[Dict[str, Any]] = None  # ADD THIS

    # Processing metadata
    processing: bool = False
    id: int = None
    stream: bool = False
    start_time: float = 0

    # Format control
    input_format: Literal["chat", "raw"] = "raw"
    output_format: Literal["simple", "openai", "raw"] = "simple"


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

    user: Optional[str] = None


class ModelStatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    status: str
    message: str
