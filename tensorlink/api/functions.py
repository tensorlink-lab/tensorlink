"""Stateless helpers for the Tensorlink API: job payloads and request translation."""

from tensorlink.api.models import GenerationRequest
from tensorlink.ml.utils.model_types import ModelType
from fastapi import HTTPException


def build_hf_job_data(
    *,
    model_name: str,
    author: str,
    model_type: ModelType = ModelType.CHAT,
    payment: int = 0,
    duration: int = 0,
    hosted: bool = True,
    training: bool = False,
    seed_validators=None,
):
    if seed_validators is None:
        seed_validators = [author]

    return {
        "author": author,
        "api": True,
        "active": True,
        "hosted": hosted,
        "training": training,
        "payment": payment,
        "time": duration,
        "capacity": 0,
        "n_pipelines": 1,
        "dp_factor": 1,
        "distribution": {"model_name": model_name},
        "n_workers": 0,
        "model_name": model_name,
        "seed_validators": seed_validators,
        "model_type": ModelType(model_type).value,
    }


def parse_chat_messages(messages):
    """
    Parse chat messages into system messages, history, and last user message.
    Returns: (system_messages, history, last_user_message)
    """
    system_messages = []
    conversation = []

    for msg in messages:
        if msg.role not in ("system", "user", "assistant"):
            continue

        if msg.role == "system":
            system_messages.append(msg.content)
        else:
            conversation.append({"role": msg.role, "content": msg.content})

    # Find last user message
    last_user_message = None
    last_user_idx = None

    for idx in range(len(conversation) - 1, -1, -1):
        if conversation[idx]["role"] == "user":
            last_user_message = conversation[idx]["content"]
            last_user_idx = idx
            break

    if last_user_message is None:
        raise HTTPException(status_code=400, detail="No user message found")

    # Build history (everything before the last user message)
    history = conversation[:last_user_idx]

    # Prepend system message to history if present
    if system_messages:
        combined_system = "\n".join(system_messages)
        history.insert(0, {"role": "system", "content": combined_system})

    return system_messages, history, last_user_message


def build_generation_request(request) -> GenerationRequest:
    """Shared factory: ChatCompletionRequest or TextResponseRequest → GenerationRequest."""
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")
    _, history, last_user_message = parse_chat_messages(request.messages)
    return GenerationRequest(
        hf_name=request.model,
        message=last_user_message,
        history=history,
        temperature=request.temperature,
        top_p=request.top_p,
        max_new_tokens=request.max_tokens,
        stream=request.stream,
        input_format="chat",
        output_format="openai",
        do_sample=(request.temperature or 0) > 0,
        is_chat_completion=True,
    )
