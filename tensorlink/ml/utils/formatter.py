from typing import Dict, Any, Optional
import json

import logging
import time
import re

from tensorlink.ml.utils.limits import (
    DEFAULT_MAX_NEW_TOKENS,
    MIN_GENERATION_TOKENS,
    ContextLengthError,
    InvalidGenerationParameter,
    resolve_context_window,
)
from tensorlink.ml.utils.reasoning import REASONING_FIELD


def _fit_generation_budget(
    requested: Optional[int], prompt_tokens: Optional[int], window: int
) -> int:
    """
    The model response budget (max_new_tokens), always bounded.
    """
    wanted = requested if requested is not None else DEFAULT_MAX_NEW_TOKENS
    if prompt_tokens is None:
        return wanted

    available = window - prompt_tokens
    if available < MIN_GENERATION_TOKENS:
        raise ContextLengthError(
            f"This model's maximum context length is {window} tokens, but the "
            f"prompt alone is {prompt_tokens} tokens, leaving {max(available, 0)} "
            f"for the reply. Please shorten the messages."
        )
    if wanted > available:
        logging.info(
            f"Reduced max_new_tokens from {wanted} to {available} to fit the "
            f"{window}-token context window"
        )
        return available
    return wanted


def _infer_do_sample(temperature, top_p) -> Optional[bool]:
    """
    OpenAI convention: temperature 0 means greedy decoding, a positive temperature
    (or a top_p below 1) means sample. None means the client expressed no preference,
    so leave the model's default.
    """
    if temperature is not None:
        return temperature > 0
    if top_p is not None and top_p < 1.0:
        return True
    return None


def normalize_generate_args(
    request,
    tokenizer,
    prompt_tokens: Optional[int] = None,
    model_max_length: Optional[int] = None,
    allowed_generate_args: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Normalize and validate generation arguments.

    Sampling arguments are passed through only when the client provided them.
    The reply budget (max_new_tokens) is the exception: it is always present and
    always bounded by the model's context window (see _fit_generation_budget).
    ``model_max_length`` should be the resolved context window; if omitted it is
    derived from the tokenizer.
    """
    # Token IDs
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)

    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0
    if eos_token_id is None:
        eos_token_id = pad_token_id

    if pad_token_id == eos_token_id:
        if pad_token_id == 0 and vocab_size > 1:
            eos_token_id = 1
        elif pad_token_id > 0:
            eos_token_id = 0
        eos_token_id = min(eos_token_id, vocab_size - 1)

    if model_max_length is None:
        model_max_length = resolve_context_window(tokenizer).tokens

    args: Dict[str, Any] = {
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }

    max_new_tokens = getattr(request, "max_new_tokens", None)
    if max_new_tokens is not None and max_new_tokens < 1:
        raise InvalidGenerationParameter("max_new_tokens must be >= 1")

    args["max_new_tokens"] = _fit_generation_budget(
        max_new_tokens, prompt_tokens, model_max_length
    )

    # temp
    temperature = getattr(request, "temperature", None)
    top_p = getattr(request, "top_p", None)

    # do sample
    do_sample = getattr(request, "do_sample", None)
    if do_sample is None:
        do_sample = _infer_do_sample(temperature, top_p)
    if do_sample is not None:
        do_sample = bool(do_sample)
        args["do_sample"] = do_sample

    # temperature
    if temperature is not None and do_sample:
        temperature = max(0.01, min(float(temperature), 2.0))
        args["temperature"] = temperature

    # num_beams
    num_beams = getattr(request, "num_beams", None)
    if num_beams is not None:
        if num_beams < 1:
            raise InvalidGenerationParameter("num_beams must be >= 1")
        args["num_beams"] = num_beams
    if do_sample and num_beams and num_beams > 1:
        print(
            f"do_sample=True incompatible with num_beams={num_beams}, "
            f"forcing num_beams=1"
        )
        args["num_beams"] = 1

    # top_p
    if top_p is not None and do_sample:
        top_p = max(0.0, min(float(top_p), 1.0))
        if top_p < 1.0:
            args["top_p"] = top_p

    # filter / drop None
    if allowed_generate_args is not None:
        args = {k: v for k, v in args.items() if k in allowed_generate_args}

    args = {k: v for k, v in args.items() if v is not None}

    return args


def _universal_chat_prompt_fallback(
    current_message: str,
    history: Optional[list],
    enable_thinking: bool = False,
) -> str:
    """
    Universal ChatML-style fallback used when the tokenizer has no chat
    template.
    """
    system_content = "You are a helpful assistant."
    if not enable_thinking:
        system_content += " Provide concise, direct answers."

    parts = [f"<|im_start|>system\n{system_content}<|im_end|>"]

    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    parts.append(f"<|im_start|>user\n{current_message}<|im_end|>")
    parts.append("<|im_start|>assistant")

    return "\n".join(parts)


def format_chat_prompt(
    current_message: str,
    history: Optional[list],
    enable_thinking: bool = False,
    tokenizer=None,
):
    """
    Build a formatted prompt string and return (prompt, reasoning_supported).

    Priority:
      1. tokenizer.apply_chat_template with enable_thinking=True  (Qwen3+)
      2. tokenizer.apply_chat_template standard                   (all chat models)
      3. Universal ChatML fallback                                (no chat template)
    """
    messages = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": current_message})

    if (
        tokenizer
        and hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template
    ):
        # Try the enable_thinking kwarg first (Qwen3 and future reasoning models)
        if enable_thinking:
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                return prompt, True
            except TypeError:
                pass

        # Standard chat template (covers the vast majority of models)
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt, False
        except Exception:
            pass

    # Universal fallback, no model-specific logic
    prompt = _universal_chat_prompt_fallback(
        current_message, history, enable_thinking=enable_thinking
    )
    return prompt, False


def extract_reasoning_and_answer(text: str):
    """
    Extract reasoning blocks and return (reasoning, answer).
    """
    reasoning_blocks = []

    def _collect(match):
        reasoning_blocks.append(match.group(0))
        return ""

    cleaned = re.sub(
        r"<\s*(think|reflection|thought|internal|analysis)\s*>(.*?)<\s*/\1\s*>",
        lambda m: _collect(m),
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    reasoning = "\n\n".join(
        re.sub(r"<[^>]+>", "", b).strip() for b in reasoning_blocks
    ).strip()

    # Strip role scaffolding tokens
    cleaned = re.sub(r"<\|im_start\|>\s*\w+\s*", "", cleaned)
    cleaned = re.sub(r"<\|im_end\|>", "", cleaned)
    cleaned = re.sub(r"<\|assistant\|>", "", cleaned)
    cleaned = re.sub(r"<\|user\|>", "", cleaned)
    cleaned = re.sub(r"<\|system\|>", "", cleaned)
    cleaned = re.sub(r"(?i)\bassistant\s*[:：]\s*", "", cleaned)
    cleaned = re.sub(r"(?i)\b(system|user)\s*[:：]\s*", "", cleaned)

    cleaned = cleaned.strip().replace("\r", "")

    if "\n\n" in cleaned:
        parts = [p.strip() for p in cleaned.split("\n\n") if len(p.strip()) > 10]
        if parts:
            cleaned = parts[-1]

    return reasoning, cleaned or "[No output produced]"


def _get_think_end_token_id(tokenizer):
    """Resolve the closing think-token ID from the tokenizer vocab at runtime."""
    if hasattr(tokenizer, "_cached_think_end_id"):
        return tokenizer._cached_think_end_id

    candidates = ["</think>", "<|/think|>", "</reflection>", "</thought>"]
    for token in candidates:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            tokenizer._cached_think_end_id = token_id
            return token_id

    tokenizer._cached_think_end_id = None
    return None


def post_process_output_ids(output_ids, tokenizer, enable_thinking: bool):
    """
    Split new token IDs into (reasoning_text, answer_text).
    Uses the tokenizer vocab to find the think-end token.
    Falls back to regex splitting when no such token exists.
    """
    think_end_id = _get_think_end_token_id(tokenizer)

    if think_end_id is not None and think_end_id in output_ids:
        try:
            index = len(output_ids) - output_ids[::-1].index(think_end_id)
            reasoning = tokenizer.decode(
                output_ids[:index], skip_special_tokens=True
            ).strip("\n")
            answer = tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")
            return (reasoning if enable_thinking else None), answer
        except ValueError:
            pass

    full_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    reasoning, answer = extract_reasoning_and_answer(full_text)
    return (reasoning if enable_thinking else None), answer


class ResponseFormatter:
    """Centralised response formatting for all API endpoints.

    Supported output formats:
      - "openai"  OpenAI chat-completion schema (default for ChatCompletionRequest)
      - "raw"     Plain text / minimal dict (default for GenerationRequest)
    """

    @staticmethod
    def format_non_streaming_response(
        request,
        output_text: str,
        prompt_tokens: int,
        completion_tokens: int,
        start_time: float,
        reasoning_text: Optional[str] = None,
        reasoning_tokens: int = 0,
        finish_reason: str = "stop",
    ) -> Dict[str, Any]:
        """Format a complete non-streaming response."""
        processing_time = time.time() - start_time

        if request.output_format == "openai":
            message = {"role": "assistant", "content": output_text}
            if reasoning_text:
                message[REASONING_FIELD] = reasoning_text

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            if reasoning_text:
                usage["completion_tokens_details"] = {
                    "reasoning_tokens": reasoning_tokens
                }

            return {
                "id": str(request.id),
                "object": "chat.completion",
                "created": int(start_time),
                "model": request.hf_name,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage,
                "processing_time": processing_time,
            }

        # "raw" - minimal plain-text response
        response: Dict[str, Any] = {"text": output_text}
        if finish_reason != "stop":
            response["finish_reason"] = finish_reason
        if reasoning_text:
            response[REASONING_FIELD] = reasoning_text
        return response

    @staticmethod
    def format_stream_chunk(
        request,
        token_text: str,
        index: int,
        start_time: float,
        kind: str = "content",
    ) -> str:
        """Format a single streaming chunk as an SSE event string.

        kind: "content" or "reasoning". Each chunk carries exactly one of
        delta.content / delta.<REASONING_FIELD>, never both.
        """
        token_text = str(token_text)

        if request.output_format == "openai":
            chunk_data = {
                "id": str(request.id),
                "object": "chat.completion.chunk",
                "created": int(start_time),
                "model": request.hf_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            (
                                REASONING_FIELD if kind == "reasoning" else "content"
                            ): token_text
                        },
                        "finish_reason": None,
                    }
                ],
            }
        else:
            # "raw" - lightweight token event
            chunk_data = {
                "id": str(request.id),
                "token": token_text,
                "kind": kind,
                "index": index,
                "done": False,
            }

        return f"data: {json.dumps(chunk_data)}\n\n"

    @staticmethod
    def format_stream_final(
        request,
        start_time: float,
        prompt_tokens: int,
        completion_tokens: int,
        full_text: Optional[str] = None,
        reasoning_text: Optional[str] = None,
        reasoning_tokens: int = 0,
        finish_reason: str = "stop",
    ) -> str:
        """Format the final streaming chunk with usage statistics."""
        if request.output_format == "openai":
            final_data: Dict[str, Any] = {
                "id": str(request.id),
                "object": "chat.completion.chunk",
                "created": int(start_time),
                "model": request.hf_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            # Reasoning was already streamed as deltas; only report its cost.
            if reasoning_tokens:
                final_data["usage"]["completion_tokens_details"] = {
                    "reasoning_tokens": reasoning_tokens
                }

        else:
            # "raw" final chunk
            final_data = {
                "id": str(request.id),
                "done": True,
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            if full_text is not None:
                final_data["text"] = full_text
            if reasoning_text:
                final_data[REASONING_FIELD] = reasoning_text

        return f"data: {json.dumps(final_data)}\n\ndata: [DONE]\n\n"

    @staticmethod
    def format_error_response(
        error_message: str,
        error_type: str = "internal_error",
        status_code: int = 500,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format a structured error response."""
        error_data: Dict[str, Any] = {
            "error": {
                "message": error_message,
                "type": error_type,
                "code": status_code,
            }
        }
        if request_id:
            error_data["id"] = request_id
        return error_data

    @staticmethod
    def format_stream_error(
        error_message: str, error_type: str = "generation_error"
    ) -> str:
        """Format an error as an SSE event string for streaming responses."""
        error_data = {"error": {"message": error_message, "type": error_type}}
        return f"data: {json.dumps(error_data)}\n\n"
