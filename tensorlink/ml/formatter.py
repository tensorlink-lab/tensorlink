from typing import Dict, Any, Optional
import json
import time
import re


def normalize_generate_args(
    request,
    tokenizer,
    prompt_tokens: Optional[int] = None,
    model_max_length: Optional[int] = None,
    allowed_generate_args: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Normalize and validate generation arguments without injecting defaults.
    Only user-provided, non-None values are included.
    """

    # TOKEN IDs
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
        model_max_length = getattr(tokenizer, "model_max_length", 2048)
        if model_max_length > 1_000_000:
            model_max_length = 2048

    args: Dict[str, Any] = {
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
    }

    # ---------- MAX_NEW_TOKENS ----------
    max_new_tokens = getattr(request, "max_new_tokens", None)

    if max_new_tokens is not None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

        if prompt_tokens is not None:
            available_space = model_max_length - prompt_tokens

            if available_space < 10:
                raise ValueError(
                    f"Prompt is too long ({prompt_tokens} tokens). "
                    f"Model max length is {model_max_length}, leaving only "
                    f"{available_space} tokens for generation."
                )

            if max_new_tokens > available_space:
                original = max_new_tokens
                max_new_tokens = available_space
                print(
                    f"Reduced max_new_tokens from {original} to {max_new_tokens} "
                    f"to fit model limit ({model_max_length})"
                )

        args["max_new_tokens"] = max_new_tokens

    # ---------- DO_SAMPLE ----------
    do_sample = getattr(request, "do_sample", None)
    if do_sample is not None:
        do_sample = bool(do_sample)
        args["do_sample"] = do_sample
    else:
        do_sample = None

    # ---------- TEMPERATURE ----------
    temperature = getattr(request, "temperature", None)
    if temperature is not None and do_sample:
        temperature = max(0.01, min(float(temperature), 2.0))
        args["temperature"] = temperature

    # ---------- NUM_BEAMS ----------
    num_beams = getattr(request, "num_beams", None)
    if num_beams is not None:
        if num_beams < 1:
            raise ValueError("num_beams must be >= 1")
        args["num_beams"] = num_beams

    if do_sample and num_beams and num_beams > 1:
        print(
            f"do_sample=True incompatible with num_beams={num_beams}, "
            f"forcing num_beams=1"
        )
        args["num_beams"] = 1

    # ---------- TOP_P ----------
    top_p = getattr(request, "top_p", None)
    if top_p is not None and do_sample:
        top_p = max(0.0, min(float(top_p), 1.0))
        if top_p < 1.0:
            args["top_p"] = top_p

    # ---------- FILTER ALLOWED ----------
    if allowed_generate_args is not None:
        args = {k: v for k, v in args.items() if k in allowed_generate_args}

    # ---------- DROP NONE ----------
    args = {k: v for k, v in args.items() if v is not None}

    return args


def extract_reasoning_and_answer(text: str):
    """
    Extract reasoning blocks and clean answer.
    Returns (reasoning, answer)
    """

    reasoning_blocks = []

    def _collect(match):
        reasoning_blocks.append(match.group(0))
        return ""

    # Capture <think>, <analysis>, etc
    cleaned = re.sub(
        r"<\s*(think|reflection|thought|internal|analysis)\s*>(.*?)<\s*/\1\s*>",
        lambda m: _collect(m),
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    reasoning = "\n\n".join(
        re.sub(r"<[^>]+>", "", b).strip() for b in reasoning_blocks
    ).strip()

    # Clean scaffolding
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


def format_chat_prompt_manual(
    model_name, current_message, history, enable_thinking=True
):
    """
    Manually format the chat history and current message into a prompt.
    This is the fallback for models without native reasoning support.

    Args:
        model_name: Name of the model
        current_message: Current user message
        history: Conversation history
        enable_thinking: Whether to allow reasoning/thinking tokens
    """
    # Different models require different formatting
    if "Qwen" in model_name:
        system_prompt = (
            "You are a helpful assistant. Respond directly to the user's questions."
        )

        # Modify system prompt to discourage thinking if disabled
        if not enable_thinking:
            system_prompt += " Provide concise, direct answers without showing your reasoning/thinking process."

        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

        if history and len(history) > 0:
            for msg in history:
                role = msg["role"]
                content = msg["content"]
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        formatted_prompt += f"<|im_start|>user\n{current_message}<|im_end|>\n"
        formatted_prompt += "<|im_start|>assistant\n"

        return formatted_prompt

    elif "llama" in model_name.lower():
        system_prompt = (
            "You are a helpful assistant. Respond directly to the user's questions."
        )

        if not enable_thinking:
            system_prompt += " Provide concise, direct answers without showing your reasoning process."

        formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"

        if history and len(history) > 0:
            for i, msg in enumerate(history):
                if msg["role"] == "user":
                    if i > 0:
                        formatted_prompt += "[/INST]\n\n[INST] "
                    formatted_prompt += f"{msg['content']}"
                else:
                    formatted_prompt += f" [/INST]\n\n{msg['content']}\n\n[INST] "

        formatted_prompt += f"{current_message} [/INST]\n\n"
        return formatted_prompt

    else:
        system_prompt = (
            "You are a helpful assistant. Respond directly to the user's questions."
        )

        if not enable_thinking:
            system_prompt += " Provide concise, direct answers without showing your reasoning process."

        formatted_prompt = f"System: {system_prompt}\n\n"

        if history and len(history) > 0:
            for msg in history:
                role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
                formatted_prompt += f"{role_prefix}{msg['content']}\n\n"

        formatted_prompt += f"User: {current_message}\n\nAssistant: "
        return formatted_prompt


def format_chat_prompt(
    model_name, current_message, history, enable_thinking=True, tokenizer=None
):
    """
    Format the chat history and current message into a prompt.
    Uses tokenizer's apply_chat_template if it supports enable_thinking,
    otherwise falls back to manual formatting.

    Args:
        model_name: Name of the model
        current_message: Current user message
        history: Conversation history
        enable_thinking: Whether to allow reasoning/thinking tokens
        tokenizer: Optional tokenizer instance (if None, uses manual formatting)

    Returns:
        tuple: (formatted_prompt, reasoning_supported)
    """
    supports_reasoning = getattr(tokenizer, "supports_reasoning", False)
    # Check if tokenizer supports native reasoning
    if tokenizer and supports_reasoning:
        # Build messages list
        messages = []
        if history and len(history) > 0:
            messages.extend(history)
        messages.append({"role": "user", "content": current_message})

        # Use tokenizer's native reasoning support
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        return formatted_prompt, True

    else:
        # Fall back to manual formatting
        formatted_prompt = format_chat_prompt_manual(
            model_name, current_message, history, enable_thinking=enable_thinking
        )

        return formatted_prompt, False


def format_stream_final(request, start_time, prompt_tokens, token_count):
    if request.output_format == "openai":
        return {
            "id": request.id,
            "object": "chat.completion.chunk",
            "created": int(start_time),
            "model": request.hf_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": token_count,
                "total_tokens": prompt_tokens + token_count,
            },
        }


def format_stream_chunk(request, token_text, index, start_time):
    """Format a single streaming token chunk"""
    token_text = str(token_text)  # ensure it's always a string

    if request.output_format == "openai":
        return {
            "id": request.id,
            "object": "chat.completion.chunk",
            "created": int(start_time),
            "model": request.hf_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": token_text},
                    "finish_reason": None,
                }
            ],
        }
    else:
        return {
            "id": request.id,
            "token": token_text,
            "index": index,
            "done": False,
        }


class ResponseFormatter:
    """Centralized response formatting for all API endpoints"""

    @staticmethod
    def format_non_streaming_response(
        request,
        output_text: str,
        prompt_tokens: int,
        completion_tokens: int,
        start_time: float,
        reasoning_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Format a complete non-streaming response.

        Args:
            request: GenerationRequest object
            output_text: The generated text
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens generated
            start_time: Generation start timestamp
            reasoning_text: Extracted reasoning/thinking text (if any)

        Returns:
            Formatted response dict based on output_format
        """
        processing_time = time.time() - start_time

        if request.output_format == "openai":
            response = {
                "id": str(request.id),
                "object": "chat.completion",
                "created": int(start_time),
                "model": request.hf_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": output_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "processing_time": processing_time,
            }

            # Add reasoning to message if present
            if reasoning_text:
                response["choices"][0]["message"]["reasoning"] = reasoning_text

            return response

        elif request.output_format == "simple":
            response = {
                "id": str(request.id),
                "model": request.hf_name,
                "text": output_text,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "processing_time": processing_time,
                "finish_reason": "stop",
            }

            # Add reasoning as separate field if present
            if reasoning_text:
                response["reasoning"] = reasoning_text

            return response
        else:
            # Raw format
            response = {"text": output_text}
            if reasoning_text:
                response["reasoning"] = reasoning_text
            return response

    @staticmethod
    def format_stream_chunk(
        request, token_text: str, index: int, start_time: float
    ) -> str:
        """
        Format a single streaming token chunk as SSE.

        Args:
            request: GenerationRequest object
            token_text: The token text to stream
            index: Token index in the sequence
            start_time: Generation start timestamp

        Returns:
            Formatted SSE chunk string (includes "data: " prefix and newlines)
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
                        "delta": {"content": token_text},
                        "finish_reason": None,
                    }
                ],
            }
        else:
            chunk_data = {
                "id": str(request.id),
                "model": request.hf_name,
                "token": token_text,
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
    ) -> str:
        """
        Format the final streaming chunk with usage stats.

        Args:
            request: GenerationRequest object
            start_time: Generation start timestamp
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens generated
            full_text: Complete generated text (optional, for simple format)

        Returns:
            Formatted final SSE chunk string
        """
        if request.output_format == "openai":
            final_data = {
                "id": str(request.id),
                "object": "chat.completion.chunk",
                "created": int(start_time),
                "model": request.hf_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

            if reasoning_text:
                final_data["reasoning"] = reasoning_text

            return f"data: {json.dumps(final_data)}\n\ndata: [DONE]\n\n"
        else:
            # Simple format final chunk
            final_data = {
                "id": str(request.id),
                "model": request.hf_name,
                "done": True,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            if full_text is not None:
                final_data["full_text"] = full_text

            if reasoning_text:
                final_data["reasoning"] = reasoning_text

            return f"data: {json.dumps(final_data)}\n\ndata: [DONE]\n\n"

    @staticmethod
    def format_error_response(
        error_message: str,
        error_type: str = "internal_error",
        status_code: int = 500,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Format an error response.

        Args:
            error_message: Error description
            error_type: Type of error
            status_code: HTTP status code
            request_id: Optional request ID

        Returns:
            Formatted error dict
        """
        error_data = {
            "error": {"message": error_message, "type": error_type, "code": status_code}
        }

        if request_id:
            error_data["id"] = request_id

        return error_data

    @staticmethod
    def format_stream_error(
        error_message: str, error_type: str = "generation_error"
    ) -> str:
        """
        Format an error for streaming responses.

        Returns:
            SSE-formatted error chunk
        """
        error_data = {"error": {"message": error_message, "type": error_type}}
        return f"data: {json.dumps(error_data)}\n\n"
