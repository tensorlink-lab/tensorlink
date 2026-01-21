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
    Normalize and validate generation arguments to prevent errors.

    Args:
        request: GenerationRequest object
        tokenizer: The tokenizer for the model
        prompt_tokens: Number of tokens in the prompt (if already computed)
        model_max_length: Maximum sequence length the model supports
        allowed_generate_args: Generate function to get input args
    Returns:
        Dictionary of validated generation arguments
    """

    # TOKEN IDs
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)

    # Fallback if None
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0
    if eos_token_id is None:
        eos_token_id = pad_token_id

    # Prevent identical pad and eos (causes generation to stop immediately)
    if pad_token_id == eos_token_id:
        if pad_token_id == 0 and vocab_size > 1:
            eos_token_id = 1
        elif pad_token_id > 0:
            eos_token_id = 0
        # Ensure within vocab bounds
        eos_token_id = min(eos_token_id, vocab_size - 1)

    # MODEL CONSTRAINTS
    # Get model's maximum sequence length
    if model_max_length is None:
        model_max_length = getattr(tokenizer, 'model_max_length', 2048)
        # Some tokenizers have unrealistic defaults
        if model_max_length > 1000000:
            model_max_length = 2048

    # MAX_NEW_TOKENS
    max_new_tokens = getattr(request, "max_new_tokens", None)

    # Default if not specified
    if not max_new_tokens or max_new_tokens < 1:
        max_new_tokens = 256

    # Ensure we have room for generation
    if prompt_tokens is not None:
        # Calculate available space for new tokens
        available_space = model_max_length - prompt_tokens

        if available_space < 10:
            # Prompt is too long, we need at least some room to generate
            raise ValueError(
                f"Prompt is too long ({prompt_tokens} tokens). "
                f"Model max length is {model_max_length}, leaving only "
                f"{available_space} tokens for generation. "
                f"Please use a shorter prompt."
            )

        # Cap max_new_tokens to available space
        if max_new_tokens > available_space:
            original = max_new_tokens
            max_new_tokens = available_space
            print(
                f"Reduced max_new_tokens from {original} to {max_new_tokens} "
                f"to fit within model's {model_max_length} token limit "
                f"(prompt uses {prompt_tokens} tokens)"
            )

    # Ensure minimum generation length
    max_new_tokens = max(max_new_tokens, 1)

    # TEMPERATURE
    temperature = getattr(request, "temperature", None)
    if temperature is None or temperature <= 0:
        temperature = 0.7

    # Clamp to reasonable range
    temperature = max(0.01, min(temperature, 2.0))

    # SAMPLING
    do_sample = bool(getattr(request, "do_sample", True))

    # Force do_sample=False if temperature is very low (greedy decoding)
    if temperature < 0.01:
        do_sample = False
        temperature = 1.0  # Temperature ignored when do_sample=False

    # BEAMS
    num_beams = getattr(request, "num_beams", None)
    if not num_beams or num_beams < 1:
        num_beams = 1

    # INCOMPATIBLE COMBINATIONS
    # Can't use sampling with beam search (in most implementations)
    if do_sample and num_beams > 1:
        print(
            f"do_sample=True is incompatible with num_beams={num_beams}. "
            f"Setting num_beams=1"
        )
        num_beams = 1

    # TOP_P (if provided)
    top_p = getattr(request, "top_p", None)
    if top_p is not None:
        top_p = max(0.0, min(top_p, 1.0))

    # BUILD ARGS DICT and FILTER BY GENERATE SIGNATURE
    args = {
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "num_beams": num_beams,
    }
    if top_p is not None and do_sample:
        args["top_p"] = top_p
    if getattr(request, "reasoning", None) is not None:
        args["reasoning"] = request.reasoning
    if getattr(request, "enable_thinking", None) is not None:
        args["enable_thinking"] = request.enable_thinking

    # Filter based on allowed kwargs
    if allowed_generate_args is not None:
        if allowed_generate_args is not None:
            if allowed_generate_args != None:
                args = {k: v for k, v in args.items() if k in allowed_generate_args}

    return args


def extract_assistant_response(text: str, model_name: str = None) -> str:
    """
    Universal extractor that removes system/user/thought tags and returns
    the final human-readable assistant response.
    """

    # Remove reasoning or hidden thought blocks (e.g. <think>...</think>)
    text = re.sub(
        r"<\s*(think|reflection|thought|internal|analysis)\s*>.*?<\s*/\1\s*>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove common chat tags used by newer models
    text = re.sub(r"<\|im_start\|>\s*\w+\s*", "", text)
    text = re.sub(r"<\|im_end\|>", "", text)
    text = re.sub(r"<\|assistant\|>", "", text)
    text = re.sub(r"<\|user\|>", "", text)
    text = re.sub(r"<\|system\|>", "", text)

    # Strip out any prefixes like "assistant:" or "Assistant:"
    text = re.sub(r"(?i)\bassistant\s*[:：]\s*", "", text)

    # Remove lingering system/user scaffolding
    text = re.sub(r"(?i)\b(system|user)\s*[:：]\s*", "", text)
    text = text.strip().replace("\r", "")

    # If multiple paragraphs, prefer the last coherent chunk
    # (models sometimes prepend hidden reasoning)
    if "\n\n" in text:
        parts = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 10]
        if parts:
            text = parts[-1]

    # Fallback: if text still empty, just return as-is (safe default)
    return text.strip() or "[No output produced]"


def format_chat_prompt(model_name, current_message, history):
    """Format the chat history and current message into a prompt suitable for the specified model."""

    # Different models require different formatting
    if "Qwen" in model_name:
        # Qwen-specific formatting
        system_prompt = (
            "You are a helpful assistant. Respond directly to the user's questions."
        )

        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

        # Add conversation history
        if history and len(history) > 0:
            for msg in history:
                role = msg["role"]
                content = msg["content"]
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        # Add the current message
        formatted_prompt += f"<|im_start|>user\n{current_message}<|im_end|>\n"
        formatted_prompt += "<|im_start|>assistant\n"

        return formatted_prompt

    elif "llama" in model_name.lower():
        # Llama-style formatting
        system_prompt = (
            "You are a helpful assistant. Respond directly to the user's questions."
        )
        formatted_prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"

        # Add conversation history
        if history and len(history) > 0:
            for i, msg in enumerate(history):
                if msg["role"] == "user":
                    if i > 0:
                        formatted_prompt += "[/INST]\n\n[INST] "
                    formatted_prompt += f"{msg['content']}"
                else:  # assistant
                    formatted_prompt += f" [/INST]\n\n{msg['content']}\n\n[INST] "

        # Add the current message and prepare for response
        formatted_prompt += f"{current_message} [/INST]\n\n"

        return formatted_prompt

    else:
        # Generic formatting for other models
        system_prompt = (
            "You are a helpful assistant. Respond directly to the user's questions."
        )
        formatted_prompt = f"System: {system_prompt}\n\n"

        # Add conversation history
        if history and len(history) > 0:
            for msg in history:
                role_prefix = "User: " if msg["role"] == "user" else "Assistant: "
                formatted_prompt += f"{role_prefix}{msg['content']}\n\n"

        # Add the current message
        formatted_prompt += f"User: {current_message}\n\nAssistant: "

        return formatted_prompt


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
    ) -> Dict[str, Any]:
        """
        Format a complete non-streaming response.

        Args:
            request: GenerationRequest object
            output_text: The generated text
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens generated
            start_time: Generation start timestamp

        Returns:
            Formatted response dict based on output_format
        """
        processing_time = time.time() - start_time

        if request.output_format == "openai":
            return {
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
        elif request.output_format == "simple":
            # Simple format with metadata
            return {
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
        else:
            # Raw format - just the text (legacy compatibility)
            return {"text": output_text}

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
            # Simple streaming format
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
