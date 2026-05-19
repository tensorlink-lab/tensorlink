"""
HTTP API example usage for queries on the public P2P network.
"""

import requests
import json


SERVER_URL = "https://tensorlink.ddns.net/tensorlink"  # Use HTTP if HTTPS fails
MODEL_NAME = "Qwen/Qwen3-14B"


def chat_completion():
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": "Explain distributed inference in one sentence.",
            },
        ],
        "max_tokens": 128,
        "temperature": 0.7,
        "stream": False,
    }

    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        timeout=120,
    )

    result = response.json()

    print(json.dumps(result, indent=2))


def chat_completion_stream():
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": "Count from one to five.",
            }
        ],
        "max_tokens": 32,
        "temperature": 0.1,
        "stream": True,
    }

    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=120,
    )

    full_text = ""

    for line in response.iter_lines():
        if not line:
            continue

        decoded = line.decode("utf-8")

        if not decoded.startswith("data: "):
            continue

        data = decoded[6:]

        if data == "[DONE]":
            break

        chunk = json.loads(data)

        delta = chunk["choices"][0].get("delta", {})
        token = delta.get("content")

        if token:
            full_text += token
            print(token, end="", flush=True)

    print("\n\nDone.")


def responses_text():
    payload = {
        "model": MODEL_NAME,
        "type": "text",
        "messages": [
            {
                "role": "user",
                "content": "What is 2 + 2?",
            }
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
    }

    response = requests.post(
        f"{SERVER_URL}/v1/responses",
        json=payload,
        timeout=120,
    )

    result = response.json()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    # request_model(SERVER_URL, MODEL_NAME)

    # Non-streaming
    chat_completion()

    # Streaming
    chat_completion_stream()

    # Responses API
    responses_text()
