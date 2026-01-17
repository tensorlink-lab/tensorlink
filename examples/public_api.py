"""
HTTP API example usage for queries on the public P2P network.
"""

import requests
import json


SERVER_URL = "https://smartnodes.ddns.net/tensorlink-api"  # Use HTTP if HTTPS fails
MODEL_NAME = "Qwen/Qwen3-8B"


def generate():
    payload = {
        "hf_name": MODEL_NAME,
        "message": "Describe the role of AI in medicine.",
        "max_length": 4096,
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "do_sample": True,
        "num_beams": 3,
        "history": [
            {"role": "user", "content": "What is artificial intelligence?"},
            {"role": "assistant", "content": "Artificial intelligence refers to..."},
        ],
    }

    response = requests.post(f"{SERVER_URL}/v1/generate", json=payload)
    print(response.json())


def generate_stream():
    payload = {
        "hf_name": MODEL_NAME,
        "message": "Hello!",
        "max_new_tokens": 32,
        "stream": True,
    }

    response = requests.post(
        f"{SERVER_URL}/v1/generate",
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
        token = chunk["choices"][0]["delta"].get("content")

        if token:
            full_text += token
            print(token, end="", flush=True)

    print("\n\nDone.")


def request_model():
    payload = {"hf_name": MODEL_NAME}
    response = requests.post(f"{SERVER_URL}/request-model", json=payload)
    print(response.json())


if __name__ == "__main__":
    request_model()
    generate()
