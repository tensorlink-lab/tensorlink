# Inference API

OpenAI-compatible and native HTTP endpoints for distributed inference.

Tensorlink exposes a lightweight HTTP API for running Hugging Face models across the network. This is the simplest way 
to get started, with no GPU or Python required.

> The API is served by a validator node at `http://localhost:64747` by default. To use the public network without running your own node, see [Node Setup](nodes.md) for connecting to a hosted endpoint.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/generate` | Simple single-turn text generation |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |
| `POST` | `/v1/responses` | Multi-modal response API (text, image, embeddings) |
| `POST` | `/request-model` | Preload a model onto the network |

---

## Simple Generation

```python
import requests

response = requests.post(
    "http://localhost:64747/v1/generate",
    json={
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "message": "Explain quantum computing in one sentence.",
        "max_new_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": False,
    }
)

print(response.json()["generated_text"])
```

**Streaming (SSE):**

```python
response = requests.post(
    "http://localhost:64747/v1/generate",
    json={
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "message": "Write a haiku about distributed computing.",
        "max_new_tokens": 100,
        "stream": True,
    },
    stream=True,
)

for line in response.iter_lines():
    if line:
        decoded = line.decode()
        if decoded.strip() == "data: [DONE]":
            break
        if decoded.startswith("data: "):
            print(decoded[6:], end="", flush=True)
```

---

## OpenAI-Compatible Chat

The `/v1/chat/completions` endpoint is a drop-in replacement for the OpenAI API and supports the same message format, parameters, and streaming behaviour.

```python
import requests

response = requests.post(
    "http://localhost:64747/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the benefits of distributed computing?"}
        ],
        "max_tokens": 150,
        "temperature": 0.8,
        "stream": False
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | — | Hugging Face model identifier |
| `messages` | array | — | Chat message list (`role`, `content`) |
| `max_tokens` | int | 1024 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature |
| `top_p` | float | 1.0 | Nucleus sampling probability |
| `stream` | boolean | false | Stream tokens via SSE |
| `stop` | string \| array | null | Stop sequence(s) |
| `n` | int | 1 | Number of completions to generate |

---

## Responses API

The `/v1/responses` endpoint is a unified multi-modal API. The request type is declared via the `type` field, and each modality has its own set of parameters.

### Text *(type: "text")*

Functionally equivalent to `/v1/chat/completions` under the new envelope.

```python
import requests

response = requests.post(
    "http://localhost:64747/v1/responses",
    json={
        "type": "text",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "user", "content": "Summarize the theory of relativity."}
        ],
        "max_tokens": 200,
        "temperature": 0.7,
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Image *(type: "image") — coming soon*

```python
{
    "type": "image",
    "model": "black-forest-labs/FLUX.1-schnell",
    "prompt": "A futuristic city at night, cyberpunk style",
    "n": 1,
    "size": "1024x1024",
    "quality": "standard",           # "standard" | "hd"
    "response_format": "url"         # "url" | "b64_json"
}
```

### Embeddings *(type: "embedding") — coming soon*

```python
{
    "type": "embedding",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "The quick brown fox jumps over the lazy dog",
    "encoding_format": "float",      # "float" | "base64"
    "dimensions": 384                # optional
}
```

---

## Model Preloading

Request a model be loaded across the network before inference to reduce first-request latency. If the model is already loaded or loading, the response reflects that status.

```python
import requests

requests.post(
    "http://localhost:64747/request-model",
    json={
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_type": "chat"
    }
)
```

**Response:**

```json
{
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "status": "loading",
    "message": "Model has been requested and is loading."
}
```

Possible `status` values: `"loaded"`, `"loading"`, `"requested"`, `"error"`.

> If you send an inference request to a model that isn't loaded yet, the API will automatically trigger loading and return a `503` — retry after a short delay.