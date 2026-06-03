# Inference API

OpenAI-compatible and native HTTP endpoints for distributed inference.

Tensorlink exposes a lightweight HTTP API for running Hugging Face models across the network. This is the simplest way to get started, with no GPU or Python required.

> The API is served by a validator node at `http://localhost:64747` by default. To use the public network without running your own node, see [Node Setup](nodes.md) for connecting to a hosted endpoint.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completions |
| `POST` | `/v1/responses` | Multi-modal response API (text, image, embeddings) |
| `POST` | `/v1/models/request` | Request a model to be loaded on the network |
| `GET`  | `/v1/models/status` | Check the loading status of a specific model |
| `GET`  | `/v1/models/available` | List all currently active (fully loaded) models |
| `GET`  | `/v1/models/demand` | API demand statistics by model |

---

## OpenAI-Compatible Chat

The `/v1/chat/completions` endpoint is a drop-in replacement for the OpenAI API and supports the same message format, parameters, and streaming behaviour.

```python
import requests

response = requests.post(
    "http://localhost:64747/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-14B",
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

**Streaming (SSE):**

```python
response = requests.post(
    "http://localhost:64747/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-14B",
        "messages": [{"role": "user", "content": "Write a haiku about distributed computing."}],
        "max_tokens": 100,
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

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | -       | Hugging Face model identifier |
| `messages` | array | -       | Chat message list (`role`, `content`) |
| `max_tokens` | int | 1024    | Maximum tokens to generate |
| `temperature` | float | 0.7     | Sampling temperature |
| `top_p` | float | 1.0     | Nucleus sampling probability |
| `stream` | boolean | false   | Stream tokens via SSE |
| `stop` | string \| array | null    | Stop sequence(s) |
| `n` | int | 1       | Number of completions to generate |
| `presence_penalty` | float | 0.0     | Presence penalty |
| `frequency_penalty` | float | 0.0     | Frequency penalty |

> If the requested model is not yet loaded, the API returns a `503` and automatically triggers loading. Retry after a short delay.

---

## Responses API

The `/v1/responses` endpoint is a unified multi-modal API. The request type is declared via the `type` field.

### Text *(type: "text")*

Functionally equivalent to `/v1/chat/completions` under the new envelope.

```python
import requests

response = requests.post(
    "http://localhost:64747/v1/responses",
    json={
        "type": "text",
        "model": "Qwen/Qwen3-14B",
        "messages": [
            {"role": "user", "content": "Summarize the theory of relativity."}
        ],
        "max_tokens": 200,
        "temperature": 0.7,
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Image *(type: "image") - coming soon*

```python
{
    "type": "image",
    "model": "black-forest-labs/FLUX.1-schnell",
    "prompt": "A futuristic city at night, cyberpunk style",
    "n": 1,
    "size": "1024x1024",
    "quality": "standard",       # "standard" | "hd"
    "response_format": "url"     # "url" | "b64_json"
}
```

### Embeddings *(type: "embedding") - coming soon*

```python
{
    "type": "embedding",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "The quick brown fox jumps over the lazy dog",
    "encoding_format": "float",  # "float" | "base64"
    "dimensions": 384            # optional
}
```

---

## Model Management

### Request a Model - `POST /v1/models/request`

Explicitly request a model to be loaded on the network. If the model is already loaded or loading, the response reflects that status. If not found, loading is initiated immediately.

```python
import requests

response = requests.post(
    "http://localhost:64747/v1/models/request",
    json={
        "hf_name": "Qwen/Qwen3-14B",
        "model_type": "chat",   # optional
        "time": 1800,           # lease duration in seconds (optional)
        "payment": 0            # reserved for future paid jobs (optional)
    }
)

print(response.json())
```

**Response:**

```json
{
    "model_name": "Qwen/Qwen3-14B",
    "status": "inactive",
    "message": "Model Qwen/Qwen3-14B not found. Loading has been initiated."
}
```

Possible `status` values:

| Status | Meaning |
|--------|---------|
| `"active"` | Validator and all workers have fully loaded the model |
| `"loading"` | Job exists but worker(s) are still loading modules |
| `"inactive"` | Model not found; loading has been initiated |

> Paid jobs for private model access are not yet available. All models are currently loaded as public shared resources.

---

### Check Model Status - `GET /v1/models/status`

Check whether a specific model is active, loading, or not present. Pass the model name as a query parameter to avoid path-routing issues with names that contain slashes.

```
GET /v1/models/status?model_name=Qwen/Qwen3-14B
```

```python
import requests

response = requests.get(
    "http://localhost:64747/v1/models/status",
    params={"model_name": "Qwen/Qwen3-14B"}
)

print(response.json())
# {"model_name": "Qwen/Qwen3-14B", "status": "active", "message": "..."}
```

---

### List Available Models - `GET /v1/models/available`

Returns all models that are currently active and ready to serve inference requests.

```python
import requests

response = requests.get("http://localhost:64747/v1/models/available")
print(response.json())
# {"active_models": ["Qwen/Qwen3-14B", ...]}
```

---

### Demand Statistics - `GET /v1/models/demand`

Returns API request demand statistics, ranked by model popularity. Useful for understanding which models are worth preloading.

```
GET /v1/models/demand?days=30&limit=10
```

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `days` | int | 30 | 1–90 | Lookback window in days |
| `limit` | int | 10 | 1–50 | Maximum number of models to return |
