# Inference API

*OpenAI-style HTTP endpoints for running PyTorch and Hugging Face models on Tensorlink.*

Tensorlink exposes an OpenAI-compatible API, so most existing OpenAI clients and SDKs work by chaning the base URL
and model name. This page covers sending requests and managing which models are loaded. For network status and node 
information endpoints, see [Network Endpoints](network-api.md).

---

## How it works

1. **Connect.** Point your client at a Tensorlink node that exposes the HTTP API (see [Choosing a base URL](#choosing-a-base-url)).
2. **Make sure the model is loaded.** You can request a model explicitly with `POST /v1/models/request` and wait for it to become `active`. You can also skip this step: if you call `/v1/chat/completions` with a model that isn't loaded, the network starts loading it and returns a `503`. Retry until the model is ready.
3. **Send requests.** Once the model is `active`, use `/v1/chat/completions` or `/v1/responses` as you would with any OpenAI-style API.

Loading time depends on the model's size and on how many workers are available. A small model on a healthy network can take a minute or two, and larger models take longer.

---

## Choosing a base URL
 
Every example on this page uses `http://localhost:64747`, the default address of a node's HTTP endpoint. Replace it with the address that matches your setup:
 
| Setup | Base URL                                                                  | What you need |
|-------|---------------------------------------------------------------------------|---------------|
| **Private network** (your own machines) | Your validator's address, e.g. `http://192.168.1.100:64747`               | Run your own validator with `endpoint` enabled. See [Node Setup](nodes.md). |
| **Public network, through a seed node** | `https://<seed-node-host>`, e.g. `https://tensorlink.ddns.net/tensorlink` | No local setup. Use a seed node's HTTP endpoint directly. |
| **Public network, through your own node** | Your node's address, usually `http://localhost:64747`                     | Run a node in public mode, which connects to the network through the seed validators in its config. See [Node Setup](nodes.md). |

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

Network and node endpoints (`/stats`, `/network-history`, `/node-info`, and others) are documented in [Network Endpoints](network-api.md).

---

## Chat Completions

`POST /v1/chat/completions` is a drop-in replacement for the OpenAI endpoint. It accepts the same message format and parameters, and streams the same way.

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

### Using the OpenAI SDK
 
Because the API is OpenAI-compatible, you can use the official SDK by setting `base_url`.
 
```python
from openai import OpenAI
 
client = OpenAI(
    base_url="http://localhost:64747/v1",
    api_key="tensorlink",  # placeholder; the SDK requires a value
)
 
response = client.chat.completions.create(
    model="Qwen/Qwen3-14B",
    messages=[{"role": "user", "content": "What is distributed computing?"}],
)
print(response.choices[0].message.content)
```
 
### Streaming (SSE)
 
Set `"stream": True` to receive tokens as server-sent events. Each event is a JSON chunk in the OpenAI format, and the stream ends with `data: [DONE]`.
 
```python
import json
import requests
 
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
    if not line:
        continue
    decoded = line.decode()
    if not decoded.startswith("data: "):
        continue
    data = decoded[6:]
    if data.strip() == "[DONE]":
        break
    delta = json.loads(data)["choices"][0]["delta"]
    print(delta.get("content", ""), end="", flush=True)
```
 
### Parameters
 
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | - | Hugging Face model identifier, e.g. `Qwen/Qwen3-14B`. The public network supports open models that don't require gated access. |
| `messages` | array | - | Chat message list (`role`, `content`) |
| `max_tokens` | int | 1024 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature |
| `top_p` | float | 1.0 | Nucleus sampling probability |
| `stream` | boolean | false | Stream tokens via SSE |
| `stop` | string \| array | null | Stop sequence(s) |
| `n` | int | 1 | Number of completions to generate |
| `presence_penalty` | float | 0.0 | Presence penalty |
| `frequency_penalty` | float | 0.0 | Frequency penalty |
 
### When the model isn't loaded yet
 
If the requested model isn't loaded, the API returns `503` and starts loading it automatically. Retry after a short delay, or check progress with [`/v1/models/status`](#check-model-status).
 
```python
import time
import requests
 
def chat(payload, base_url="http://localhost:64747", retries=30, delay=10):
    for _ in range(retries):
        r = requests.post(f"{base_url}/v1/chat/completions", json=payload)
        if r.status_code != 503:
            r.raise_for_status()
            return r.json()
        time.sleep(delay)  # model is still loading
    raise TimeoutError("The model did not finish loading in time.")
```
 
---
 
## Responses API
 
`POST /v1/responses` is a unified multi-modal endpoint. The `type` field selects the kind of request.
 
### Text (`type: "text"`)
 
Functionally equivalent to `/v1/chat/completions`, using the newer request envelope.
 
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
 
### Image (`type: "image"`), coming soon
 
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
 
### Embeddings (`type: "embedding"`), coming soon
 
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
 
## Model management
 
These endpoints control which models are loaded on the network. You don't need them for every request, because chat completions load a missing model on demand. They are useful when you want a model ready before your first request, or want to see what is available.
 
### Request a model
 
`POST /v1/models/request`
 
Asks the network to load a model. If the model is already loaded or loading, the response reports its current status. Otherwise, loading starts immediately.
 
```python
import requests
 
response = requests.post(
    "http://localhost:64747/v1/models/request",
    json={
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_type": "chat",   # optional
        "time": 1800,           # lease duration in seconds (optional)
        "payment": 0            # reserved for future paid jobs (optional)
    }
)
 
print(response.json())
```
 
Example response:
 
```json
{
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "status": "inactive",
    "message": "Model Qwen/Qwen2.5-7B-Instruct not found. Loading has been initiated."
}
```
 
The `status` field is one of:
 
| Status | Meaning |
|--------|---------|
| `"active"` | The validator and all workers have fully loaded the model. It is ready for requests. |
| `"loading"` | The job exists, but one or more workers are still loading modules. |
| `"inactive"` | The model was not found, and loading has been initiated. |
 
> Paid jobs for private model access are not yet available. All models are currently loaded as public shared resources.
 
### Check model status
 
`GET /v1/models/status?model_name=<hf_name>`
 
Returns the current status of a model. Pass the name as a query parameter, since model names contain slashes and don't route cleanly as path segments.
 
```python
import requests
 
response = requests.get(
    "http://localhost:64747/v1/models/status",
    params={"model_name": "Qwen/Qwen2.5-7B-Instruct"}
)
 
print(response.json())
# {"model_name": "Qwen/Qwen2.5-7B-Instruct", "status": "active", "message": "..."}
```
 
### List available models
 
`GET /v1/models/available`
 
Returns every model that is currently active and ready to serve requests.
 
```python
import requests
 
response = requests.get("http://localhost:64747/v1/models/available")
print(response.json())
# {"active_models": ["Qwen/Qwen2.5-7B-Instruct", ...]}
```
 
### Demand statistics
 
`GET /v1/models/demand`
 
Returns API request demand ranked by model popularity. This is useful for deciding which models are worth keeping loaded.
 
```
GET /v1/models/demand?days=30&limit=10
```
 
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `days` | int | 30 | 1-90 | Lookback window in days |
| `limit` | int | 10 | 1-50 | Maximum number of models to return |
