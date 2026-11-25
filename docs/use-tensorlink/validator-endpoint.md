# Run a Validator and API Endpoint

A validator coordinates jobs and can expose the [Inference API](api.md) over HTTP. Run one to host a private endpoint for your own apps, to coordinate a private cluster, or to bridge a private cluster to the public network.

Read [Architecture](../architecture.md) first if you're new to node roles.

---

## Requirements

- Linux or macOS (Windows through WSL), Python 3.10+
- The node package from the [GitHub releases page](https://github.com/tensorlink-lab/tensorlink/releases) (extract it and edit `config.json`, as in [Run a worker](worker-quickstart.md#step-2-download-the-node-package))
- Workers to run models on. They can be the same machine or others on your network. See [Run a worker](worker-quickstart.md)

---

## Private validator with an API endpoint

Set `endpoint` to `true` to serve the API. List your workers in `priority_nodes`:

```json
{
  "node": {
    "type": "validator",
    "mode": "private",
    "endpoint": true,
    "endpoint_url": "0.0.0.0",
    "endpoint_port": 64747,
    "priority_nodes": [
      ["192.168.1.101", 38752],
      ["192.168.1.102", 38753]
    ]
  },
  "ml": {
    "trusted": true
  }
}
```

- `endpoint_url: "0.0.0.0"` serves the API on all network interfaces. The default bind address is `127.0.0.1`, which is local only.
- `endpoint_port` defaults to `64747`.
- `trusted: true` allows custom model execution. Use it only on infrastructure you fully control.

Start it with `./run-node.sh`, then check the API:

```bash
curl http://localhost:64747/v1/models/available
```

An empty list is normal until you request a model. Load one with [`POST /v1/models/request`](api.md#request-a-model), or send a chat request and retry after the `503`.

Workers in a private cluster use a minimal config. See [Configuration](configuration.md#private-worker).

---

## Hybrid: private cluster plus the public network

To let work overflow to community GPUs when local capacity runs out, run the validator in public mode alongside your private workers. See the topology in [Architecture](../architecture.md#hybrid-public-and-private).

---

## Enable web search

Chat requests can set `web_search` (see [Web search](api.md#web-search)) once the node has a search provider. Configure one with environment variables before starting the node:

| Variable | Description |
|---|---|
| `TENSORLINK_SEARXNG_URL` | Base URL of a SearXNG instance with the JSON format enabled (`search.formats` in its `settings.yml`) |
| `TENSORLINK_BRAVE_API_KEY` | Brave Search API key |
| `TENSORLINK_SEARCH_PROVIDER` | Optional: `searxng` or `brave`, to force one when both are configured |
| `TENSORLINK_SEARCH_RPM` | Optional: searches allowed per minute (default `60`) |
| `TENSORLINK_SEARCH_TIMEOUT` | Optional: provider timeout in seconds (default `8`) |

Without a provider, requests that set `web_search` return `501`. Search only talks to the provider you configure. It never fetches result pages itself.

---

## Secure the endpoint

The API has no built-in authentication. Anyone who can reach the port can use your GPUs.

- **Use a firewall.** Binding to `0.0.0.0` exposes the API on every interface. Restrict it to trusted IP ranges.
- **Add authentication at a reverse proxy** if you expose the endpoint beyond your LAN.
- **Browser access.** Only `https://tensorlink.io` and `localhost` origins are allowed by CORS. Other web apps must call the API from a server.

See [Configuration → Security](configuration.md#security).

---

## Connect Python clients

Point a user node at your validator. See [Private clusters](private-clusters.md#connect-to-a-private-cluster).
