# Configuration

Reference for `config.json`, which controls a node's role, networking, rewards and security. For step-by-step setup, see [Run a worker](worker-quickstart.md) and [Run a validator](validator-endpoint.md).

---

## Field reference

### `node`

| Field | Values | Description |
|---|---|---|
| `type` | `"worker"` \| `"validator"` | The node's role. See [Architecture](../architecture.md#node-types) |
| `mode` | `"public"` \| `"private"` | Connect to the public network, or run an isolated cluster |
| `endpoint` | `true` \| `false` | Serve the HTTP API. Applies to validators |
| `endpoint_url` | IP address | Bind address for the API. `0.0.0.0` serves all interfaces. The default is `127.0.0.1` |
| `endpoint_port` | port number | API port. The default is `64747` |
| `priority_nodes` | list of `[IP, port]` | Nodes to connect to explicitly. Empty for public workers |
| `logging` | e.g. `"INFO"` | Log level |

### `crypto`

| Field | Description |
|---|---|
| `address` | Your public wallet address (Ethereum/Base) for rewards. Never put a private key here |
| `mining` | `true` runs `mining_script` during idle periods |
| `mining_script` | Path to the executable to run when `mining` is on |
| `seed_validators` | Bootstrap validators, as `[host, port, node_id]` entries |

### `ml`

| Field | Description |
|---|---|
| `trusted` | `false` accepts only verified Hugging Face models. `true` allows custom, user-supplied models and arbitrary code execution |
| `max_vram_gb` | Cap on GPU memory the node may use |

---

## Examples

### Public worker

Earns rewards on the public network. This is the same config used in [Run a worker](worker-quickstart.md#step-3-configure-your-node).

```json
{
  "node": {
    "type": "worker",
    "mode": "public",
    "endpoint": false,
    "priority_nodes": [],
    "logging": "INFO"
  },
  "crypto": {
    "address": "0xYourWalletAddress",
    "mining": false,
    "mining_script": "path/to/mining.executable",
    "seed_validators": [
      ["tensorlink.ddns.net", 38752, "58ef79797cd451e19df4a73fbd9871797f9c6a2995783c7f6fd2406978a2ba2e"]
    ]
  },
  "ml": {
    "trusted": false,
    "max_vram_gb": 24
  }
}
```

### Private validator

Coordinates a private cluster and serves the API. See [Run a validator](validator-endpoint.md).

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

### Private worker

A minimal worker for a private cluster. Workers discover validators on private networks automatically.

```json
{
  "node": {
    "type": "worker",
    "mode": "private"
  },
  "ml": {
    "trusted": true,
    "max_vram_gb": 24
  }
}
```

---

## Security

**Never enable `trusted: true` on public nodes.** It allows arbitrary code execution. Enable it only on infrastructure you fully control. Public workers must use `trusted: false`.

**Firewall private endpoints.** Binding to `0.0.0.0` exposes the API on every network interface. Use firewall rules to restrict access to trusted IP ranges.

**Add authentication yourself.** The API doesn't check API keys. If you expose it beyond a trusted network, put it behind a reverse proxy that handles authentication.

**Protect your wallet keys.** `config.json` needs only your public wallet address. Never share or store private keys in it.

**Browser access is limited by CORS.** Only `https://tensorlink.io` and `localhost` / `127.0.0.1` origins can call the API from a browser.
