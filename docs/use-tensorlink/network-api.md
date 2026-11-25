# Network Endpoints

Read-only endpoints for network status, history and node information. These are for dashboards and monitoring. For running models, see the [Inference API](api.md).

The base URL is the same as for the inference API (`http://localhost:64747` by default).

| Method | Path | Description |
|---|---|---|
| `GET` | `/stats` | Current network and node status |
| `GET` | `/network-history` | Historical network metrics |
| `GET` | `/proposal-history` | Historical proposals from the node's archive |
| `GET` | `/node-info` | Information about one node |
| `GET` | `/claim-info` | Reward claim data for a worker |

---

## Network stats

`GET /stats` returns the current network and node status.

```python
import requests

print(requests.get("http://localhost:64747/stats").json())
```

---

## Network history

`GET /network-history` returns historical network metrics.

```
GET /network-history?days=30&include_weekly=false&include_summary=true&include_device=false
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `days` | int | `30` | Lookback window, 1–1093 |
| `include_weekly` | bool | `false` | Include weekly breakdowns |
| `include_summary` | bool | `true` | Include summary statistics |
| `include_device` | bool | `false` | Include device-level data |

---

## Proposal history

`GET /proposal-history` returns historical proposals from the node's archive cache.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `limit` | int | `30` | Number of proposals, 1–180 |

---

## Node info

`GET /node-info?node_id=<pubKeyHash>` returns information about a node in the network.

```json
{
  "pubKeyHash": "<node id>",
  "type": "W",
  "lastSeen": 1731900000,
  "data": {}
}
```

`type` is the node's role (`"V"` for validator, `"W"` for worker). An unknown `node_id` returns an empty object (`{}`).

---

## Claim info

`GET /claim-info?node_address=<address>` returns reward claim data for a worker's wallet address.

You can also view and claim rewards in the [Smartnodes dashboard](https://smartnodes.ca/app).
