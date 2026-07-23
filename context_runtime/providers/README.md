# Cloud providers

Managed clouds plug into Context Runtime **behind the runtime's existing plugin Protocols** — the
kernel never imports a cloud SDK. AWS, GCP and DigitalOcean ship today; Azure is the next drop-in.
This directory is the mechanical form of the audit's finding that *every integration is an adapter, not
a kernel change.*

| Provider | model | document retrieval | analytical | guardrail | telemetry |
|---|---|---|---|---|---|
| `aws` | Bedrock (`converse`) | OpenSearch + Bedrock KB | Athena | Bedrock Guardrails | CloudWatch |
| `gcp` | Gemini (google-genai) | Vertex AI Search | BigQuery | Model Armor | Cloud Monitoring |
| `digitalocean` (`do`) | Gradient inference (OpenAI-compat) | Gradient knowledge base | — | — | — |
| `azure` | *planned* | *planned* | *planned* | *planned* | *planned* |

DigitalOcean is the lean platform: it ships no serverless SQL warehouse and its guardrails are applied
inside an agent (not as a standalone check), so `analytical_backend()` / `guardrail()` /
`identity_broker()` honestly return `None` and the caller falls back to the in-tree defaults.

## The seam (`base.py`)

A provider subclasses `CloudProvider` and returns objects that satisfy plugin Protocols the runtime
already knows how to use, or `None` when it doesn't offer that capability (the caller falls back to the
in-tree default):

| Factory | Returns | Satisfies |
|---|---|---|
| `model()` | managed model plane (Bedrock) | `ModelPlugin` |
| `document_retriever()` | managed search (OpenSearch) | `RetrieverPlugin` |
| `managed_kb_retriever()` | managed RAG KB (Bedrock KB) | `RetrieverPlugin` |
| `analytical_backend()` | warehouse for text-to-SQL (Athena) | `WarehouseBackend` |
| `guardrail()` | content safety (Bedrock Guardrails) | `Guardrail` |
| `identity_broker()` | delegated tokens (AgentCore Identity) | `IdentityBroker` |
| `telemetry_reader()` | ops telemetry (CloudWatch) | `TelemetryReader` |

The four small neutral Protocols (`Guardrail`, `IdentityBroker`, `TelemetryReader`, `WarehouseBackend`)
are the only new interfaces; everything else reuses `plugins/base.py`.

## Using a provider

```python
from context_runtime.providers import get_provider
from context_runtime.providers.wiring import build_runtime
from context_runtime.adapters.store_inmemory import InMemoryStore

aws = get_provider("aws", knowledge_base_id="KB123", athena_database="lake",
                   athena_output="s3://results/", guardrail_id="g-1", region="us-east-1")

rt = build_runtime(aws, local_single_hop=InMemoryStore(docs), learning=True)
rt.run("how many invoices are open per customer?")
```

`build_runtime_kwargs` slots the cloud's pieces in where offered and the in-tree defaults everywhere
else. Every AWS adapter lazy-imports `boto3` (the `aws` extra) and accepts an **injected client**, so
the base runtime and the test suite never require boto3.

## Adding a provider (Azure / GCP / DigitalOcean)

1. `providers/<cloud>/` with adapters implementing the pieces the cloud offers — each satisfying the
   Protocol in the table above (e.g. `AzureOpenAIModel(ModelPlugin)`, `AzureSearchRetriever`,
   `SynapseBackend(WarehouseBackend)`, `AzureContentSafety(Guardrail)`).
2. A `<Cloud>Provider(CloudProvider)` returning them (or `None`).
3. `register()` calling `register_provider("<cloud>", <Cloud>Provider)`; wire it into
   `get_provider`'s lazy self-registration, and add an `<cloud>` extra in `pyproject.toml`.

Nothing above the seam changes — not the planner, the cost model, the router, the reasoners, or the
learning loop.
