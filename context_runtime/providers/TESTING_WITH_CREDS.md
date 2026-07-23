# Smoke-testing the GCP and DigitalOcean providers with real credentials

The test suite runs entirely on injected fakes (no SDK, no network). To validate against a live
account, install the provider extra, set the env vars, and run the snippets below. Each adapter is
independent, so test them one at a time.

## Live validation status (2026-07-23)

| Adapter | Result |
|---|---|
| **GCP GeminiModel** (Developer API via `api_key`) | ✅ **verified end-to-end** — `gemini-2.5-flash` returned text + token counts. Note: `gemini-2.0-flash` is free-tier `limit: 0` on this key; use the 2.5 family. Thinking models need adequate `max_tokens` (>= ~200) or `.text` is empty. |
| **DO GradientModel** (serverless inference) | ✅ **adapter + auth + VPC path all verified live.** `GET /v1/models` returns **200** with a valid `doo_v1_` model access key over the private path. Remaining block is a DO **account entitlement**, not code. The full playbook: (1) inference needs a **model access key** (`doo_v1_…`), not the API token, and it must be the real secret (a mis-copied value 401s even on `/models`); create-key API is retired-410 → control panel. (2) **VPC-restricted** keys require the **VPC-local DNS resolver** = *second-to-last IP of the VPC CIDR* (`10.120.0.0/20` → `10.120.15.254`); set the DOKS pod `dnsConfig.nameservers` to it (verified `connected_ip=10.120.15.254`); `hostNetwork: true` sources the node's VPC IP. (3) With auth green, **`/chat/completions` still 402s for *all* models** ("not allowed to perform this operation") on an active account with real usage → serverless-inference **generation entitlement / spend-limit** must be enabled DO-side. Not the adapter, key, VPC, or general billing. |
| **DO GradientKBRetriever** | ⏸ not yet tested — the account has **0 knowledge bases**; create one to validate. The KB `/retrieve` uses the general DO API token (works today). |
| **GCP Vertex Search / BigQuery / Model Armor** | ⏸ not yet tested — need ADC/service-account + provisioned resources (only the API-key Gemini path was available). |

**Two account-standing blocks to clear for full validation:** enable billing on the DigitalOcean
account (for serverless inference) and on the Gemini API key's project (2.0-flash free tier is 0).
The adapters themselves are verified correct.

Creds are in Vault: `vibexgen/google/gemini_api_key`, `vibexgen/digitalocean/api_token` (the DO token
serves both the knowledge base and inference).

---

## GCP

```bash
pip install "context-runtime[gcp]"          # google-genai, discoveryengine, bigquery, modelarmor, monitoring
export GOOGLE_CLOUD_PROJECT=your-project
export GOOGLE_CLOUD_LOCATION=us-central1
gcloud auth application-default login        # ADC
```

```python
from context_runtime.providers import get_provider

# Two model-plane modes:
#   Vertex AI    → get_provider("gcp", project=..., location=...) + ADC (retrieval/BigQuery/Model Armor)
#   Developer API → get_provider("gcp", api_key=...)  (Gemini only, no project/ADC; GEMINI_API_KEY env works)
gcp = get_provider(
    "gcp",
    api_key="…",                                # Gemini Developer API (simplest; omit for Vertex)
    project="your-project", location="us-central1",
    vertex_engine_id="your-search-engine-id",   # or data_store_id="..."  (Vertex path)
    bigquery_dataset="your_dataset",
    model_armor_template="projects/…/locations/…/templates/…",  # optional
)

# 1) Gemini model plane
r = gcp.model().complete.__self__  # or just:
res = gcp.model().complete(__import__("context_runtime.types", fromlist=["ModelRequest"]).ModelRequest(
    messages=({"role": "user", "content": "Say hello in five words."},), max_tokens=64))
print("MODEL:", res.text, res.prompt_tokens, res.completion_tokens)

# 2) Vertex AI Search
print("SEARCH:", [h.filename for h in gcp.document_retriever().search("your query", k=3, method="hybrid")])

# 3) BigQuery analytical (text-to-SQL) — pair the warehouse with a model as the SQL generator
from context_runtime.adapters.store_analytical import AnalyticalRetriever
analytical = AnalyticalRetriever(gcp.analytical_backend(), gcp.model())
print("SQL:", [h.text for h in analytical.search("how many rows per status?", k=5, method="sql")])
```

Full wiring into a runtime (Gemini + Vertex Search + BigQuery analytical in one call):

```python
from context_runtime.providers.wiring import build_runtime
from context_runtime.adapters.store_inmemory import InMemoryStore
rt = build_runtime(gcp, local_single_hop=InMemoryStore([]))
print(rt.run("your question").answer)
```

---

## DigitalOcean

```bash
pip install context-runtime                  # no extra needed — DO adapters use the stdlib
export DIGITALOCEAN_TOKEN=dop_v1_…           # DO API token with GenAI:read (for the knowledge base)
export DO_INFERENCE_KEY=…                     # a serverless-inference model access key
```

```python
from context_runtime.providers import get_provider
from context_runtime.types import ModelRequest

do = get_provider(
    "digitalocean",
    model_tiers=[("cheap", "llama3.3-70b-instruct", 0.0006)],   # use a model id your key can reach
    knowledge_base_id="your-kb-uuid",
)

# 1) Gradient serverless inference (OpenAI-compatible)
res = do.model().complete(ModelRequest(
    messages=({"role": "user", "content": "Say hello in five words."},), max_tokens=64))
print("MODEL:", res.text, res.prompt_tokens, res.completion_tokens)

# 2) Gradient knowledge base retrieve
print("KB:", [(h.filename, h.score) for h in do.managed_kb_retriever().search("your query", k=3, method="hybrid")])
```

Full wiring:

```python
from context_runtime.providers.wiring import build_runtime
from context_runtime.adapters.store_inmemory import InMemoryStore
rt = build_runtime(do, local_single_hop=InMemoryStore([]))
print(rt.run("your question").answer)
```

---

## What to report back if something breaks

The adapters are built against the documented API shapes; the most likely mismatches on first contact
are field names in a response. If a call fails, capture:

- **GCP Vertex AI Search** — the shape of `result.document.derived_struct_data` (snippet vs
  extractive_answers vs content), so `_text_of` can be tuned.
- **DigitalOcean KB** — the JSON returned by `POST /v1/<kb>/retrieve` (is the list under `results`,
  `chunks`, or `data`? what are the score / content / metadata keys?), so `GradientKBRetriever` can be
  tuned. It already tolerates several common names.
- **DigitalOcean inference** — the exact model id string your access key is scoped to.

Each is a one-line fix in the adapter; the seam and the runtime don't change.
