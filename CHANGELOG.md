# Changelog

## 0.4.1 — unreleased (security release; date set when tagged)

Upgrade notes and the full list of changes: [docs/release-0.4.1.md](docs/release-0.4.1.md).

- **Behaviour change (control plane, authentication):** the control plane fails closed. With no `CONTEXT_RUNTIME_API_KEY` (or `AGENTIC_OS_API_KEY`) set, every route except `GET /health` answers `503`; with a key set, a missing or wrong credential is `401`. Before, the mutating routes were open when no key was set. The routes that were open with or without a key (`/approvals`, `/jobs/*`, `/status`, `/api/fleet`, `/notify/status`, `/agent/policy/*`, `/modules`, the dashboards, `/vibex/*` reads, `/librechat/retrieve|compare|explain|image|policy`, `/v1/*`) now need a credential. `GET /health` returns `{"status": "ok"}` only (no module count). `/docs`, `/redoc` and `/openapi.json` are not served. Every route has an access class (public, reader, admin); see SECURITY.md.
- **Behaviour change (tenant keys):** the `reader` routes accept the control-plane key or a key from `CR_TENANT_KEYS`; callers without a credential no longer reach the default tenant. `CR_ALLOW_ANONYMOUS_TENANTS=1` now lets every authenticated caller use every tenant and no longer admits callers without a credential.
- **Behaviour change (ingestion):** `POST /librechat/ingest` takes a corpus identifier (or a path relative to the root) and only indexes inside `CONTEXT_RUNTIME_CORPUS_ROOT`; with no root configured it answers `503`. Absolute paths, traversal, encoded or Windows-style separators and symlinks that leave the corpus are refused.
- Credentials are compared with `hmac.compare_digest` (control-plane key and tenant keys).
- `Dockerfile` (and `deploy/proxmox-demo/Dockerfile.contextruntime`): installs `agent-harness` at the commit `pyproject.toml` pins, verified by the archive's sha256, so the image starts; before, the image failed at start with `ModuleNotFoundError: agent_harness`. `tests/test_container.py` builds the image and checks 503, 401 and 200.
- Deployment templates no longer ship a default key (`context-runtime`); the proxmox demo proxy is an allowlist of the reader routes, adds a read-only tenant key and answers every other `/librechat/*` and `/v1/*` path with 404; `tests/test_demo_proxy.py` runs it end to end with Caddy (skipped when caddy is absent).
- **Behaviour change (MCP):** stdio MCP servers no longer inherit the whole parent environment. `MCPClient.stdio` passes an allowlisted set of
  variables plus the explicit `env` mapping; secrets in the parent environment do not reach the server. Servers that relied on an inherited
  variable (for example an API key) need it in `env`, or `inherit_env=True`, or `CONTEXT_RUNTIME_MCP_INHERIT_ENV=1`. The bundled `deploy_scan` mount now passes its own `SCAN_*` settings (and `DOCKER_HOST`, `DOCKER_CONTEXT`) explicitly. See SECURITY.md.

## Merged from the w4 line (formerly v4), 2026-10-27

- Lossless transcript compaction for agent loops: `compression/transcript.py`, `compression/contract.py`, `integrations/agent_harness.py` (`ContextRuntimeManager`; offline BM25 archive search by default, `redevops-rag` optional via `use_redevops_rag=True`) and its tests (#48, #57).
- The tool contract now comes from `agent-harness` (pinned to a commit): `tools/base.py` re-exports it (#48).
- `[rag]` extra pinned to `redevops-rag` `3a6c253` (was the `v0.4.0` tag).
- Test dependencies declared as the `test` extra; CI installs it and runs the whole suite; gitleaks secret scan in CI; HippoRAG and graphiti pinned to commits.
- Docs: public-clarity pass (STATUS with a versions section, evidence cards, docs check), `docs/transcript-contract.md`.
- Version declared as 0.4.0 (`pyproject.toml`, `context_runtime.__version__`), matching the `v0.4.0` tag.
- Already on `main` and unchanged by this merge: the control-plane hardening of #44 and the licence of #51 (via #60).

## 0.3.x — unreleased

The 0.3.x line is additive and default-serial throughout: each new path is opt-in and
byte-for-byte the previous behavior unless enabled. This "Unreleased" section accumulates
the v0.3.x work staged across the coordinated-release feature branches, so both the
geospatial and the concurrency entries appear here on every 0.3.x branch (which is also
what keeps this file merge-clean).

### Intelligence Gateway integration (opt-in)

- `context_runtime/integrations/intelligence_gateway.py`: a goal can declare a gateway need (`gateway_source(...)`); `GatewayCandidateGenerator` offers gateway retrieval plans only when the
  gateway says the capability is served and entitled, a quote is available and fits `max_cost_usd`, and the goal is not `restricted` (nor `internal`, unless opted in).
  `GatewayRetriever` runs only needs the planner recorded, re-quotes, refuses a price above the planned quote, and sends the quoted price as the gateway's ceiling. Only the capability, subject
  references, fields and purpose leave: never the goal text, a provider list or a jurisdiction. Any refusal or outage yields no gateway evidence. Goals without a gateway source are unchanged.
- Known gateway-side gap, pinned by a contract test: the stock gateway has no answer synthesizer, so `/v1/resolve` charges and returns an empty answer.

### Control plane hardening

- Module proxy validates paths; tenants are bound to credentials (`CR_TENANT_KEYS`, `CR_ALLOW_ANONYMOUS_TENANTS`).

### Geospatial capabilities + zoning intelligence

Geospatial support as a **Runtime capability, not a separate Geospatial Runtime**: a parcel's geometry,
CRS, jurisdiction and temporal state become typed evidence, and spatial operations become capabilities
the planner selects, composes, verifies and governs. Fully additive and offline; no existing path changes.

- **Typed spatial evidence** (`context_runtime.geospatial.contracts`): `GeoRef` (geometry identity via a
  canonical, hashable `geometry_hash`; explicit CRS; jurisdiction; valid/observed time) extends — never
  replaces — the evidence-native model. Domain vocabulary: `ParcelEntity`, `UseDisposition`
  (PERMITTED / CONDITIONAL / SPECIAL_EXCEPTION / PROHIBITED / **UNKNOWN**), `LandUseConstraint`, a
  normalized use ontology.
- **CPU-authoritative spatial engine** (`context_runtime.geospatial.engine`): pure-Python, exact,
  dependency-free point-in-polygon, polygon intersection, area, centroid, distance, and centroid spatial
  join. Binary ops **refuse a silent CRS mismatch** (raise so the planner inserts an explicit REPROJECT).
  Heavier backends (shapely/PostGIS/DuckDB-Spatial/GPU) are planner-selectable on measured crossover —
  the accelerator changes latency, never the answer.
- **Zoning-intelligence tenant** (`context_runtime.integrations.zoning_intelligence` +
  `examples/zoning_intelligence.py`): the geospatial reference benchmark. The Runtime learns *which
  evidence to acquire* (Regrid · ATTOM · official GIS · ordinance) per use-difficulty bucket. A
  deterministic-first, fail-safe resolver reconciles evidence and only concludes PERMITTED when the
  evidence is sufficient — making **false-permitted a structural impossibility** for reconciled bundles
  (the blocking SLO). Measured: learned **+0.894 reward at 1.4× lower evidence cost** than
  always-thorough, **0 false-permits vs 1** for single-provider. Includes use-first land search,
  dependency-scoped incremental recomputation, and population-governance drift detection.

### Opt-in reasoner + retrieval concurrency

The independent calls a plan already makes now overlap instead of running strictly one after another.

- **Reasoner fork/join** (`context_runtime/reasoner/_concurrency.py`): a mixture strategy's independent
  model calls — `debate` passes, `plan_worker_critic` workers, self-consistency samples — fan out
  concurrently when `CR_REASONER_CONCURRENCY>1` (default 1 = serial). Order-preserving, so the
  judge/critic/vote reconciliation sees identical inputs in identical order — the answer is unchanged,
  only wall-clock. Concurrency is *how* a strategy's calls run, never *which* calls it makes.
- **Retrieval overlap** (`context_runtime/_parallel.py`): the hybrid BM25 ∥ vector legs and the
  two-stage stage-1 recalls overlap when `CR_RETRIEVAL_CONCURRENCY>1` (default 1 = serial),
  order-preserved so RRF fusion is deterministic regardless of the setting.

## 0.2.0 — freshness sourced from evidence (v0.2.x final stabilization)

Correctness/completeness fix to functionality v0.2.x already advertised. The freshness *scoring*, the
`REFRESH` verdict, and the EXPLAIN lineage *renderer* shipped in Slice 5 — but freshness was never
**sourced** from the evidence being evaluated, the REFRESH gate was off the normal serving path, and
the EXPLAIN lineage was not populated by any production path. This release wires all three end-to-end.
Entirely opt-in: with no `FreshnessPolicy`, freshness is `1.0` and every path is byte-for-byte the
legacy one.

**What is now true**

- **`Hit` carries evidence identity** — `version`, `content_hash`, `observed_at` flow from the
  retriever. The redevops-rag binding (`store_redevops`) supplies them from the canonical `EvidenceRef`
  that redevops-rag 0.2.1 now emits; `store_inmemory` carries them when the corpus provides them.
- **Freshness derived from evidence** (`context_runtime.freshness`): `FreshnessPolicy`
  (`age_decay` | `ttl`) turns a hit's `observed_at` (vs an `as_of` reference) into a `[0,1]` score;
  `score_hits` aggregates (worst-evidence governs). Unknown timestamps → `1.0` (never penalized).
- **REFRESH on the normal serving path** — `ContextRuntime(..., freshness_policy=...)` computes
  freshness from the retrieved evidence in `run()`/`execute()`, records it on `PlanScore.freshness` and
  `RunResult.freshness`, and — when the evidence is staler than `min_freshness` — declines to serve,
  returning `RunResult(refresh=True, refresh_reason=...)` before the reason step. No longer a
  test-only `BanditOptimizer` configuration.
- **EXPLAIN populated, not merely rendered** — `freshness.lineage_from_hits(...)` /
  `explain_hit_row(...)` build the lineage from the actual served evidence, so EXPLAIN names the exact
  source ref / version / content-hash / freshness used.

**Backward compatibility** — `Hit`/`RunResult` gained fields with defaults; `run`/`execute` gained an
optional `as_of`. With no `FreshnessPolicy` the runtime behaves identically to 0.2.0 (verified). New
tests: `tests/test_freshness_sourced.py` (the Slice-5 primitive tests remain in
`tests/test_freshness_slice5.py`).
