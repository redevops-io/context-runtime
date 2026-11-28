# Changelog

## Unreleased

Notes: [docs/exactness-and-completeness.md](docs/exactness-and-completeness.md).

- **Fix (an exact aggregate was costed as a temporal or multi-hop question).** `HeuristicEstimator` tested the bucket before the analytical representation, so `hybrid` top-k retrieval outranked `sql` for questions that need a computed answer. The representation is now evaluated first. Behaviour change on the default path.
- **Fix (the retriever and the model were shown the plan-cache key as the question).** The normalised key is sorted and de-duplicated. Exactness-constrained requests, and any request with `Config.natural_question=True`, now carry the user's own question to the retriever and the reasoners. Other requests are unchanged.
- **Feature:** `Constraints.require_exact` / `require_complete`. Feasibility (a provider that declares the `exact` capability) is checked before cost; with none, `NoEligiblePlan` / a typed `Limitation`, never a silent top-k answer. Defaults are off and plan-cache keys are unchanged when they are off.

## 0.5.2 — 2026-11-25

Notes: [docs/release-0.5.2.md](docs/release-0.5.2.md).

- **Self-Improvement Memory selection** (`context_runtime.learning.sim_select`): when an `agent-harness.sim/1` record does not fit a budget, a view that keeps every problem visible, every attempt and measured field of a shown problem, and the regression evidence that bears on the decision; the complete record is returned when it fits. No claim that a view is as good as the complete record.

## 0.5.1 — 2026-11-25

Notes: [docs/release-0.5.1.md](docs/release-0.5.1.md).

- **Fix:** an answer that failed the generation check was recorded in the learning stream as verified. (#102)
- **Security fix:** the compaction archive wrote whole tool outputs readable by every local account (`0755` directory, `0644` files). New directories are `0700` and files `0600`; existing ones keep their modes. (#102)
- **CI:** the geo demo image publishes from a version tag again; the `v0.5.0` tag published none. (#103)

## 0.5.0 — 2026-11-25

Upgrade notes: [docs/release-0.5.0.md](docs/release-0.5.0.md).

- **Security fix (a policy-rejected plan was executed).** When no candidate was both within the hard limits and permitted, `KnapsackOptimizer` and `BanditOptimizer` fell back to all candidates and served a rejected plan as feasible; a candidate over a hard limit was never shown to the policy. The fallback is now limited to permitted candidates, and `PolicyDenied` is raised when there are none. **Breaking:** callers with a policy must handle it. (#99)
- **Security fix (a cached plan was replayed against the policy).** `ContextRuntime.plan()` returned the cached plan whatever the policy now said, including one cached for another identity. A cached plan is replayed only if the policy permits it now. (#100)
- **Learning:** a ledger with provenance for the shared bandit, per-context rollback and reset, a kill switch, and a registry of every learner; a corrupt policy file is set aside, not overwritten ([docs/LEARNERS.md](docs/LEARNERS.md)). (#98)
- **Fix:** a reward that is not a finite number is refused, and a policy file poisoned by one is healed on load. (#97)
- **Fix (CrowdSec decisions tool showed simulated bans when the configured CrowdSec could not be read).** `crowdsec_decisions` fell back to its built-in sample feed on any failure of the live local API and answered `ok` with three made-up banned addresses. A configured CrowdSec that is unreachable, refuses the key or answers something that is not a list of decisions is now an error with no decisions; "no decisions" (`null`) is an empty list. The sample feed is used only where no CrowdSec is configured, and every row and line of it says `simulated`.
- **Vulnerability feeds: a failed feed can be told from an empty one.** `fetch_nvd`, `fetch_osv`, `fetch_osv_by_id`, `fetch_snyk` and `collect` take `errors=` (a list) and add one entry per feed that could not be read. Without it they answer `[]` as before, which is also what "no known vulnerabilities" looks like. Advisory and organisation ids are written into feed URLs as one path segment.

- **Security fix (`pure_read` could be claimed by code that leaves the process).** The check that proves an injected tool inert treated a name bound by an `import` inside the callable as visible, so `import subprocess` above `subprocess.getoutput(...)`, `httpx.request("POST", ...)`, `sqlite3.connect(...).execute("DELETE ...")`, `os.kill(...)` and a computed callee (`getattr(x, name)(...)`) were all registered as `read`. An import now makes its name visible only when the module is one of `INERT_MODULES`; reflective builtins (`getattr`, `__import__`, `eval`, `exec`, `open`, ...) and computed callees are opaque. A callable refused by this needs `delegated(...)`, as before for any other opaque callee. `tests/test_pure_read_local_imports.py`.
- **Security fix (runtime scan: the host that was checked was not always the host that was scanned).** The scope check read the host of `allowed.example:80@other.example` as `allowed.example`, while nuclei and ZAP were given `http://allowed.example:80@other.example`, whose host is `other.example`. One parse now decides both. A target with user information, a backslash, a blank or control character, a scheme other than http(s), or a host that is not a DNS name or an IP address (a leading `-` would be read by a tool as an option) is out of scope and no process is started for it.
- **Security fix (Salesforce account lookup).** `search_account` put the domain or name it was given into the SOQL text unescaped; the domain comes from a tool call. Both are now escaped as literals (quotes, backslash, and the `LIKE` wildcards), and `get_account` / `update_account` refuse an id that is not a record id before any request.

- **Breaking change (tool authorization): a Context Runtime tenant no longer starts without an authorization plane.** Every integration tenant and the module fleet resolves an authorizer at construction (`context_runtime/tools/authz.py`): an explicit `authorizer=`, else a plane installed process-wide with `context_runtime.tools.set_default_authorizer(...)` (what `agentic_os.app_gate.install_gate` does), else an explicitly declared local authority — `local_trust=True` with a written `local_trust_reason`, or `CR_LOCAL_AUTHORITY="<why this process is trusted>"`. None of the three raises `RegistryConfigurationError` at construction instead of denying every call later; `CR_LOCAL_AUTHORITY=1` is refused, because a value that carries no reason is how an allow-all switch reaches a production manifest. This follows agent-harness #33 (a registry with no authorizer denies everything) and #34 (a tool with no declared risk class is permitted by no grant).
- **Behaviour change (risk classes are declared, never inferred).** `mount_mcp(..., actions={...}, default_action="write")` and `AgentConsole.tool(..., action=...)` take the class from the integrator; `readOnlyHint`/`title` pass through as annotations and relax the **approval** gate only. An MCP server is not allowed to grade its own risk. `AgentConsole.validate()` is the start-up check that refuses to serve a console with no plane.
- **Behaviour change (the module fleet):** standing a module up with no plane no longer kills the process — `Fleet.up()` follows `AgentConsole`'s rule (`on_missing="deny_later"`), because the embedding application installs its gate a moment after import and a container that never answers `/health` is worse than one that denies. It warns once, and every tool call stays denied (`missing_authorizer`) until a plane exists. **`/status` looks identical either way** — that is what the warning is for.
- **Behaviour change (fleet actions of discovered modules):** a `CATALOG` action without a recorded class is still a start-up error, because this file wrote those names. An action inferred from a module the registry discovered at run time (e.g. `sidekick.coder`) is now registered **unclassified** — matched by no grant, so denied — with one warning naming it. Before this it was impossible to run a custom module at all, and guessing `read` for an unread action is the failure this migration removes.
- **Pin:** `agent-harness` `6b8ba3a` → `7e422f541f` (fail-closed authorizer decisions + `ToolSpec.action`). `Dockerfile`, `deploy/geo-demo/Dockerfile.aks` and `deploy/proxmox-demo/Dockerfile.contextruntime` move with it; `tests/test_container.py` / `tests/test_geo_image.py` fail if they drift apart.

- **Fix (Windows):** `JsonlExporter` appends each trace through `locked_append` (inter-process and in-process lock, one write, explicit newline, no per-trace `fsync`). A bare `open(path, "a")` loses lines when writers overlap on Windows, where the C runtime appends as seek-then-write. `locked_append` gains `durable=False`. The exporter waits at most 1 s for the lock; if it cannot get it the trace is dropped and one warning is logged per exporter. A test fails on any new lock-free append writer in the repository's Python code. **Visible differences:** a `<name>.jsonl.lock` sidecar file (0600) is created next to the trace file; in a read-only directory, or one holding a sidecar owned by another user, `export` now raises `PermissionError` where the bare append worked (the runtime and `MultiExporter` swallow exporter errors, so the trace is lost; one warning is logged); on Windows each line now ends in LF, where text-mode append wrote CRLF, so a trace file written by an older version and then appended to can mix the two (`splitlines()` and `json.loads` accept both).

Native Windows and macOS portability of the core library. Details and the verification status of each item: [docs/native-portability.md](docs/native-portability.md).

- **Breaking change (corpus identifiers and policy scopes, all operating systems):** a corpus reference whose path segment is a Windows device name (`con`, `prn`, `aux`, `nul`, `conin$`, `conout$`, `com0`-`com9`, `lpt0`-`lpt9`, case-insensitive, with or without an extension such as `nul.txt`) is refused: `POST /librechat/ingest` answers `400 {"detail": "invalid corpus reference"}`. These names were accepted on Linux and macOS before. The policy store stores a scope with such a name under `<name>~` (for example `con~/app.jsonl`; `con` and `con_` stay separate files) and shortens file-name components over 128 characters (`...~<digest>`); rules saved under the old file names are not found and nothing is migrated automatically.
- **Behaviour change (`Archive`):** a default `Archive()` (no directory given) now lives in a `TemporaryDirectory` that is removed when the archive is garbage-collected or the interpreter exits. Before, the `mkdtemp` directory and the archived tool outputs survived the process. Pass an explicit `directory` to keep them; an explicit directory is never removed.
- Explicit `encoding="utf-8"` for every text-mode file access and text subprocess call (the locale code page is no longer used); `tests/test_encoding_explicit.py` and a ruff PLW1514 step guard the common call forms.
- `NamedTemporaryFile` that is reopened by name uses `delete_on_close=False`; the ZAP work directory is removed after the report is read; the `json.load(open(...))` handle leaks are fixed.
- CI: Windows and macOS (Apple Silicon) test lanes next to the Ubuntu job; the test workflow also runs for pull requests of any base branch.
- `iter_corpus_files(root)`: one deterministic enumeration (NFC, POSIX-style labels, component-wise code point order) for every corpus scan; for NFC names the order is what `sorted(Path.rglob("*"))` gave on Linux. **Behaviour changes:** a single file passed as the root is now indexed by the text stores (it returned nothing before); NFD file names get NFC labels, ids and `filename` values; Windows junctions and other directory reparse points are not entered and are refused by ingestion; files that share a name get the same chunk id as before, and one warning now says so (the id format is unchanged so existing indexes stay valid).
- New `context_runtime/platform_services.py`: per-user state, config and cache directories (macOS Application Support and Caches, Windows `%LOCALAPPDATA%`, XDG on Linux), `atomic_write`, `file_lock`, `locked_append`, `ensure_private_dir`; the rename-based stores and append-only logs use them. Containers keep their layout: in a container the historical locations (`.context-runtime`, `config.yaml`, `/data/context-runtime/policy`, `/data/market-radar/...`) apply even when `CONTEXT_RUNTIME_HOME` is set (it is `/data` in the compose files); an item's own variable (`POLICY_DIR` ...) still wins. Detection recognises Docker/Podman marker files, Kubernetes, ECS/Fargate, Cloud Run and `/proc/1/cgroup`; `CONTEXT_RUNTIME_CONTAINER=1|0` overrides it.
- **Behaviour change (outside containers):** state, config and caches default to the platform locations. A non-empty `.context-runtime` (or a `config.yaml`) in the working directory is still used, even if the platform location exists, and one warning names both paths and how to migrate; nothing is moved automatically.
- **Behaviour change (permissions):** state files written through `atomic_write`, `locked_append` and the lock files are created 0600, and directories created by `ensure_private_dir` 0700 (the process umask, typically 0644 / 0755, applied before). An existing file keeps its mode and a read-only target is refused as `write_text` did; a symlink at the target is written through. Set `CONTEXT_RUNTIME_SHARED=1` to get the umask defaults back for a directory shared with other users or containers.
- **Behaviour change (locking):** `RuleStore` read-modify-write and the other stores' logs take an inter-process lock. On a file system that cannot lock (`ENOLCK`, `ENOTSUP`, `EOPNOTSUPP`) `CONTEXT_RUNTIME_LOCKING=best-effort` logs one warning and serialises threads of the process only; `strict` raises. Unset, the policy store (governance data) is strict and the other stores best-effort.
- **Behaviour change (DuckDB `fts`):** `DuckDBStore` no longer runs `INSTALL fts` by default. It loads a configured (`CONTEXT_RUNTIME_DUCKDB_FTS_EXTENSION`, `CONTEXT_RUNTIME_DUCKDB_EXTENSION_DIR`) or already installed extension, or `CONTEXT_RUNTIME_DUCKDB_INSTALL_FTS=1` allows a one-time download. Without the extension it logs one warning and uses an APPROXIMATE in-memory lexical fallback that is not score-compatible with the engine: no stop-word or digit removal, no stemming (`the` and `404` match, `errors` does not match `error`), other score scale.
- `ImageRetriever` falls back to exact cosine search when turbovec rejects the embedding width (turbovec needs a multiple of 8); before, such a width raised `ValueError` whenever turbovec was installed.
- CI additions: a `macos-26` Apple Silicon lane next to `macos-15`; the Windows and macOS lanes install a pinned, SHA-512-verified caddy so the demo-proxy tests run there; every test run ends with a skip summary that marks `UNVERIFIED` items.
- Docs: [docs/native-portability.md](docs/native-portability.md) (breaking changes, directories, DuckDB extension provisioning, Windows ACL scope, what is verified).

## 0.4.1 — 2026-10-30

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
