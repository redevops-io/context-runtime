# Context Runtime — Specification (v0.1-draft)

Normative interface specification. Where [ARCHITECTURE.md](./ARCHITECTURE.md)
explains *why* and [ROADMAP.md](./ROADMAP.md) explains *when*, this document pins
down *exactly what* the four foundational seams are, so implementation and the
reused substrate can proceed against stable contracts.

The four seams everything else hangs off:

1. **Plugin contracts** (§4) — the only interfaces the runtime depends on. This
   includes the **Reasoner** seam (§4.4): `ModelPlugin` is transport to *one* model;
   `ReasonerPlugin` is the *strategy* that may orchestrate several.
2. **Execution Graph IR** (§5) — the planner's output; the durable, replayable plan.
   It is also the **Planner/Scheduler boundary** (§5.1): the Planner decides *what*,
   the Scheduler decides *when/where*, the graph is the artifact between them.
3. **Trace schema** (§6) — what every run emits; the `EXPLAIN`/`SIMULATE` objects and
   the learning loop's training data, including **cost-model statistics** (§3.1).
4. **Plan-Cache key** (§7) — what makes "plan once, reuse 1000×" correct.

**Conventions.** MUST / SHOULD / MAY per RFC 2119. Types are Python ≥3.12 stubs
(`Protocol`, `@dataclass(frozen=True)`, `Literal`, `TypedDict`). Anything that
crosses a process boundary or is persisted (Execution Graph, Trace, Plan-Cache
entry) MUST also have a stable **JSON form** (§8); in-process-only types need not.
All timestamps are RFC 3339 UTC. All IDs are stable opaque strings (`<kind>_<ulid>`).

---

## 1. Scope

This draft specifies the **v0.1 conformance subset** in full and declares the
forward-compatible shape of v0.2+ fields (marked `# v0.2+`). A v0.1-conformant
runtime MUST implement everything not so marked, and MUST round-trip (parse +
re-serialize without loss) the JSON forms in §8 including unknown forward fields.

Out of scope for this draft: the policy language grammar (OPA/Rego, v0.5), the
distributed execution protocol (v1.0), and the wire protocol of the reference
server (v1.0).

---

## 2. Core data types

### 2.1 Request-side

```python
from dataclasses import dataclass, field
from typing import Literal, Any

@dataclass(frozen=True)
class Constraints:
    """Hard limits (feasibility) + soft requirements. The optimizer treats the
    numeric ceilings as hard constraints and the booleans as plan requirements."""
    max_cost_usd: float | None = None
    max_latency_seconds: float | None = None
    max_tokens: int | None = None
    require_citations: bool = False
    require_verification: bool = False
    sensitivity: Literal["public", "internal", "restricted"] = "public"
    # weight overrides for PlanScore (§3); merged over config defaults
    weight_overrides: dict[str, float] = field(default_factory=dict)

@dataclass(frozen=True)
class SourceRef:
    """A handle to an available source, not its contents."""
    name: str                      # "github", "kubernetes", "grafana", "docs"
    kind: Literal["docs", "code", "logs", "metrics", "api", "graph", "memory"]
    uri: str | None = None
    version: str | None = None     # content fingerprint; fills the Plan-Cache key (§7)

@dataclass(frozen=True)
class Goal:
    text: str
    sources: tuple[SourceRef, ...] = ()
    constraints: Constraints = Constraints()
    conversation_id: str | None = None
```

### 2.2 Intent (output of the Intent Analyzer, §4.2)

```python
@dataclass(frozen=True)
class Intent:
    bucket: Literal[                 # the rule bucket; drives Candidate Generation
        "exact_lookup", "conceptual", "incident", "code_reasoning",
        "synthesis", "high_risk", "sensitive", "unknown"]
    entities: tuple[str, ...] = ()   # extracted ids/error-codes/symbols
    risk: Literal["low", "medium", "high"] = "low"
    normalized: str = ""             # canonical form → Plan-Cache semantic key (§7)
    confidence: float = 0.0          # 0..1
```

`normalized` MUST be deterministic for semantically-equivalent goals under a fixed
analyzer version (it is the cache key's stable half). The analyzer SHOULD be the
cheapest model tier or a non-LLM classifier.

### 2.3 Candidate, PlanScore, Plan

```python
@dataclass(frozen=True)
class StepSpec:
    type: Literal["retrieve", "rerank", "compress", "route", "delegate", "verify"]
    params: dict[str, Any]           # e.g. {"method": "hybrid", "top_k": 50}
    plugin: str | None = None        # which plugin impl; None = runtime default

@dataclass(frozen=True)
class Candidate:
    """A possible plan, pre-scoring. The Candidate Generator emits many; pruning and
    the optimizer select one."""
    steps: tuple[StepSpec, ...]
    model_tier: str                  # "local" | "cheap" | "premium"

@dataclass(frozen=True)
class PlanScore:
    """The soft objective (§3). Each estimate is normalized to [0,1] before weighting."""
    expected_accuracy: float
    cache_hit_probability: float
    verification_confidence: float
    cost_usd: float                  # raw $; normalized at scoring time
    latency_seconds: float           # raw s; normalized at scoring time
    risk: float
    hallucination_probability: float
    context_loss: float
    total: float                     # the weighted PlanScore (§3) — what's maximized
    feasible: bool                   # passed all hard constraints?

@dataclass(frozen=True)
class Plan:
    id: str                          # "plan_<ulid>"
    intent: Intent
    chosen: Candidate
    score: PlanScore
    rejected: tuple[tuple[Candidate, str], ...] = ()   # (candidate, reason) for EXPLAIN
    cache: Literal["hit", "miss", "bypass"] = "miss"
    spec_version: str = "0.1"
```

---

## 3. The Cost Model objective (PlanScore)

The Cost Optimizer (§4.4) selects, among **feasible** candidates, the one maximizing:

```
PlanScore.total =
      w_acc   · n(expected_accuracy)
    + w_cache · n(cache_hit_probability)
    + w_vrf   · n(verification_confidence)
    − w_cost  · n(cost_usd)
    − w_lat   · n(latency_seconds)
    − w_risk  · n(risk)
    − w_hall  · n(hallucination_probability)
    − w_loss  · n(context_loss)
```

- `n(·)` normalizes each term to `[0,1]` (min–max against the candidate set, or a
  configured scale for the $ / seconds terms). Normalization is REQUIRED — the terms
  have different units and an un-normalized sum is meaningless.
- Weights `w_*` come from config (`costmodel.weights`), overridden per request by
  `Constraints.weight_overrides`. Weights are themselves tunable offline (Optuna).
- **Feasibility is separate.** `Constraints` numeric ceilings and boolean
  requirements define a hard-constraint set evaluated by `constraints/` and enforced
  by `optimizer/`. PlanScore only *ranks* the feasible set. A candidate that violates
  `max_cost_usd` is `feasible=False` and excluded regardless of its score.
- The estimator behind these fields is the `costmodel/estimators.py` plugin: v0.1
  heuristic, later learned/neural. Its outputs are the **only** thing the learning
  loop updates (§6.4).

### 3.1 Cost-model statistics (the trust layer)

An optimizer nobody can audit is an optimizer nobody trusts. Every estimator MUST
self-report its calibration, exactly as PostgreSQL keeps `pg_statistic` per column.
Statistics are keyed per estimated field (and SHOULD be sliced by `Intent.bucket`).

```python
@dataclass(frozen=True)
class FieldStatistics:
    field: str                       # "cost_usd", "expected_accuracy", …
    mean_absolute_error: float       # over observed estimate-vs-actual pairs
    calibration: float               # 0..1; fraction of actuals inside the predicted CI
    ci_low: float; ci_high: float    # predicted interval at p=0.9 for a fresh estimate
    sample_count: int
    last_updated: str | None         # RFC3339; None until first samples land

@dataclass(frozen=True)
class CostModelStatistics:
    estimator_version: str
    fields: tuple[FieldStatistics, ...]
    bucket: str | None = None        # Intent.bucket this slice covers, or None = global

@runtime_checkable
class CostEstimator(Protocol):
    def estimate(self, candidate: "Candidate", goal: "Goal") -> PlanScore: ...
    def statistics(self, bucket: str | None = None) -> CostModelStatistics: ...
    def observe(self, plan: "Plan", trace: "Trace") -> None: ...   # update from actuals
```

- **Collection starts v0.1.** `observe()` is called on every completed run; the
  estimate-vs-actual error is recorded even though the *learner* (which uses it to
  improve estimates) ships v0.3. v0.1 MAY report `calibration=0`, wide CIs, and small
  `sample_count` — the contract is that the numbers are *present and honest*.
- These statistics are what `simulate()` (§9) turns into confidence intervals and
  what `explain()` surfaces so a caller can weigh how much to trust an estimate.

---

## 4. Plugin contracts (seam 1)

The runtime imports **none** of `openai`, `anthropic`, `duckdb`, `psycopg`, nor the
literal `"BM25"`. It depends only on these Protocols. v0.1 ships in-tree
implementations; out-of-tree registration is v1.0.

```python
from typing import Protocol, runtime_checkable
```

### 4.1 Common

```python
@dataclass(frozen=True)
class PluginInfo:
    name: str
    kind: Literal["model","store","retriever","knowledge","compression",
                  "verifier","router","policy","planner"]
    version: str
    capabilities: frozenset[str] = frozenset()
```

Every plugin MUST expose `info() -> PluginInfo`. Discovery is by `info()`, never by
hard-coded assumption (principle #5).

### 4.2 PlannerPlugin (the three-stage core)

```python
@runtime_checkable
class IntentAnalyzer(Protocol):
    def analyze(self, goal: Goal) -> Intent: ...

@runtime_checkable
class CandidateGenerator(Protocol):
    def generate(self, intent: Intent, goal: Goal) -> list[Candidate]: ...
    def prune(self, candidates: list[Candidate], goal: Goal) -> list[Candidate]: ...

@runtime_checkable
class CostOptimizer(Protocol):
    def score(self, candidate: Candidate, goal: Goal) -> PlanScore: ...
    def select(self, scored: list[tuple[Candidate, PlanScore]], goal: Goal) -> Plan: ...
```

The default planner composes these three; replacing any one is a research seam.

### 4.3 ModelPlugin

```python
@dataclass(frozen=True)
class ModelCapabilities:
    max_context_tokens: int
    prompt_cache: bool
    tool_calling: bool
    structured_outputs: bool
    vision: bool

@dataclass(frozen=True)
class ModelRequest:
    messages: tuple[dict[str, str], ...]
    capability: str = "draft"        # maps to the native Tier.good_for
    max_tokens: int = 1024
    system: str | None = None
    tools: tuple[dict, ...] | None = None

@dataclass(frozen=True)
class ModelResult:
    text: str
    model: str
    tier: str
    prompt_tokens: int
    completion_tokens: int
    est_cost_usd: float
    cache_hit: bool = False

@runtime_checkable
class ModelPlugin(Protocol):
    def complete(self, req: ModelRequest) -> ModelResult: ...
    def capabilities(self, model: str) -> ModelCapabilities: ...
    def count_tokens(self, text: str, model: str) -> int: ...
    def info(self) -> PluginInfo: ...
```

**Binding (v0.1):** `LiteLLMModel` wraps LiteLLM for transport + token counting +
cost, with a **native cost-tiered routing policy** (`Tier`/`Task`/`Router`/`BudgetExceeded`)
as the tier-selection policy. `ModelRequest.capability` ↔ `Task.capability`;
`ModelResult` mirrors `RouteResult` (`tier`, `model`, `text`, `est_cost_usd`) plus
token counts.

### 4.4 ReasonerPlugin (strategy over ≥1 ModelPlugin)

`ModelPlugin` assumes one model: *LLM → reasoning → answer*. But reasoning is
becoming a **mixture** — *planner → worker → critic → tool → merge* — which is no
longer a single model call. The `ReasonerPlugin` is that strategy layer. It turns
assembled context into a result, and MAY issue several `ModelRequest`s to one or more
`ModelPlugin`s, each routed independently.

```python
ReasoningStrategy = Literal[
    "single_shot",        # one ModelPlugin call — the v0.1 default
    "plan_worker_critic", # decompose → execute → self-critique → merge
    "debate",             # N independent passes → reconcile
    "tool_loop"]          # interleave model + tool calls until done

@dataclass(frozen=True)
class ReasonRequest:
    context: "BuiltContext"
    strategy: ReasoningStrategy = "single_shot"
    capability: str = "synthesis"
    constraints: "Constraints" = Constraints()

@runtime_checkable
class ReasonerPlugin(Protocol):
    def reason(self, req: ReasonRequest) -> ModelResult: ...   # may aggregate sub-calls
    def info(self) -> PluginInfo: ...
```

**Layering.** `ReasonerPlugin → (per sub-call) RouterPlugin → ModelPlugin`. The
Reasoner picks the *strategy* and how many calls; the Router picks the *tier/model*
for each; the Model is the transport. The returned `ModelResult` rolls up the
sub-calls' tokens/cost and notes the model(s) used.

**Boundary vs. Agent Scheduler.** A Reasoner is a *single reasoning step* over one
assembled context (the `reason` node, §5). The **Agent Scheduler** (sidekick, v0.4)
is plan-level *delegation* — separate agents with their own context, tools,
worktrees, and output contracts (the `delegate` node). Mixture-of-models inside one
step ⇒ Reasoner; fan-out to independent agents ⇒ Scheduler.

**Binding (v0.1):** `SingleShotReasoner` — a thin default that wraps one
`ModelPlugin.complete()`. Richer strategies arrive with the learning loop / agents
(v0.3–v0.4); the *contract* exists from v0.1 so the `reason` node is the abstraction
from the start, never a raw model call.

### 4.5 RetrieverPlugin + StorePlugin

```python
@dataclass(frozen=True)
class Hit:
    chunk_id: str
    filename: str
    text: str
    score: float                     # boosted_score (or rerank score if reranked)
    created_at: str | None = None
    source: str | None = None        # SourceRef.name
    meta: dict[str, Any] = field(default_factory=dict)

Retrieval = Literal["vector","bm25","hybrid","graph","sql","api","logs","file","code"]

@runtime_checkable
class RetrieverPlugin(Protocol):
    def search(self, query: str, k: int, method: Retrieval) -> list[Hit]: ...
    def info(self) -> PluginInfo: ...

@runtime_checkable
class StorePlugin(Protocol):
    def index(self, path: str) -> dict: ...       # ingest stats
    def graph(self) -> "GraphHandle": ...          # v0.2+ Knowledge graph
    def info(self) -> PluginInfo: ...
```

**Binding (v0.1):** `RedevopsRagRetriever` wraps `redevops_rag.RAG`:
`RAG.search(query, k)` returns `list[dict]` with `chunk_id/filename/text/
created_at/boosted_score` → mapped field-for-field into `Hit` (`score =
boosted_score`, or the rerank score when `use_reranker=True`). `StorePlugin.index`
delegates to `RAG.index(path)`. `method="hybrid"` is the v0.1 default; `"vector"` /
`"bm25"` select the single-mode paths; `"graph"` is v0.2+.

### 4.6 SchedulerPlugin (decides *when/where*, v2-first-class)

The Planner decides *what* (the Execution Graph); the **Scheduler** decides *when and
where* — execution order, parallel waves, concurrency limits, retry timing, resource
/ backpressure. Today this is delegated to Dagster + sidekick waves; naming the
contract now keeps the v2 pipeline (§5.1) a fill-in, not a refactor.

```python
@dataclass(frozen=True)
class Schedule:
    waves: tuple[tuple[str, ...], ...]   # node-ids grouped into ordered parallel waves
    max_concurrency: int
    retry: dict[str, int]                # node-id → max attempts

@runtime_checkable
class SchedulerPlugin(Protocol):
    def schedule(self, graph: "ExecutionGraph", constraints: Constraints) -> Schedule: ...
    def info(self) -> PluginInfo: ...
```

**Binding (v0.1):** `DagsterScheduler` — a trivial default that topo-sorts the graph
into waves and hands them to Dagster. Cost-aware scheduling (reordering for latency,
budget-aware concurrency) is v2.

### 4.7 Remaining plugins (contracts; bindings per §2 of ARCHITECTURE)

```python
@runtime_checkable
class CompressionPlugin(Protocol):
    def compress(self, text: str, target_tokens: int) -> "Compressed": ...   # keeps provenance

@runtime_checkable
class VerifierPlugin(Protocol):
    def verify(self, result: ModelResult, plan: Plan, ctx: "BuiltContext") -> "Verdict": ...

@runtime_checkable
class RouterPlugin(Protocol):       # tier policy over ModelPlugin
    def choose(self, req: ModelRequest, caps: dict[str, ModelCapabilities]) -> str: ...

@runtime_checkable
class PolicyPlugin(Protocol):
    def check(self, action: "Action", ctx: "PolicyContext") -> "Decision": ...

@runtime_checkable
class KnowledgePlugin(Protocol):    # umbrella: graph + memory lifecycle + sources
    def remember(self, entry: "MemoryEntry") -> str: ...
    def recall(self, query: str, k: int) -> list["MemoryEntry"]: ...
    def graph(self) -> "GraphHandle": ...
```

`Compressed`, `Verdict`, `Action`, `Decision`, `MemoryEntry`, `GraphHandle` are
specified at the phase that introduces them (compression/verify v0.1; knowledge
graph v0.2; policy v0.5). v0.1 MUST provide `ReasonerPlugin` (single-shot),
`SchedulerPlugin` (Dagster default), `CompressionPlugin` (sidekick structural +
LLMLingua-2), and `VerifierPlugin` (Instructor/RAGAS); the rest MAY be no-op stubs
that satisfy the Protocol.

---

## 5. Execution Graph IR (seam 2)

The planner's output, distinct from a pure DAG: it MUST be able to express
conditional branches, loops/retries, parallel waves, human-approval gates, agent
fan-out, verify nodes, merge, and rollback. v0.1 MAY emit only linear/parallel
graphs, but the IR and its JSON form MUST already carry the richer node/edge kinds so
v0.4 slots in without a schema change.

### 5.1 The Planner/Scheduler boundary

The Execution Graph is the artifact between two distinct responsibilities, mirroring
every operating system:

```
Intent → Planner → Execution Graph → Scheduler → Execution
         (what)      (the boundary)    (when/where)
```

- **Planner** (§4.2) produces the *logical* graph: which steps, which plugins, which
  reasoning strategy, what depends on what. Pure function (principle #7).
- **Scheduler** (§4.6) produces the *physical* `Schedule`: wave grouping, concurrency,
  retry timing — then compiles to ≥1 Dagster run.
- v0.1–v1.0 fold scheduling into a trivial topo-sort default; **v2** promotes it to a
  cost-aware first-class stage. Because the boundary (this IR) already exists, that
  promotion adds a plugin, it does not reshape the pipeline.

```python
NodeKind = Literal[
    "retrieve","rerank","compress","route","reason","delegate",
    "verify","branch","loop","approval","merge","rollback"]
    # "reason" invokes a ReasonerPlugin (§4.4); a single model call is the
    # SingleShotReasoner default, never a raw ModelPlugin call from the graph.

@dataclass(frozen=True)
class GraphNode:
    id: str                          # "node_<ulid>", stable within a plan
    kind: NodeKind
    plugin: str | None
    params: dict[str, Any]
    budget_tokens: int | None = None

EdgeKind = Literal["then","on_success","on_failure","on_condition","parallel"]

@dataclass(frozen=True)
class GraphEdge:
    src: str
    dst: str
    kind: EdgeKind = "then"
    condition: str | None = None     # for on_condition / branch / loop guard

@dataclass(frozen=True)
class ExecutionGraph:
    id: str                          # "xg_<ulid>"
    plan_id: str
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    spec_version: str = "0.1"
```

**Compilation.** `execution/dagster_compile.py` compiles an `ExecutionGraph` to **≥1
Dagster run**. DAG-shaped subgraphs map to a single run; loops, rollback, and
conditional re-planning are driven by the runtime across multiple runs (Dagster
stays a DAG-of-assets executor). The `ExecutionGraph` — not any Dagster run — is the
durable, replayable artifact and the unit cached by the Plan Cache (§7).

**Validity.** A graph MUST be acyclic *except* through `loop` nodes whose back-edges
carry an `on_condition` guard with a bounded iteration count in `params`. A
`rollback` node MUST name the nodes it compensates. Validation runs before execution
and before cache insertion.

---

## 6. Trace schema (seam 3)

Every run emits one `Trace` (OpenLLMetry spans → Langfuse). It is simultaneously: the
observability record, the `EXPLAIN ANALYZE` data, the replay input (principle #7), and
the learning loop's training row.

```python
@dataclass(frozen=True)
class Span:
    id: str
    parent_id: str | None
    name: str                        # node id or stage name
    kind: Literal["intent","candidate","optimize","schedule","retrieve","reason",
                  "compress","verify","delegate","cache"]
    start: str; end: str             # RFC3339
    attrs: dict[str, Any]            # tokens, cost_usd, model, scores, hit/miss …

@dataclass(frozen=True)
class Trace:
    id: str                          # "trace_<ulid>"
    plan_id: str
    goal_text: str
    spans: tuple[Span, ...]
    # rolled-up actuals (mirror PlanScore fields for estimate-vs-actual diff)
    actual_cost_usd: float
    actual_latency_seconds: float
    actual_tokens: int
    citations: tuple[str, ...]
    verification_passed: bool | None
    cache: Literal["hit","miss","bypass"]
    spec_version: str = "0.1"
```

### 6.1 EXPLAIN object

`runtime.explain(goal)` returns this without executing; `analyze=True` runs and
overlays actuals.

```python
@dataclass(frozen=True)
class Explanation:
    intent: Intent
    candidates: tuple[tuple[Candidate, PlanScore], ...]   # all scored
    chosen: Plan
    retrieved_sources: tuple[str, ...]
    rejected_sources: tuple[tuple[str, str], ...]         # (source, reason)
    token_budget: dict[str, int]                          # keep/compress/drop totals
    estimated: PlanScore
    statistics: CostModelStatistics   # §3.1 — how trustworthy the estimates are
    plan_cache: Literal["hit","miss","bypass"]
    analyze: Trace | None = None      # present iff analyze=True (estimate vs actual)
```

### 6.2 SIMULATE object

`runtime.simulate(goal)` plans **without executing** and projects the resource
envelope — for enterprise budgeting, quoting, and pre-approval. Where `explain()`
answers *"why this plan?"*, `simulate()` answers *"what will it cost me?"* with
honest uncertainty drawn from the cost-model statistics (§3.1).

```python
@dataclass(frozen=True)
class Interval:
    point: float; low: float; high: float    # p=0.9; collapses to point when sample_count low

@dataclass(frozen=True)
class Simulation:
    expected_cost_usd: Interval
    expected_latency_seconds: Interval
    expected_tokens: Interval
    expected_confidence: float                # = PlanScore.expected_accuracy
    expected_models: tuple[str, ...]          # tiers/models the plan would invoke
    expected_retrieval: tuple[Retrieval, ...] # methods the plan would use
    plan_id: str
    based_on_samples: int                     # CostModelStatistics.sample_count
```

The intervals are `[point ± CI]` from `FieldStatistics.ci_low/ci_high`. With few
samples the CI is wide (and `based_on_samples` is small) — that honesty is the point.

### 6.4 Learning-loop coupling

The `(Plan, Trace)` pair is the bandit/BO training row. Feedback updates the
**cost-model estimators only** (`expected_accuracy`, `cost_usd`, etc.) — never the
optimizer or solver, which stay deterministic functions of those estimates. v0.1
MUST persist `(Plan, Trace)` pairs even though the learner ships in v0.3.

---

## 7. Plan-Cache key (seam 4)

The Plan Cache (v0.2) caches `Intent → ExecutionGraph → PlanScore`. Correctness
depends entirely on the key. A hit reuses the *plan*; it does NOT skip execution.

```python
@dataclass(frozen=True)
class PlanCacheKey:
    intent_normalized: str           # Intent.normalized — matched SEMANTICALLY, not ==
    source_fingerprint: str          # hash of {SourceRef.name: version} for sources used
    policy_fingerprint: str          # hash of active policy + permission/visibility scope
    constraint_envelope: str         # hash of the hard-constraint set (ceilings + flags)
    analyzer_version: str
    planner_version: str
```

- **Semantic match.** Two goals share a plan iff their `intent_normalized` match
  within an embedding-distance threshold *and* all other key fields are equal. String
  equality is insufficient (it misses the "1000 phrasings of the same K8s error"
  case); exact equality on the non-intent fields is REQUIRED (a different permission
  scope or budget MUST NOT share a plan).
- **Invalidation.** An entry is invalid when any of: a referenced source `version`
  changes (→ `source_fingerprint`), policy/permission changes, a model's
  `ModelCapabilities` change, the analyzer/planner version changes, or TTL expires.
- **Soundness.** Rests on deterministic replay (principle #7): equal key ⇒ identical
  `ExecutionGraph`. This is why the Plan Cache cannot predate the v0.2 Knowledge graph
  that supplies versioned sources.

---

## 8. JSON forms & versioning

`Plan`, `ExecutionGraph`, `Trace`, and Plan-Cache entries MUST serialize to stable
JSON (snake_case keys; tuples → arrays; frozen dataclasses → objects). Each carries
`spec_version`. Parsers MUST:

- accept unknown fields (forward-compat) and re-serialize them unchanged;
- reject a `spec_version` with a higher **major** than they implement;
- treat missing optional fields as their documented defaults.

The spec itself is versioned `MAJOR.MINOR`; additive fields bump MINOR, breaking
shape changes bump MAJOR.

---

## 9. Public API contract

```python
class ContextRuntime(Protocol):
    @classmethod
    def from_config(cls, path: str) -> "ContextRuntime": ...

    def run(self, goal: str | Goal, *, sources=..., constraints=...) -> "RunResult": ...
    def plan(self, goal, *, sources=..., constraints=...) -> Plan: ...
    def build_context(self, plan: Plan) -> "BuiltContext": ...   # → ExecutionGraph + assembled ctx
    def execute(self, ctx_or_graph) -> "RunResult": ...
    def verify(self, result: "RunResult") -> "RunResult": ...
    def explain(self, goal, *, analyze: bool = False, sources=..., constraints=...) -> Explanation: ...
    def simulate(self, goal, *, sources=..., constraints=...) -> Simulation: ...   # plan, never execute
```

`explain` and `simulate` are both **non-executing** (they call `plan()`, not
`execute()`) and share its determinism guarantee. `explain` is for debugging a plan;
`simulate` is for forecasting its envelope.

`RunResult` MUST expose `.answer: str`, `.trace: Trace`, `.plan: Plan`,
`.citations: tuple[str,...]`, `.cost_usd: float`. `run` is `plan → build_context →
execute → verify` composed; the granular methods exist for inspection and reuse.

**Determinism.** `plan(goal)` MUST be a pure function of `(goal, config, source
versions, plugin versions)`. `execute` MAY be non-deterministic (model sampling);
`build_context` MUST NOT be (principle #7).

---

## 10. v0.1 conformance checklist

A runtime is **v0.1-conformant** iff it:

- [ ] implements `ModelPlugin` (LiteLLM + native cost-tiered routing) and `StorePlugin`
      for **both** DuckDB and pgvector behind the *same* `RetrieverPlugin` contract;
- [ ] implements the three planner Protocols (Intent / Candidate / Optimizer) with a
      rule-table intent analyzer and a **heuristic** `costmodel` producing `PlanScore`
      with normalized terms and config/request weights;
- [ ] implements `CostEstimator.observe()` so estimate-vs-actual error is recorded per
      run, and `statistics()` returns honest (possibly low-confidence) numbers;
- [ ] implements `ReasonerPlugin` (`SingleShotReasoner`) and `SchedulerPlugin`
      (`DagsterScheduler` topo-sort default) — the `reason`/scheduling seams exist
      even though mixtures and cost-aware scheduling come later;
- [ ] selects via `optimizer/knapsack.py` (token-budget knapsack / greedy
      value-density); CP-SAT NOT required;
- [ ] emits a valid `ExecutionGraph` (linear/parallel acceptable) and executes it;
- [ ] emits a `Trace` per run and persists every `(Plan, Trace)` pair;
- [ ] implements `explain(goal[, analyze=True])` and `simulate(goal)` returning
      populated `Explanation` / `Simulation` objects (CIs MAY be wide early);
- [ ] round-trips all §8 JSON forms including unknown forward fields;
- [ ] requires neither Plan Cache, Knowledge graph, CP-SAT, mixture reasoners,
      cost-aware scheduling, policy engine, agent scheduler, nor dynamic plugin
      loading (all v0.2+).
```
