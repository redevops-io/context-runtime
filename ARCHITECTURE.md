# Context Runtime — Architecture

> **Mental model: Context Runtime is a database query planner for LLM context.**
>
> The application no longer says *"retrieve these chunks, rerank them, summarize
> them, send them to Claude."* It says **"I need an answer."** Everything else
> becomes an **execution plan** the runtime decides — exactly as SQL replaced
> *open table → scan pages → hash join → sort* with `SELECT ...`.
>
> That makes Context Runtime an **infrastructure layer** that higher-level frameworks
> (LangGraph, CrewAI, LlamaIndex) build *on top of* — a far more durable position
> than competing with them.

This document is the engineering architecture. It commits to two things the
original plan left open:

1. **What we reuse vs. build.** Context Runtime is a *thin decision layer over a thick
   reused substrate*. The novel code is the **Planner pipeline + Cost Model + trace
   schema + plugin contracts**. Almost everything else is assembly of existing OSS.
2. **How decisions are made.** Not linear programming — a **cost-based query
   optimizer + constrained selection + online learning** loop, with first-class
   `EXPLAIN` and a **Plan Cache**, exactly like a SQL planner.

See [ROADMAP.md](./ROADMAP.md) for phasing.

---

## 1. Design constraints (non-negotiable)

| # | Principle | Consequence |
|---|---|---|
| 1 | **Context is versioned state**, not a string | The Knowledge Layer (§5.4), not the prompt, is the source of truth. |
| 2 | **Retrieval is one strategy**, not the runtime | RAG is one plugin the planner *may* select, alongside BM25, graph, SQL, API, logs. |
| 3 | **Compression is lossy until proven otherwise** | Every summary keeps `derived_from`, what it omitted, and a refresh trigger. |
| 4 | **Verification is part of execution** | A result is incomplete until it passes its required verification policy. |
| 5 | **Provider & backend independence** | Nothing in the runtime knows about BM25, DuckDB, or Anthropic. It knows only **plugin contracts** (§2.3). |
| 6 | **Observability is mandatory** | Unobservable decisions are unacceptable — and the trace is the training data for the learning loop (§6). |
| 7 | **Deterministic replay** | Same inputs + policies + source versions ⇒ the *plan* reproduces (model output may vary). This is what makes the **Plan Cache** (§5.13) sound. |

**Plugin-first (decided).** Generalizing the "adapter-first" substrate decision:
*every* subsystem is a plugin behind a contract — Retriever, Memory, Verifier,
Compression, Router, Policy, Planner, and the two original adapters (Model, Store).
The runtime depends only on interfaces, like Kubernetes controllers. The *same plan*
runs local (DuckDB + Ollama/vLLM) or cloud (pgvector + Anthropic/OpenAI) by swapping
the Model/Store plugins — never by touching planner or runtime code. **Contracts
ship in v0.1; dynamic plugin loading is a v1.0 concern** (in-tree implementations
until then — don't build an entry-point registry before the interfaces are proven).

---

## 2. The substrate we reuse

Context Runtime does **not** reimplement retrieval, routing, agent orchestration, provider
SDKs, policy, or memory from scratch. It composes existing code behind plugin
contracts.

### 2.1 In-house repos (already own)

| Context Runtime plugin | Repo | What it provides |
|---|---|---|
| **Retriever** | [`redevops-rag`](https://github.com/redevops-io/redevops-rag) (single-hop) + [`HippoRAG`](https://github.com/redevops-io/HippoRAG) (multi-hop) | DuckDB dense + BM25 → **RRF** → rerank for single-hop; knowledge graph + Personalized PageRank for multi-hop. The planner routes between them per query. |
| **Router** | native (`context_runtime/adapters/model_litellm.py`) | Capability-tiered routing (local → cheap → premium), per-task budget, fallback up tiers. *Context Runtime is the control plane — routing is native; prototyped in agentic-os (now the public v6 cockpit).* |
| **Policy / Tools** | native (`context_runtime/tools/`, `context_runtime/policy/`) | ApprovalPolicy + dangerous-pattern scan, human-in-the-loop gates, append-only audit log. *Absorbed from agentic-os `safety.py`/`control_plane`.* |
| **Agent Scheduler** | [`sidekick`](https://github.com/redevops-io/sidekick) `orchestrator.py`, `planner.py`, `worktree.py` | DAG decomposition → bounded parallel waves → isolated worktrees → acceptance checks → merge. Validated 5.4× speedup, 0 conflicts. |
| **Compression (structural) + Token Budget** | `sidekick` `context_budget.py`, `memory.py` | `clip()` + `reduce_transcript()` (tiered, dedup). |
| **Observability seed** | `sidekick` `metrics.py`, `dashboard.py`, `events.py` | Per-agent token/turn/cost/latency + live trace doc. |

### 2.2 External OSS (glue in, do not build)

| Context Runtime plugin/component | Primary OSS | Role / alternatives |
|---|---|---|
| **Model / Store adapters (6, 5.6)** | **LiteLLM** | Unified API for 100+ providers, normalized cost/token, fallbacks, prompt-cache passthrough. *Replaces the hand-written `providers/` directory.* |
| **Knowledge Layer — graph (5.2)** | **Kùzu** (embedded property graph) | `contains/derived_from/supersedes/contradicts` edges. Alt: `networkx`, Neo4j, **OpenLineage+Marquez** (Dagster-native lineage). |
| **Knowledge Layer — memory (5.4)** | **Graphiti (Zep)** — bi-temporal KG | Tracks when a fact was true *and* recorded; native supersession → matches `verified/confidence/expires_at/supersedes`. Alt: **mem0** (v0.1), **Letta/MemGPT**. |
| **Compression — semantic (5.5)** | **LLMLingua-2** | Learned 2–5× token cut; pairs with sidekick's structural reducer. + `tree-sitter`, `Chonkie`. |
| **Prompt Cache (5.6)** | provider-native (LiteLLM) + **vLLM prefix / SGLang RadixAttention** (local) | + **GPTCache / RedisVL** for cross-provider *semantic* cache. |
| **Verification (5.9)** | **Instructor + Pydantic** · **RAGAS / DeepEval** | + Guardrails/NeMo, Semgrep/Bandit/`terraform validate`/OPA-conftest. |
| **Policy (5.10)** | **Open Policy Agent (Rego)** + **Presidio** (PII) | §5.10 YAML → Rego; `safety.py` stays as fast inline pre-filter. |
| **Token counting (5.11)** | `tiktoken` + provider tokenizers (LiteLLM) | — |
| **Observability (5.12)** | **OpenTelemetry + OpenLLMetry** → **Langfuse** | Self-hostable trace/cost/eval + replay (principle #7). Alt: Arize **Phoenix**. |
| **Execution (5.8 / §4)** | **Dagster** | Runs the Execution Graph; Context Runtime *decides* it. |
| **Optimizer (§6)** | **OR-Tools CP-SAT** · **Optuna** · **River** · (Ray Tune later) | Constrained selection · offline tuning · online contextual bandits. |

### 2.3 Plugin contracts (the only thing the runtime knows)

```python
class RetrieverPlugin(Protocol):
    def search(self, query: str, k: int, method: Retrieval) -> list[Hit]: ...
class ModelPlugin(Protocol):                     # transport to ONE model
    def complete(self, req: ModelRequest) -> ModelResult: ...
    def capabilities(self, model: str) -> ModelCapabilities: ...   # discovered, not hard-coded
    def count_tokens(self, text: str, model: str) -> int: ...
class ReasonerPlugin(Protocol):                  # STRATEGY over ≥1 ModelPlugin
    def reason(self, req: ReasonRequest) -> ModelResult: ...        # single-shot | plan-worker-critic | debate
class StorePlugin(Protocol):
    def index(self, path: str) -> IndexStats: ...
    def graph(self) -> GraphHandle: ...
class SchedulerPlugin(Protocol):                 # decides WHEN/WHERE (vs Planner: WHAT)
    def schedule(self, graph: ExecutionGraph, constraints: Constraints) -> Schedule: ...
# …and KnowledgePlugin, CompressionPlugin, VerifierPlugin, RouterPlugin,
#    PolicyPlugin, PlannerPlugin — same pattern.  See SPEC.md §4 for full contracts.
```

Two seams worth calling out (added after review): **`ReasonerPlugin`** exists because
a model call is no longer *LLM→answer* — it is increasingly a mixture
(*planner→worker→critic→merge*). The Reasoner is that strategy; it orchestrates one
or more `ModelPlugin`s (`Reasoner → Router → Model`). And **`SchedulerPlugin`** is
split from the Planner: the Planner decides *what* (the Execution Graph), the
Scheduler decides *when/where* (§5.1).

The runtime never imports `openai`, `anthropic`, `duckdb`, or knows the string
`"BM25"`. It calls `RetrieverPlugin.search()`. Capabilities are **discovered**
(context window, prompt-cache, tool calling, vision), never hard-coded.

---

## 3. Layered architecture

```
 Application
     │  runtime.run(goal, sources, constraints)   ·   runtime.explain(goal) → EXPLAIN
     ▼
┌──────────────────────────── Context Runtime Runtime ────────────────────────────┐
│                                                                            │
│   ┌─────────── PLANNER PIPELINE (the core, §5.1) ───────────┐              │
│   │  Intent Analyzer → Candidate Generator → Cost Optimizer │              │
│   │   "what is        "what plans are       "which is        │             │
│   │    wanted?"         possible?"           cheapest & feasible?"         │
│   └───────────┬──────────────────────────────────┬──────────┘             │
│        reads  │                          emits    │                        │
│               ▼                                    ▼                        │
│   ┌──────────────────────┐            ┌──────────── PLAN CACHE (§5.13) ──┐  │
│   │   KNOWLEDGE LAYER     │            │  intent → plan → exec-graph →   │  │
│   │  (§5.4: graph+memory+ │            │  cost estimates  (keyed on      │  │
│   │  docs+logs+metrics+   │            │  source versions + policy)      │  │
│   │  conversation+tools)  │            └─────────────────────────────────┘  │
│   └──────────────────────┘                                                  │
│               │ Execution Graph (the IR — §4) ── the Planner/Scheduler boundary │
│               ▼                                                             │
│        ┌─────────────┐  physical Schedule (waves, concurrency, retry)      │
│        │  SCHEDULER  │  decides WHEN/WHERE (§5.1; trivial topo-sort → v2 cost-aware)│
│        └──────┬──────┘                                                      │
│               ▼                                                             │
│   Retriever · Compression · PromptCache · Agent Scheduler · Verifier ·     │
│   Router · Policy · REASONER  — all PLUGINS, all calls cross contracts (§2.3)│
│               │   reason-node → Reasoner → Router → Model (mixture-capable) │
│   Observability (OpenLLMetry→Langfuse): every decision is a trace span ─────┼─┐
└──────────────────────────────────────────────────────────────────────────┘  │
               │ Schedule compiles to ≥1 Dagster run                           │
               ▼                                                               │
          Dagster execution ───── observed outcomes ─── feed Cost Model ───────┘
               │                  (CostEstimator.observe → statistics, §6.3)
               ▼
          Response / Action + Trace + Knowledge update

  v2 pipeline (the boundary already exists, so this is a fill-in not a refactor):
     Intent → Planner → Execution Graph → Scheduler → Execution   (mirrors an OS)
```

---

## 4. The Execution Graph (not just a DAG)

The planner's output is an **Execution Graph**, an intermediate representation richer
than a pure DAG. It must express: conditional branches, loops/retries, parallel
waves, human approval gates, agent fan-out, verification nodes, merge, and
**rollback**.

- **Context Runtime plans the Execution Graph; Dagster executes it.** Because Dagster is
  fundamentally a DAG-of-assets executor, loops and rollback are handled by the
  runtime compiling the Execution Graph to **one or more** Dagster runs (driving
  iterations, conditional re-planning, and compensating actions across runs) — not
  by assuming Dagster natively loops. The Execution Graph is the durable, replayable
  artifact; Dagster runs are its physical executions.

Lifecycle: `Goal → Intent → Plan → Execution Graph → (Retrieve│Knowledge│Cache│
Tools) → Assembly → Model → Verification → Response/Action → Trace + Knowledge
update`.

---

## 5. Components

### 5.1 Planner pipeline — *split into three responsibilities*

The single biggest structural change: the Planner is **not one module**. It is three
plugins with very different jobs, which keeps each researchable in isolation.

```
Goal → [Intent Analyzer] → [Candidate Generator] → [Cost Optimizer] → Execution Graph
        "what is wanted?"    "what plans are          "which feasible
                              possible?"                plan is best?"
```

- **Intent Analyzer** — classifies the goal (exact-id lookup? conceptual? incident?
  high-risk? sensitive?), extracts entities, picks the rule bucket. Cheap, fast,
  cacheable — and the cache key for the Plan Cache (§5.13).
- **Candidate Generator** — enumerates *possible* Execution Graphs: retrieval method
  × model tier × reasoning strategy × compression × verification × execution shape.
  Rule-based pruning cuts impossible/forbidden candidates before scoring.
- **Cost Optimizer** — scores survivors with the **Cost Model** (§6) and selects the
  cheapest plan satisfying all hard constraints (knapsack → CP-SAT).

> **Planner ≠ Scheduler.** The Cost Optimizer does *planning* (which plan — *what*).
> *Scheduling* (execution order, parallel waves, concurrency, retry timing — *when/
> where*) is a separate responsibility, the `SchedulerPlugin` (§5.14). The Execution
> Graph is the artifact between them. Through v1.0 the scheduler is a trivial
> topo-sort over Dagster; **v2** makes it cost-aware and first-class:
> `Intent → Planner → Execution Graph → Scheduler → Execution`.

### 5.2 Knowledge Layer — *was "Memory Manager"*

Renamed because memory, graph, documents, logs, metrics, conversation, and tool
output are **all knowledge** — "memory" is too LLM-specific. The Knowledge Layer is
the umbrella over:

- **Graph** (Kùzu) — provenance/contradiction/staleness: *where did this come from?
  is it stale? what depends on it? has it been contradicted?*
- **Memory** (mem0 → Graphiti) — the *lifecycle* sub-concern: promotion, expiration,
  compaction, contradiction detection, permission checks. (We keep the word "memory"
  for this lifecycle, not as the layer's name.)
- **Sources** — docs/logs/metrics/conversation/tool output, all addressable as nodes.

### 5.3 Retriever, 5.5 Compression, 5.6 Prompt Cache, 5.7 Router, 5.8 Agent
Scheduler, 5.9 Verifier, 5.10 Policy, 5.11 Token Budget, 5.12 Observability — as in
the reuse map (§2). Each is a plugin; the planner selects and parameterizes it.

### 5.3a Reasoner — *the model is no longer the unit of reasoning*

`ModelPlugin` is transport to one model. The **Reasoner** is the *strategy* a
`reason` node runs: `single_shot` (one model — the v0.1 default), or a mixture
(`plan_worker_critic`, `debate`, `tool_loop`) that issues several model calls, each
independently routed. Layering: `Reasoner → Router → Model`. Distinct from the Agent
Scheduler (§5.8): a Reasoner is *one step over one assembled context*; the Agent
Scheduler is *delegation to independent agents* with their own context and tools.
The `reason` node is the abstraction from v0.1 so mixtures slot in without touching
the graph.

### 5.14 Scheduler — *decides when/where*

Split from the Planner (see the boxed note in §5.1). Takes an Execution Graph +
constraints → a physical `Schedule` (wave grouping, concurrency cap, retry timing) →
compiles to ≥1 Dagster run. v0.1 = `DagsterScheduler` topo-sort; v2 = cost-aware
(reorder for latency, budget-aware concurrency).

### 5.13 Plan Cache — *new subsystem (distinct from Prompt Cache)*

Caches the **planning decision**, not the model context. If 1000 users ask "explain
this Kubernetes error," the planner should optimize **once** and reuse:

```
Intent → Execution Graph → Cost Estimates    (cache this whole chain)
```

Exactly like a database execution-plan cache. Engineering specifics that make it
*correct* rather than a stale-answer machine:

- **Key** = normalized Intent (semantic/embedding match, not string equality — the
  hard part) **+ source-version fingerprint + active policy/permission context +
  constraint envelope**. Two users with different permissions or budgets may not
  share a plan.
- **Invalidation** on: source change, policy change, model-capability change,
  permission change, TTL. Soundness rides on principle #7 (deterministic replay):
  identical key ⇒ identical plan.
- **Hit ≠ skip execution.** A plan-cache hit reuses the *plan*; retrieval/model calls
  still run (or hit the Prompt Cache / semantic cache separately).

This is plausibly a flagship feature: planning is the expensive cognitive step;
caching it is where large-scale cost collapses.

---

## 6. How the Planner decides (the optimizer)

Not linear programming — a **cost-based query optimizer + constrained selection +
online learning**, closer to PostgreSQL's planner than to generic stochastic
optimization.

```
   Candidate Generator (§5.1)
            │   retrieval × model × compression × verification × execution
            ▼
   Rule-Based Pruning      (exact-id→BM25+logs; conceptual→vector;
            │               incident→logs+git+docs; sensitive→local;
            │               high-risk→verifier required)
            ▼
   COST MODEL / PlanScore  ◄──────────────────────────────┐   the LEARNED part
            │   estimates per candidate (§6.1)             │   (its own package)
            ▼                                              │
   Constraint Solver       (hard constraints only:         │
            │  v0.1: knapsack   cost≤$X ∧ latency≤Ys ∧      │
            │  v0.2: CP-SAT     tokens≤N ∧ sensitive→local  │
            │                   ∧ must-cite …)              │
            ▼                                              │
   Dagster Execution ── observed outcomes ── Langfuse ──────┘
                         River (online bandit) + Optuna (offline)
                         update the COST MODEL's estimates
```

### 6.1 CostModel as a first-class package — *and the PlanScore objective*

The Cost Model gets its own package, on a path to learned then neural estimators,
exactly like PostgreSQL statistics:

```
context_runtime/planner/        # intent, candidates (orchestration)
context_runtime/costmodel/      # PlanScore + estimators: v1 heuristic → learned → neural
context_runtime/constraints/    # hard-constraint definitions (feasibility)
context_runtime/optimizer/      # knapsack / CP-SAT selection over feasible set
context_runtime/execution/      # Execution Graph IR → Dagster compilation
```

**The objective (soft) — what the Cost Optimizer maximizes:**

```
PlanScore = + w_acc · ExpectedAccuracy
            + w_cache · CacheHitProbability
            + w_vrf · VerificationConfidence
            − w_cost · Cost
            − w_lat · Latency
            − w_risk · Risk
            − w_hall · HallucinationProbability
            − w_loss · ContextLoss
```

Two engineering refinements on top of the raw formula:

1. **It's a weighted utility, not a bare sum.** The terms have different units
   (accuracy ∈ [0,1], cost in $, latency in s). Each is **normalized to [0,1]** and
   weighted by `w_*`; the weights are themselves tunable (Optuna offline) and can be
   overridden per request via `constraints` (e.g. a latency-critical caller raises
   `w_lat`).
2. **Soft objective vs. hard constraints are separate stages.** PlanScore *ranks*
   feasible plans; the **constraints package** (`cost≤$X`, `tokens≤N`, `sensitive→
   local`, `must-cite`) defines *feasibility* and is enforced by the solver. CP-SAT
   is a pure deterministic function of (estimates + constraints); learning improves
   the *estimates*, then the solver re-solves. **Feedback closes onto the Cost Model,
   not the solver.**

### 6.1a Cost-model statistics — the trust layer

An optimizer nobody can audit is an optimizer nobody trusts. Like PostgreSQL's
`pg_statistic`, every estimator self-reports calibration **per estimated field**
(optionally sliced by intent bucket): mean absolute error, calibration (fraction of
actuals inside the predicted interval), a p=0.9 confidence interval, sample count,
last-updated. `CostEstimator.observe(plan, trace)` records estimate-vs-actual on
every run; `statistics()` exposes the numbers.

- **Collection starts v0.1** even though the *learner* (which consumes the error to
  improve estimates) is v0.3. Early numbers are low-confidence and wide-interval —
  the contract is that they are *present and honest*, not that they are good yet.
- These statistics are what `simulate()` (§7) turns into confidence intervals and
  what `explain()` surfaces, so a caller can weigh how much to trust a number.

### 6.2 Two learners

- **River (online contextual bandit)** — per-request "which retrieval / which model
  for *this* query." Learns from streaming feedback. (Vowpal Wabbit at scale.) The
  reward is *judged quality − efficiency penalty*; when retrieval-score **calibration**
  is enabled it also blends the mean calibrated `P(relevant)` of the served passages, so
  the bandit is scored on retrieved relevance, not the coarse per-query judge alone.
  Learning runs off the response path (the judge + policy update never add to serving
  latency). See `BENCHMARKS.md`.
- **Optuna (offline batch)** — global params *and* PlanScore weights: `top_k`,
  reranker threshold, compression ratio, parallel workers, `w_*`. (DSPy is an
  alternative for the prompt/pipeline-compilation slice.)
- **Observability is a v0.1 dependency**, not a v0.5 feature: bandits/BO need logged
  plan→outcome traces, so Langfuse/OTel ships first.

**When CP-SAT earns its place:** only once ≥3 hard constraints genuinely interact
across a multi-step plan. Until then, **knapsack over the token budget + greedy
value-density** suffices. That is the v0.1 → v0.2 boundary.

---

## 7. Public API — `run`, `plan`, and `explain`

```python
from context_runtime import ContextRuntime
runtime = ContextRuntime.from_config("context_runtime.yaml")

# The core abstraction is run, not ask:
result = runtime.run(
    goal="Explain why deployment X failed",
    sources=["github", "kubernetes", "grafana", "docs"],
    constraints={"max_cost_usd": 2.00, "max_latency_seconds": 90,
                 "require_citations": True, "require_verification": True},
)
print(result.answer); print(result.trace)
```

### 7.1 EXPLAIN — the killer feature

Debug an AI system the way you debug SQL:

```python
runtime.explain(goal)          # plan only, no execution  (like EXPLAIN)
runtime.explain(goal, analyze=True)   # plan + real stats (like EXPLAIN ANALYZE)
```

returns, in one inspectable object:

```
Intent                 ·  Candidate Plans (with scores)
Chosen Plan            ·  Rejected Sources (and why)
Retrieved Sources      ·  Estimated Cost / Accuracy / Latency
Token Budget           ·  Verification plan
Plan-Cache: hit|miss   ·  (ANALYZE: actual cost/latency/tokens vs. estimate)
```

`EXPLAIN ANALYZE` overlays the trace's *actual* numbers on the planner's *estimates*
— the same diff that lets database engineers spot bad row-count estimates.

### 7.2 SIMULATE — forecast without executing

```python
sim = runtime.simulate(goal)   # plans, never executes
# → expected_cost_usd / latency / tokens as CONFIDENCE INTERVALS (from §6.1a stats),
#   expected_confidence, expected_models, expected_retrieval, based_on_samples
```

Where `explain()` answers *"why this plan?"*, `simulate()` answers *"what will it
cost me?"* — the enterprise budgeting/quoting/pre-approval seam. Its intervals come
straight from the cost-model statistics, so they are honestly wide when the model is
under-sampled.

```python
# Advanced seams:
plan  = runtime.plan(goal, sources=sources, constraints=constraints)
graph = runtime.build_context(plan)      # → Execution Graph
res   = runtime.execute(graph)
ver   = runtime.verify(res)
```

---

## 8. Configuration (plugin-first shape)

```yaml
runtime:
  plugins:
    model:   litellm                 # the only model path
    store:   duckdb                  # swap → pgvector / lancedb, same plan
    retriever: redevops_rag
    knowledge: mem0                  # → graphiti at v0.3
    router:  native_tiers
    policy:  opa
  plan_cache: {enabled: true, ttl_seconds: 3600, match: semantic}

models:                              # capability-discovered, not assumed
  tiers:
    - {name: local,   base_url: http://localhost:11434/v1, model: qwen, good_for: [draft, classify]}
    - {name: cheap,   provider: deepseek, good_for: [synthesis, code]}
    - {name: premium, provider: anthropic, model: claude-opus, good_for: [verify, hard]}

budgets:    {max_tokens: 120000, max_cost_usd: 5.00, max_latency_seconds: 120}
costmodel:  {estimator: heuristic, weights: {acc: 1.0, cost: 0.6, lat: 0.3, risk: 0.8, hall: 0.9, loss: 0.4, cache: 0.2, vrf: 0.5}}
retrieval:  {default_strategy: hybrid, top_k: 50, final_k: 8, reranker: bge-reranker-v2-m3}
verification: {default: true, high_risk_requires_human: true}
policy:     {engine: opa, file: policies.rego, pii: presidio}
observability: {exporter: openllmetry, sink: langfuse, traces: true}
```

---

## 9. Repository structure (target)

> This is the **target** layout — several entries are roadmap-gated (CP-SAT optimizer, mixture reasoner, OPA/Presidio policy, prompt cache) and not all present yet; see [ROADMAP.md](ROADMAP.md) for what has shipped.

```
context_runtime/
  ARCHITECTURE.md  ROADMAP.md  SPEC.md  README.md  pyproject.toml
  context_runtime/
    runtime/        runtime.py  lifecycle.py  config.py  explain.py        # EXPLAIN
    planner/        intent.py  candidates.py  rules.py                     # NEW — split (WHAT)
    costmodel/      score.py  estimators.py  statistics.py                # NEW — first-class + trust layer
    constraints/    hard.py                                               # NEW — feasibility
    optimizer/      knapsack.py  cpsat.py                                 # NEW — selection
    execution/      graph.py  dagster_compile.py                         # NEW — Execution Graph IR (boundary)
    scheduler/      schedule.py                                           # NEW — WHEN/WHERE (topo-sort → v2 cost-aware)
    reasoner/       single_shot.py  mixture.py                           # NEW — strategy over ≥1 ModelPlugin
    plancache/      cache.py  invalidation.py  keying.py                  # NEW — Plan Cache
    knowledge/      graph_kuzu.py  memory.py  sources.py                  # was "memory/"
    plugins/        base.py  registry.py        # contracts; dynamic load = v1.0
    adapters/       model_litellm.py  store_duckdb.py  store_pgvector.py
    retrieval/      redevops_rag_binding.py
    compression/    structural.py  semantic.py
    cache/          prompt_cache.py            # NOT plan cache — model-context cache
    routing/        policy.py
    agents/         scheduler.py
    verification/   verifier.py  validators.py
    policy/         opa.py  presidio.py  safety.py
    observability/  traces.py  exporters.py
  examples/  benchmarks/
```

Gone vs. the original §9: no `providers/*.py` (LiteLLM); `retrieval/` is a binding.
Added: `costmodel/ constraints/ optimizer/ execution/ plancache/ scheduler/
reasoner/` and the Planner split (planner=*what*, scheduler=*when/where*).

---

## 10. Benchmarks (plan §11 + one the plan missed)

Standard: retrieval accuracy, answer correctness, citation accuracy, token
reduction, latency, cost, lost-in-the-middle, stale detection, reproducibility,
verification effectiveness — plus the **Context Runtime vs. naive/vector/hybrid/+reranker**
corpus benchmark.

**New — Developer Time / Lines-of-Code eliminated.** Everyone benchmarks accuracy,
latency, cost. Nobody benchmarks *how much glue code disappears*:

```
Task: incident-review pipeline
  Hand-rolled (LangGraph)     ~420 LOC
  Context Runtime                     ~38 LOC
```

This is both a real metric (maintenance surface) and the sharpest selling point — it
is the SQL argument made concrete.

---

## 11. What Context Runtime is *not*

Not a chatbot framework, RAG library, vector DB, agent framework, prompt templater,
or model SDK. **It is the query planner that sits *beneath* LangGraph, CrewAI,
LlamaIndex, Haystack, and MCP** and decides how context flows through all of them —
provider-agnostically, replayably, with `EXPLAIN`. Not a competitor to those
frameworks; the infrastructure layer they build on.
