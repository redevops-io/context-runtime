# ContextOS — Roadmap

Phasing for the Context Runtime. The guiding rule from [ARCHITECTURE.md](./ARCHITECTURE.md):
**ContextOS is a thin decision layer over a thick reused substrate.** Each phase
ships the smallest decision-making increment on top of OSS we already assemble, and
*proves* it with a benchmark — ContextOS must be more than architectural language.

**The optimizer matures across phases — not the model call.** Don't optimize the
model call first; optimize the *context plan*.

```
v0.1 Planner split + cost scoring + knapsack + EXPLAIN  ← prove the core abstraction
v0.2 + CP-SAT + Knowledge graph + Plan Cache             ← constrained selection + reuse
v0.3 + Memory lifecycle + River online learning loop     ← the moat: learn from traces
v0.4 + Agent Scheduler (Execution Graph: branches/loops) ← multi-agent with contracts
v0.5 + Policy Engine (OPA/Presidio)                       ← data-routing + approval gates
v1.0 Production runtime                                    ← SDK, OTel, dynamic plugins, ref server
```

---

## Cross-cutting: ships in v0.1, not later

Foundations the original plan deferred but that everything else depends on:

1. **Plugin-first substrate** (`adapters/model_litellm.py`, `store_{duckdb,pgvector}.py`
   behind `plugins/base.py` contracts). The same plan must run local or cloud from
   day one. *Contracts* ship now; *dynamic plugin loading* is v1.0 — in-tree
   implementations until the interfaces are proven.
2. **Observability** (OpenLLMetry → Langfuse). You cannot run bandits/BO without
   logged plan→outcome traces. The trace *is* the training data for v0.3+, so it
   ships first, not at v0.5.
3. **EXPLAIN + SIMULATE from day one.** `runtime.explain(goal)` (debug a plan) and
   `runtime.simulate(goal)` (forecast its cost/latency/token envelope without
   executing) are near-free once the planner emits a Plan object. EXPLAIN makes
   ContextOS legible ("debug AI like SQL"); SIMULATE is the enterprise
   budgeting/approval seam. Both v0.1; SIMULATE's confidence intervals widen/narrow
   as the cost-model statistics accumulate (honest from the first run).
4. **Reasoner + Scheduler seams exist from v0.1**, even as trivial defaults
   (single-shot reasoner, topo-sort scheduler). Naming them now keeps mixture
   reasoning (v0.3–v0.4) and cost-aware scheduling (v2) as fill-ins, not refactors.

---

## v0.1 — Context Runtime MVP

**Goal:** prove `context.run(goal)` produces better, cheaper, and more inspectable
model calls than hand-rolled RAG.

**Build (new code — the genuinely novel ~core):**
- `runtime/` — `run()` / `plan()` / `build_context()` / `execute()` / `verify()` seams
  **+ `explain.py`** (`runtime.explain(goal[, analyze=True])`)
- **Planner split** — `planner/intent.py` (Intent Analyzer) + `planner/candidates.py`
  (Candidate Generator) + `planner/rules.py`. Three responsibilities, not one module.
- **`costmodel/`** as its own package — `score.py` (the PlanScore weighted utility) +
  `estimators.py` (v1 = heuristic). First-class from day one.
- **`optimizer/knapsack.py`** — token-budget knapsack / greedy value-density.
  *No CP-SAT yet.*
- **`costmodel/statistics.py`** — `CostEstimator.observe()`/`statistics()`: record
  estimate-vs-actual error on every run (the trust layer). Numbers may be
  low-confidence early; the contract is they're present and honest.
- **`reasoner/single_shot.py`** — `ReasonerPlugin` default wrapping one `ModelPlugin`.
  The `reason` node is the abstraction from day one; mixtures (plan-worker-critic,
  debate) come with the learning loop/agents (v0.3–v0.4).
- **`scheduler/schedule.py`** — `SchedulerPlugin` trivial topo-sort over Dagster.
  Names the Planner/Scheduler boundary now; cost-aware scheduling is v2.
- **`execution/graph.py`** — the Execution Graph IR (even if v0.1 only emits linear
  graphs, the IR carries branch/loop/approval/rollback kinds so v0.4 slots in
  without a rewrite). It is the Planner→Scheduler boundary artifact.
- `plugins/base.py` — plugin contracts; `adapters/` model (LiteLLM + agentic-os tier
  policy) and store (DuckDB **and** pgvector, same interface)
- trace schema + `observability/` (OpenLLMetry → Langfuse)

**Assemble (reuse, don't build):**
- Retrieval → **redevops-rag** behind `StoreAdapter`
- Providers → **LiteLLM** (deletes the hand-written `providers/` dir)
- Routing policy → **agentic-os** `router.py`
- Structural compression + token clipping → **sidekick** `context_budget.py`
- Semantic compression → **LLMLingua-2**
- Memory → **mem0** (simple store; Graphiti deferred to v0.3)
- Verification → **Instructor/Pydantic** + **RAGAS** (citation/groundedness)
- Token counting → `tiktoken` via LiteLLM

**Explicitly NOT in v0.1:** full multi-agent scheduler, mixture reasoners
(plan-worker-critic/debate), cost-aware scheduling, full policy language, production
UI, knowledge graph DB, Plan Cache, distributed execution, enterprise auth, CP-SAT,
dynamic plugin loading.

**Exit benchmarks:**
- *(plan §11)* answer questions over a 500-page corpus; compare (1) naive long
  context, (2) vector-only RAG, (3) hybrid RAG, (4) hybrid+reranker, (5) ContextOS
  planned context — on accuracy, citation correctness, tokens, latency, cost.
- *(new — Developer Time / LOC)* re-implement one example pipeline hand-rolled vs.
  ContextOS and record the LOC delta. This is both a real maintenance metric and the
  sharpest selling point.

**Ship gate:** ContextOS plan beats hybrid+reranker on cost-at-equal-accuracy,
produces a replayable trace for every run, `explain()` returns a populated plan, and
`simulate()` returns a cost/latency/token envelope (with intervals from the
cost-model statistics, however wide at first).

---

## v0.2 — Constrained selection + Knowledge graph + Plan Cache

**Goal:** decisions respect *interacting* hard constraints, context stops being
isolated chunks, and identical questions stop re-planning.

- `optimizer/cpsat.py` → **OR-Tools CP-SAT**, introduced exactly when ≥3 constraints
  interact across a multi-step plan (cost ∧ latency ∧ privacy ∧ tokens). The
  `optimizer/knapsack.py` path remains the v0.1 fast lane for single-constraint cases.
- `knowledge/graph_kuzu.py` → **Kùzu** Knowledge graph: `contains/derived_from/cites/
  contradicts/supersedes/depends_on` edges. Planner reads it for staleness +
  provenance, and it supplies the **source-version fingerprint** the Plan Cache keys on.
- **`plancache/`** → cache `Intent → Execution Graph → cost estimates`. Keyed on
  normalized intent (semantic match) + source-version fingerprint + policy/permission
  context + constraint envelope; invalidated on source/policy/capability/permission
  change or TTL. **Correctness rides on deterministic replay (principle #7)** — the
  reason it can ship only once the graph provides versioned sources.

**Exit benchmark:** stale-information detection + "lost-in-the-middle" improve once
the graph informs selection; CP-SAT finds feasible plans where greedy knapsack fails;
**planner cost-per-1000-identical-queries drops by the plan-cache hit rate** (the
"1000 people ask the same Kubernetes error" case planned once, not 1000×).

---

## v0.3 — Knowledge memory lifecycle + the online learning loop (the moat)

**Goal:** the cost model *learns* from observed outcomes. This is where ContextOS
stops being a clever static planner and becomes a runtime that improves.

- Memory (the lifecycle sub-concern of the Knowledge Layer) → migrate mem0 →
  **Graphiti (Zep)** bi-temporal KG: versioned, auditable, contradiction detection,
  expiration, promotion, compaction.
- Learning loop:
  - **River (online contextual bandit)** — per-request "which retrieval / which
    model" decisions, learning from streaming feedback. (Vowpal Wabbit if scale.)
  - **Optuna (offline batch)** — global params: `top_k`, reranker threshold,
    compression ratio, parallel workers.
  - Feedback closes onto the **cost model's estimates** (not the solver).
- Cost-model **statistics mature**: enough samples accumulate that calibration is
  meaningful, `simulate()` intervals tighten, and `explain()` can show trustworthy
  estimate-vs-actual. (Collection started v0.1; this is where it becomes useful.)
- First **mixture reasoners** — `reasoner/mixture.py` adds `plan_worker_critic` and
  `debate` strategies behind the `reason` node, selectable by the planner per intent.
- Memory lifecycle is policy-gated (permission checks, visibility).

**Exit benchmark:** reproducibility holds *and* routing cost-per-correct-answer
drops over a fixed eval stream as the bandit warms up (offline-replay evaluation
from Langfuse traces).

---

## v0.4 — Agent Scheduler + full Execution Graph

**Goal:** multi-agent execution with contracts, not agent sprawl — and the Execution
Graph IR grows past a DAG.

- `execution/graph.py` gains the non-DAG shapes the IR was designed for: conditional
  branches, loops/retries, human-approval gates, agent fan-out, merge, **rollback**.
  `execution/dagster_compile.py` compiles these to **≥1 Dagster run** (the runtime
  drives iterations/rollback across runs; Dagster stays a DAG-of-assets executor).
- `agents/scheduler.py` → bind **sidekick** orchestrator (waves, isolated worktrees,
  acceptance checks, merge) as `delegate` nodes in the Execution Graph.
- Every agent carries: role, input context, output contract, token budget, timeout,
  permissions, verification requirement. Verifier nodes are first-class in the graph.

**Exit benchmark:** multi-agent research/incident-review example beats single-agent
on correctness at bounded cost, with zero uncontrolled fan-out (sidekick's validated
0-conflict / 0-human-wait profile).

---

## v0.5 — Policy Engine

**Goal:** the runtime is *governed* — data movement, routing, and execution rights
are declarative and enforced.

- `policy/opa.py` → **Open Policy Agent (Rego)** for routing/access/data-movement
  decisions (the §5.10 YAML becomes real Rego).
- `policy/presidio.py` → **Microsoft Presidio** PII/secret detection drives
  "sensitive data → local model only."
- `policy/safety.py` → **agentic-os** safety scan as the fast inline pre-filter;
  approval gates + append-only audit log from `agentic-os` control plane.

**Exit:** policy tests prove `no_secret_exfiltration`, `require_human_approval_for_
prod`, and `local_model_for_sensitive_docs` are enforced, not advisory.

---

## v1.0 — Production Runtime

- Stable SDK + **dynamic plugin loading** (entry-point registry; every subsystem —
  retriever/model/store/knowledge/verifier/compression/router/policy/planner — a
  loadable plugin). The contracts existed since v0.1; v1.0 makes them pluggable
  out-of-tree.
- **OpenTelemetry** export (semantic conventions via OpenLLMetry)
- Full benchmark suite (plan §11 + Developer-Time/LOC): retrieval accuracy, answer
  correctness, citation accuracy, token reduction, latency, cost, lost-in-the-middle,
  stale detection, reproducibility, verification effectiveness, **LOC eliminated**
- `costmodel/estimators.py` path from heuristic → learned → neural (like PG statistics)
- Reference server + full docs (concepts / tutorials / api_reference)
- Ray Tune for distributed optimization experiments (if needed at scale)

---

## v2 — Cost-aware Scheduler (the OS pipeline)

The Planner/Scheduler boundary (the Execution Graph) exists from v0.1; v2 promotes
**scheduling** from a trivial topo-sort to a first-class, cost-aware stage:

```
Intent → Planner → Execution Graph → Scheduler → Execution     (mirrors every OS)
         (what)      (the boundary)    (when/where)
```

- `scheduler/schedule.py` gains cost-aware behavior: reorder for latency, budget-aware
  concurrency, retry/backpressure policy, resource-class placement.
- Because the boundary IR already exists, this **adds a plugin implementation** — it
  does not reshape the runtime. That is the payoff of splitting Planner from Scheduler
  in v0.1 even while the scheduler was trivial.

---

## Sequencing rationale (why this order)

| Decision | Reason |
|---|---|
| Plugin contracts + observability + EXPLAIN in **v0.1** | Everything downstream depends on the contracts; the trace is the learning loop's fuel; EXPLAIN is near-free once a Plan object exists and is the feature that makes the runtime legible. |
| Planner *split* + `costmodel/` package in **v0.1** | The three responsibilities (intent / candidates / optimize) and a first-class cost model are cheap to separate now and painful to retrofit; they're where all future research happens. |
| Execution Graph IR in **v0.1**, branches/loops in **v0.4** | Emit the IR from the start (even linear) so non-DAG shapes slot in without a rewrite; the shapes themselves only matter once agents arrive. |
| Reasoner + Scheduler *contracts* in **v0.1**, capability later | The `reason` node and the Planner/Scheduler boundary must exist from the start; mixtures (v0.3–v0.4) and cost-aware scheduling (v2) then fill in behind stable seams instead of forcing a reshape. |
| Cost-model **statistics** collected **v0.1**, useful **v0.3** | Calibration needs sample history; logging estimate-vs-actual from run one is cheap, and `simulate()`/trust depend on it. |
| Scheduler cost-aware only at **v2** | Planning value comes first; scheduling optimization is a separate, later problem — and the boundary already exists, so it waits cheaply. |
| CP-SAT in **v0.2**, not v0.1 | The decision space is tiny at first; knapsack suffices. CP-SAT earns its keep only when ≥3 constraints interact. |
| Plan Cache in **v0.2**, not v0.1 | It can only be *correct* once the Knowledge graph supplies versioned sources to key/invalidate on (deterministic replay, principle #7). |
| Learning loop in **v0.3**, after observability | Bandits/BO need logged plan→outcome data, which only exists once v0.1 traces flow. |
| Policy in **v0.5**, after agents | Policy governs *actions*; meaningful actions arrive with the agent scheduler (v0.4). The fast `safety.py` pre-filter ships earlier as a stopgap. |
| Dynamic plugin loading in **v1.0**, contracts in **v0.1** | Stable interfaces first; an out-of-tree registry before the interfaces are proven is premature. |
| Benchmarks **every phase** (incl. Developer-Time/LOC) | ContextOS must prove it is more than architectural language; each phase has an exit gate, not just features. |
