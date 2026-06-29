# ContextOS

**An efficiency optimizer for a fleet of apps** — a database query planner for LLM
context. The application says *"I need an answer"*; the runtime decides what the model
sees — what to retrieve, compress, route, and verify — emits an inspectable,
replayable plan, and **learns from the outcome**. It does for AI context what query
planners did for SQL. See [POSITIONING.md](./POSITIONING.md) for the thesis.

It optimizes any app with (a) a decision point about what context/config to use and
(b) a measurable outcome. Two tenants are built and green:

| Tenant | ContextOS tunes | Result |
|---|---|---|
| **sidekick** | which skills to recall · budget | drop-in for `SkillStore`; **67% vs 33%** naive baseline acceptance |
| **redevops-rag** | `pool · limit · threshold · rerank …` per query | `ContextOSRetrieverTuner`; **0.773 vs 0.323** vs fixed default |
| **edge-sentinel (SOC)** | which sources to pull per alert (CrowdSec · threat-intel · EDR) | tool-using + approval-gated; **0.900 vs 0.800** always-full baseline |

```bash
PYTHONPATH=. python examples/sidekick_learning.py   # discrete-strategy bandit
PYTHONPATH=. python examples/rag_tuning.py          # numeric-knob tuning
PYTHONPATH=. python examples/soc_triage.py          # tool-using cybersecurity tenant
```

Plus the **ToolPlugin** seam (`contextos/tools/` — how plans reach external systems,
with an approval-gated audit trail) and **trace exporters** (`contextos/observability/
exporters.py` — JSONL offline, or Langfuse / OpenLLMetry-OTel when the extras are
installed).

> Status: **v0.1 vertical slice.** Runs fully offline with stub plugins; the real
> [redevops-rag](https://github.com/redevops-io/redevops-rag) retrieval and LiteLLM
> model bindings are wired and lazy-imported. See [SPEC.md](./SPEC.md) §10 for the
> conformance checklist these tests assert against.

## Install

```bash
pip install -e .                 # core (offline stub path, zero heavy deps)
pip install -e ".[litellm]"      # real models across 100+ providers
pip install -e ".[rag]"          # redevops-rag hybrid retrieval (DuckDB + BM25 + rerank)
```

## 30-second tour

```python
from contextos import ContextRuntime, SourceRef

rt = ContextRuntime.default(docs)          # offline: stub model + in-memory store

# RUN — the core abstraction (plan → build_context → execute → verify)
res = rt.run("Explain why deployment X failed",
             sources=[SourceRef("docs", "docs")],
             constraints={"max_cost_usd": 2.0, "require_citations": True})
print(res.answer, res.cost_usd, res.trace)

# EXPLAIN — debug the plan like SQL (add analyze=True for EXPLAIN ANALYZE)
ex = rt.explain("Explain why deployment X failed")
print(ex.intent.bucket, len(ex.candidates), ex.chosen.score.total)

# SIMULATE — forecast cost/latency/tokens with confidence intervals, no execution
sim = rt.simulate("Explain why deployment X failed")
print(sim.expected_cost_usd, sim.expected_models, sim.based_on_samples)
```

Or from the CLI / config:

```bash
PYTHONPATH=. python examples/incident_review.py
contextos --corpus ./docs run "what's our incident process?"
contextos --config contextos.yaml explain --analyze "why did deploy X fail?"
```

## What's implemented (v0.1)

| Seam (SPEC) | v0.1 implementation | Real binding (lazy) |
|---|---|---|
| **Planner trio** (intent/candidate/optimizer) | rule-table intent → candidate gen → heuristic cost model | — *(the genuinely new core)* |
| **Cost model + statistics** | `PlanScore` weighted utility + `pg_statistic`-style calibration | learned/neural (v0.3+) |
| **Optimizer** | knapsack / greedy-by-utility over the feasible set | OR-Tools CP-SAT (v0.2) |
| **Execution Graph IR** | linear graph carrying branch/loop/rollback kinds | full shapes (v0.4) |
| **Scheduler** | topo-sort waves | Dagster / cost-aware (v2) |
| **Reasoner** | `SingleShotReasoner` (one model) | mixtures: plan-worker-critic (v0.3+) |
| **Model plugin** | offline `StubModel` | **LiteLLM** + agentic-os tier policy |
| **Retriever/Store** | `InMemoryStore` (keyword) | **redevops-rag** (DuckDB+BM25+RRF+rerank) |
| **Compression** | sidekick `clip` structural pack | LLMLingua-2 semantic (v0.1 optional) |
| **Verifier** | citation/grounding check | RAGAS / Instructor |
| **Observability** | in-process `Trace` + JSON | OpenLLMetry → Langfuse |
| **Plan Cache** | null/always-miss stub | semantic cache (v0.2) |

## Architecture

The decision layer is thin; the substrate is reused. See:
- [ARCHITECTURE.md](./ARCHITECTURE.md) — the layered design and the cost-based optimizer loop
- [SPEC.md](./SPEC.md) — the normative interface contracts (six plugin seams, IR, trace, plan-cache key)
- [ROADMAP.md](./ROADMAP.md) — v0.1 → v2 phasing with per-phase exit benchmarks

## Test

```bash
pip install -e ".[dev]" && pytest      # 18 tests; test_conformance.py == SPEC §10
```
