# Context Runtime

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE) ![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg) ![Go](https://img.shields.io/badge/Go-1.22%2B-00ADD8.svg) [![NVIDIA Inception](https://img.shields.io/badge/NVIDIA-Inception%20Program%20Member-76B900.svg)](https://www.nvidia.com/en-us/startups/)

> **🚀 NVIDIA Inception Program Member** — ReDevOps is a member of the NVIDIA Inception Program, supporting startups advancing AI and accelerated computing. Membership provides access to NVIDIA technology, technical resources, and the startup ecosystem. It does not imply product endorsement by NVIDIA.

**An efficiency optimizer for a fleet of apps** — a database query planner for LLM
context. The application says *"I need an answer"*; the runtime decides what the model
sees — what to retrieve, compress, route, and verify — emits an inspectable,
replayable plan, and **learns from the outcome**. It does for AI context what query
planners did for SQL. See [POSITIONING.md](./POSITIONING.md) for the thesis.

It optimizes any app with (a) a decision point about what context/config to use and
(b) a measurable outcome. Eleven tenants are built and green (each number is the
learned-vs-baseline reward its offline example in `examples/` prints):

| Tenant | Context Runtime tunes | Result |
|---|---|---|
| **sidekick** | which skills to recall · budget | drop-in for `SkillStore`; **67% vs 33%** naive baseline acceptance |
| **redevops-rag** | `pool · limit · threshold · rerank …` per query | `ContextRuntimeRetrieverTuner`; **0.780 vs 0.428** vs fixed default |
| **edge-sentinel (SOC)** | which sources to pull per alert (CrowdSec · threat-intel · EDR) | tool-using + approval-gated; **0.900 vs 0.800** always-full baseline |
| **growth-engine** | which attribution window + source bundle per lead-source query | **7.851 vs 5.282** vs fixed window |
| **control-tower** | which Metabase query set per "ask anything" question | **5.326 vs 1.643** vs core query set |
| **agentic-billing** | which usage/invoice/dunning signals to pull per account | **4.122 vs 2.442** vs full-stack |
| **social-autopilot** | which channel/timing/content strategy per goal | **3.875 vs 0.773** vs fixed strategy |
| **agentic-support** | which KB/tickets/account context to retrieve per ticket | **3.679 vs 2.394** vs full-context |
| **agentic-books** | which ledgers/reports to pull per books question | **3.632 vs 2.430** vs full-books |
| **market-radar** | which competitor watches to sweep per intel question | **3.611 vs 0.403** vs full-sweep |
| **agentic-compliance** | which rule-family evidence to pull per finding | **3.562 vs 2.463** vs full-evidence |

> Reward numbers are single-run outputs of each tenant's `examples/` script — re-run to reproduce (they drift with model/data changes). The **sidekick** and **redevops-rag** rows are current; the others may need a refresh.

```bash
PYTHONPATH=. python examples/sidekick_learning.py   # discrete-strategy bandit
PYTHONPATH=. python examples/rag_tuning.py          # numeric-knob tuning
PYTHONPATH=. python examples/soc_triage.py          # tool-using cybersecurity tenant
```

Plus the **ToolPlugin** seam (`context_runtime/tools/` — how plans reach external systems,
with an approval-gated audit trail) and **trace exporters** (`context_runtime/observability/
exporters.py` — JSONL offline, or Langfuse / OpenLLMetry-OTel when the extras are
installed).

> Status: **runnable reference implementation — Python and Go at feature parity.** Runs fully
> offline with stub models; real bindings are wired and lazy-imported: LiteLLM models, and
> [redevops-rag](https://github.com/redevops-io/redevops-rag) / DuckDB / Postgres retrieval
> (BM25 · dense · hybrid · graph-PPR · community · sharded routing · cross-modal). Beyond the
> initial slice it now ships score **calibration**, a **learned quality-aware planner**, and
> **EXPLAIN** for the retrieval decision (see [Also shipped](#also-shipped)). See
> [SPEC.md](./SPEC.md) §10 for the conformance checklist these tests assert against.

## The agentic fleet

The reward table above is the measured slice; the full fleet is **15 native agent services + a
control plane**, mirrored from the live demo at **[demo.redevops.io](https://demo.redevops.io)**. Each
service wraps a mature OSS core and exposes `/health`, a Material-3 dashboard, and its agent endpoints —
the Context Runtime is the decision layer *inside* each (which context / tools / config per request),
while the core does the domain work:

| App | OSS core | App | OSS core |
|---|---|---|---|
| agentic-billing | Lago | market-radar | changedetection |
| agentic-books | ERPNext | edge-sentinel (SOC) | CrowdSec |
| agentic-support | Chatwoot | agentic-compliance | OpenSCAP |
| agentic-crm | ERPNext (CRM) | control-tower | Metabase |
| social-autopilot | Postiz | growth-engine | Umami |
| lifecycle | Listmonk | guide | redevops-rag |

Plus **outreach-engine** (Twenty CRM), and **growth-assistant** + **agentic-privacy** (both on ERPNext — leads / contacts, not books), and the **control plane**
(`agentic_os`: deploy / observe / approve, `/m/<app>` proxy + a module catalog). Most have the offline
reward examples shown above (`examples/<app>.py`); the rest are native realizations without a
standalone tuner.

The **enterprise line** (`CR-enterprise`, v4) adds the commercial open-core layer that *composes with*
this optimizer rather than replacing it: **Policy-Constrained Planning** — provider / budget / PII /
data-residency / mandatory-verification / human-approval predicates define the *feasible* plan set
**before** the cost model ranks, so every execution can be proven within governance — and
**Trust-Aware Execution**, a trust ledger built from operator acceptance / overrides / regenerations /
grounded abstention, so the planner optimizes for the strategy operators will actually rely on. (The
mission-runtime line has since shipped — the v6 mission cockpit is live at demo.redevops.io.)

## Install

```bash
pip install -e .                 # core (offline stub path, zero heavy deps)
pip install -e ".[litellm]"      # real models across 100+ providers
pip install -e ".[rag]"          # redevops-rag — single-hop hybrid retrieval
pip install -e ".[hipporag]"     # HippoRAG — multi-hop graph retrieval (the planner picks per query)
```

**Single-hop vs multi-hop is a per-query decision.** The planner classifies intent and
routes: BM25/hybrid (redevops-rag) when the answer is in one chunk, **graph (HippoRAG)**
when it lives in the connections between documents — and the cost model only pays the
graph premium when it's warranted. `python examples/hop_routing.py` shows single-hop
missing the bridge document that multi-hop surfaces.

## 30-second tour

```python
from context_runtime import ContextRuntime, SourceRef

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
context-runtime --corpus ./docs run "what's our incident process?"
context-runtime --config context_runtime.yaml explain --analyze "why did deploy X fail?"
```

## Core seams (SPEC) & bindings

The six plugin seams and their initial slice; several "real bindings" below are now shipped (see
[Also shipped](#also-shipped)).

| Seam (SPEC) | initial slice | Real binding |
|---|---|---|
| **Planner trio** (intent/candidate/optimizer) | rule-table intent → candidate gen → heuristic cost model | — *(the genuinely new core)* |
| **Cost model + statistics** | `PlanScore` weighted utility + `pg_statistic`-style calibration | learned/neural (v0.3+) |
| **Optimizer** | knapsack / greedy-by-utility over the feasible set | OR-Tools CP-SAT (v0.2) |
| **Execution Graph IR** | linear graph carrying branch/loop/rollback kinds | full shapes (v0.4) |
| **Scheduler** | topo-sort waves | Dagster / cost-aware (v2) |
| **Reasoner** | `SingleShotReasoner` (one model) | mixtures: plan-worker-critic (v0.3+) |
| **Model plugin** | offline `StubModel` | **LiteLLM** + native cost-tiered routing |
| **Retriever/Store** | `InMemoryStore` (keyword) | **redevops-rag** (DuckDB+BM25+RRF+rerank) |
| **Compression** | sidekick `clip` structural pack | LLMLingua-2 semantic (v0.1 optional) |
| **Verifier** | citation/grounding check | RAGAS / Instructor |
| **Observability** | in-process `Trace` + JSON | OpenLLMetry → Langfuse |
| **Plan Cache** | null/always-miss stub | semantic cache (v0.2) |

## Also shipped

Beyond the initial slice, in both the Python source-of-truth and the Go port (feature parity):

- **Retrieval as a routable capability** — BM25 · dense (fastembed ONNX) · hybrid (RRF) · graph
  (Personalized-PageRank multi-hop) · community · **sharded coverage routing** for heterogeneous
  corpora · **cross-modal** (CLIP image, ColPali multi-vector, video segments — Python) ·
  quantized ANN (TurboVec) · two-stage cost-gated fusion.
- **DSpark-inspired planning** — score **calibration** (isotonic → `P(relevant)`), grounded
  **abstention**, and a **load-aware sizer** that prunes the expensive stage. Measured v1→v2 lift
  in [BENCHMARKS.md](./BENCHMARKS.md).
- **Quality-aware routing** — a quality ledger that tracks learned quality **apart from cost** per
  intent, so a genuinely better arm (or provider) wins at equal cost. Opt-in.
- **EXPLAIN** (`context_runtime/explain.py`, `POST /librechat/explain`, `examples/explain.py`) —
  the DB EXPLAIN-ANALYZE analogue for the retrieval decision: every candidate arm ranked with its
  quality/cost decomposition, the per-method trace with calibrated `P(relevant)`, served/abstain,
  and reward provenance. Read-only. Visualized at [redevops.io/planner](https://redevops.io/planner).
- **LiteLLM model binding** + native cost-tiered routing; **DuckDB** and **Postgres** stores.

## Benchmarks

Runnable results in [BENCHMARKS.md](./BENCHMARKS.md). Headline: the **v1 → v2 table measured in
both runtimes** (Python + Go) — learned-policy precision **+14.6 / +11.3 pts**, abstention
**0 → 100%**, expensive-stage depth **−62%**. Plus retrieval over a **heterogeneous financial ×
medical corpus** (coverage routing cuts cross-domain pollution 22→0 with recall intact), the
**3-index chat memory** tenant (+2.93 learned vs read-all-three), and **parallel sharded fusion**
(5.8× fan-out). Every number is produced by an example in [`examples/`](./examples).

## Architecture

The decision layer is thin; the substrate is reused. See:
- [ARCHITECTURE.md](./ARCHITECTURE.md) — the layered design and the cost-based optimizer loop
- [SPEC.md](./SPEC.md) — the normative interface contracts (six plugin seams, IR, trace, plan-cache key)
- [ROADMAP.md](./ROADMAP.md) — v0.1 → v2 phasing with per-phase exit benchmarks

## Test

```bash
pip install -e ".[dev]" && pytest      # 360+ tests; test_conformance.py == SPEC §10
```
