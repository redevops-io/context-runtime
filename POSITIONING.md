# ContextOS — Positioning

> **ContextOS is an efficiency optimizer for a fleet of apps.**
>
> Not a RAG library, not an agent framework, not a model SDK. It is the layer that
> sits *underneath* those and decides — for any app that has to choose what context or
> config to use — the cheapest choice that still meets the goal, then learns from the
> outcome.

## The one-sentence thesis

Most AI systems hard-code a hundred small decisions: top_k, which model, how much to
compress, which skill to recall, whether to rerank, how big a budget. Each is a
guess, frozen at development time. **ContextOS turns those guesses into a runtime
decision that is planned, measured, and improved** — the way SQL replaced
hand-written scan/join/sort plans with a planner that decides from statistics.

## The mental model: a query planner for context

```
Application:   "I need an answer."           (intent)
ContextOS:     plan → execute → observe → learn   (the decision + the feedback)
```

The application stops saying *"retrieve these chunks, rerank them, summarize, send to
Claude."* It states a goal and a budget; ContextOS produces an **execution plan**,
runs it through reused substrate, and records what happened so the next plan is
better. EXPLAIN/SIMULATE make every decision inspectable, like `EXPLAIN ANALYZE`.

## Why "fleet of apps", not "a RAG tool"

The decisive realization: ContextOS is not coupled to retrieval. It optimizes **any
app with two properties**:

1. a **decision point** — a choice about what context/config to use, and
2. a **measurable outcome** — the app's own success metric.

Given those, the integration is always the same four-seam wrap:

```
        ┌─────────── ContextOS ───────────┐
host →  │ plan (intent → choice) ──────────┼──→  host executes the choice
app     │                                  │            │
        │ learn ←── observe(outcome) ←──────┼────────────┘  (the app's metric = reward)
        └──────────────────────────────────┘
```

The learning core (`integrations/bandit.py` — a contextual bandit, the v0.1 stand-in
for v0.3 River) and the cost-model statistics are **shared across all tenants**. Only
the *arms* (what to choose) and the *reward* (how to score it) are app-specific.

## The tenants (all redevops repos)

| Tenant | Decision point ContextOS optimizes | Reward (the app's own metric) | Status |
|---|---|---|---|
| **sidekick** (coding agent) | which skills to recall · bundle size · token budget | acceptance rate · first-try · tokens | **built, green** — drop-in for `SkillStore`; 67% vs 33% naive baseline |
| **redevops-rag** (retrieval) | `pool · limit · vector_threshold · recency · keyword priors · rerank` per query intent | retrieval quality − efficiency penalty | **built, green** — `ContextOSRetrieverTuner`; 0.773 vs 0.323 fixed default |
| **edge-sentinel** (SOC) | which sources to pull per alert (CrowdSec · threat-intel · EDR) | correct verdict − source cost | **built, green** — tool-using + approval-gated; 0.900 vs 0.800 always-full |
| **business modules** (billing · support · BI · compliance …) | which sources/cores to query · which model tier | task success · cost-per-good-answer | next — each becomes a tenant with a goal + a metric |

These prove the pattern generalizes across very different decision types: sidekick
chooses among **discrete strategies**, redevops-rag tunes **numeric knobs**,
edge-sentinel selects **sources/tools with side effects**. Same bandit, same
cost-model, same wrap. That is the whole bet.

**ContextOS *is* the control plane.** It supersedes the earlier `agentic-os` fleet
controller — the routing, approval/safety, and audit-log capabilities prototyped there
now live natively in ContextOS (`adapters/model_litellm.py`, `tools/`,
`observability/`). The business modules that the old fleet ran become **tenants**: each
gets a goal, a metric, and a learned policy, instead of a hand-wired controller.

## Why this is a more durable position

- **It composes, it doesn't compete.** LangGraph/CrewAI/LlamaIndex *are* apps with
  decision points; ContextOS optimizes them rather than replacing them. Frameworks
  build on top; ContextOS sits beneath.
- **It compounds.** Every run of every tenant produces a `(plan, outcome)` row that
  sharpens the shared cost model. The fleet gets more efficient the more it runs —
  the asset is the accumulated statistics, not the code.
- **It is the fleet's control plane.** ContextOS plans, optimizes, and learns every
  agent's context/config decisions under one budget and one trace — the role the
  earlier `agentic-os` controller filled, now done by a learned planner instead of
  hand-wired routing.

## What "efficiency" concretely means

Not just accuracy, and not just cost — the **frontier between them**. The reward in
every tenant is *quality minus an efficiency penalty*, so ContextOS converges on the
**cheapest configuration that's still good enough**, per intent. That is the thing no
app tunes by hand because the search space is too large and the right answer drifts —
exactly the job a learned planner exists to do.

## Roadmap implication

This reframes the [ROADMAP](./ROADMAP.md): the tenants are how each capability earns
its place. sidekick exercised the **discrete-strategy bandit**; redevops-rag exercises
**numeric tuning** (the Optuna/BO half); edge-sentinel exercises **tool/source
selection with side effects**; the business modules will exercise **model routing
under budget**. v0.3's River loop and v0.2's CP-SAT are not abstract milestones — they
are the upgrades the tenants ask for once the v0.1 bandit's limits show.
