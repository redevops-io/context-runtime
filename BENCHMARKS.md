# Context Runtime — Benchmarks

The single source of truth for Context Runtime's measured results. Every number here is
produced by a runnable harness in this repo (or a linked one) — no invented numbers.

## How to read this

Each Context Runtime version adds **one capability**, and each was originally measured on the
benchmark that best stresses that capability — different datasets, different metrics. That made
the per-version numbers impossible to read against each other (a *recall* on LongMemEval next to
an *NDCG* on TEMPO is not a comparison).

So this document leads with **one comparable ladder**: a single mixed query stream, one answer
model, one judge, capabilities turned on **cumulatively**, reported as **end-to-end answer
accuracy + tokens of context passed**. Every rung answers the *same* questions, graded the *same*
way. Below the ladder, a **drill-down per version** isolates each mechanism on its home-turf
benchmark (that is where the mechanism's own effect is measured cleanly), followed by the
standalone studies (heterogeneous shards, chat-memory, fusion, tenants) and the methodology.

- Ladder → *are the capabilities worth it, end to end, on one axis?*
- Drill-downs → *does each individual mechanism do what it claims, measured precisely?*

---

## The capability ladder (headline — the comparable view)

**`redevops-rag/benchmarks/eval_ladder.py`** — one mixed stream of **45 questions** (15 each from
PopQA / MuSiQue / LongMemEval = lookup / multi-hop / temporal), each question's corpus polluted
with 30 distractor passages drawn from the rest of the stream so context management actually
matters. The **answer model is fixed** (Qwen3.6-35B-A3B) and the **judge is fixed** (grok-4.5), so
every rung's accuracy difference comes from the *context it was handed*, nothing else. Capabilities
are cumulative: each row is the row above **plus one mechanism**.

| Config | Mechanism added | Answer acc | Tokens/q | lookup | multi-hop | temporal |
|---|---|---|---|---|---|---|
| **v1 base** | naive: large-k retrieve, dump a big context | **0.533** | 4014 | 1.00 | 0.47 | 0.13 |
| **+v2** | calibrated gating · load-aware sizer · abstention | 0.444 | **1582** | 1.00 | 0.27 | 0.07 |
| **+v3** | online bandit picks the retrieval arm per bucket | 0.444 | 1647 | 1.00 | 0.27 | 0.07 |
| **+v4** | LLM knowledge routing (temporal→recency, multi-hop→fan-out) | 0.444 | 1628 | 1.00 | 0.27 | 0.07 |
| **+v5** | DIVER reasoning retrieval (expand → union → listwise rerank) | **0.511** | 1686 | 1.00 | **0.47** | 0.07 |

**What the ladder shows — the endpoint reaches the brute-force dump's accuracy (0.51 vs 0.53) at
42% of the tokens (1686 vs 4014)** — and the middle rows show *why less context alone isn't the
answer*:

- **Lookup is already solved** (1.00 everywhere): single-fact questions are easy once retrieved, so
  routing to the cheapest representation is a pure efficiency win.
- **Naive compression (v2) overshoots.** Cutting to a small top-k drops the **low-similarity bridge
  passages** multi-hop needs — a passage that shares *no query term* and only connects via a bridge
  entity ranks low, so a similarity gate prunes it (multi-hop 0.47 → 0.27). Tokens fall 2.5×, but so
  does hard-regime accuracy. Cheaper is not free.
- **Bandit + routing (v3, v4) don't recover it.** The problem isn't budget or the representation
  *label* — it's retrieval *reach* on multi-hop. Their value shows in the drill-downs (routing
  accuracy, recall), not in end-to-end answer accuracy with a small answerer.
- **DIVER (v5) recovers the accuracy at the same low cost.** Query expansion surfaces the bridge
  passage under a sub-query, and the listwise rerank floats it to the top — multi-hop returns to
  0.47 while tokens stay ~1.7k. *This* is "better context beats more context": the win came from
  **smarter** retrieval, not **more** context.
- **Temporal stays at the small-model ceiling** (~0.1). Retrieval recall on these tasks is solved
  (see v4 drill-down, SimGraph 1.000); the bottleneck is a 35B model *aggregating* facts across
  sessions, not finding them. Bigger context and a stronger model move this (see v5 context-size).

> The ladder is deliberately **not** a monotonic "every rung goes up" curve — that would hide the
> real trade-off. It shows that compression has a cost, and that the cost is bought back by
> *reasoning-aware retrieval*, not by dumping more tokens.

Reproduce: `N=15 .venv/bin/python benchmarks/eval_ladder.py` (in `redevops-rag`).

---

# Drill-downs — each mechanism on its home-turf benchmark

## v1 → v2 — calibration, abstention, and the load-aware sizer (both runtimes)

**`examples/consolidated_benchmark.py`** — the *same* seeded, ground-truth retrieval simulation run
in the Python source-of-truth **and** the Go port, merged into one table. v2 adds the DSpark-inspired
upgrades: calibrated per-passage relevance folded into the reward, grounded abstention, and the
load-aware expensive-stage sizer.

| Metric | Py v1 | Py v2 | Δ | Go v1 | Go v2 | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| Learned-policy precision | 67.6% | 82.2% | ▲ +14.6 pts | 84.6% | 95.9% | ▲ +11.3 pts |
| Abstention recall (unanswerable caught) | 0.0% | 100.0% | ▲ +100 pts | 0.0% | 100.0% | ▲ +100 pts |
| False-abstain rate (answerable dropped) | 0.0% | 0.0% | — | 0.0% | 0.0% | — |
| Expensive-stage depth (passages) | 8.00 | 3.00 | ▼ −62% | 8.00 | 3.00 | ▼ −63% |
| Precision after the sizer | 37.5% | 100.0% | ▲ +62.5 pts | 37.5% | 100.0% | ▲ +62.5 pts |

_40-seed average; precision headlined at β=0.9 (the calibration-trust knob; shipped default 0.5).
Go is an **independent re-implementation on identical methodology** — the two runtimes agree
directionally on every effect: the improvement is a property of the design, not one language's
simulation. Absolute precision differs because each runtime uses its own strategy set._

**The β sweep** (`examples/dspark_calibration_bench.py`, Go twin `cmd/dsparkbench`) isolates the
reward effect: served true-precision **v1 67.6% → v2 68.9% (β0.5) / 74.3% (β0.7) / 82.2% (β0.9)**,
monotone in how much the reward trusts calibrated relevance over the coarse coverage judge (β=0.3
is −2 pts, so the live default is 0.5). Abstention and the sizer are deterministic; only the reward
needs seed-averaging. Note v2's defining property — **0% false-abstention** — is exactly the
guardrail the *ladder's* aggressive v2 shows the cost of violating in a polluted, hard-regime
setting.

## v3 — online adaptation under drift

**`examples/online_vs_static_bench.py`** — the best plan drifts mid-run (a model upgrade, a corpus
shift). A **static** v1/v2 planner is pinned to the now-stale plan; the **v3 online** planner
re-explores, and recency-weighted (discounted) learning lets it track the shift.

| Metric | v2 (static) | v3 (online) | Δ |
| --- | --- | --- | --- |
| Post-drift served-plan reward | 0.20 | **0.67** | ▲ +0.47 |
| Post-drift reward, online **without** discounting | 0.20 | 0.34 | +0.14 |

_Seeded drift simulation, 24-seed average; post-drift oracle = 0.80._ Discounting is the mechanism:
it converts online learning from a marginal +0.14 into recovery of most of the oracle reward (0.67
of 0.80). Companion `online_learning` shows the off-policy variant serving `graph` after the
production reward (graph 0.9 / hybrid 0.3) overturns the stale prior (hybrid 0.8). This axis is
**orthogonal to** and preserves the v1→v2 calibration gains — and, as the ladder confirms, it does
not change steady-state answer accuracy; its value is adaptation, which only appears under drift.

## v4 — knowledge routing: *Better Context Beats More Context*

The full study behind the [article](https://redevops.io/blog/better-context-beats-more-context)
lives in **[`benchmarks/context-vs-model/RESULTS.md`](./benchmarks/context-vs-model/RESULTS.md)**
(deep lab reports in [`reports/`](./benchmarks/context-vs-model/reports); the Go replication in
[`context-runtime-go/benchmarks/routing/RESULTS.md`](https://github.com/redevops-io/context-runtime-go)).
Headline results:

**Routing — the foundation** (grades the classifier's label against the regime each dataset was
built for; answer-model-independent, contamination-immune):

| Regime | Keyword router (`RuleIntentAnalyzer`) | **LLM router** (Python / Go) |
|---|---|---|
| PopQA → document | 0.95 | **1.00** / 0.97 |
| MuSiQue → graph | **0.00** | **0.90** / 0.62 |
| LongMemEval → temporal | **0.00** | **1.00** / 0.68 |

The keyword head is blind to *implicit* intent ("Who is the spouse of the Green performer?" carries
no multi-hop string → defaults to document, graph engine never reached). The lift is a property of
the analyzer **type**, not model size — the LLM head reaches 0.90–1.00 even at 27B. The Go
`HybridIntentAnalyzer` (LLM-on-doubt, keeps the cheap keyword head in the loop) is more conservative
(0.62/0.68) — the honest cost of not always calling the model.

**Lesson 1 — heavy infrastructure must earn its cost, and the answer flips by regime:**

| Regime | Simple (non-lossy) | Heavy (LLM engine) | Verdict |
|---|---|---|---|
| Temporal (LongMemEval recall) | **SimGraph 1.000** | Graphiti 0.733 | drop the heavy engine |
| Multi-hop (MuSiQue recall@4) | SimGraph 0.438 · doc 0.650 | **HippoRAG 0.975** | keep the heavy engine |

Only HippoRAG's real entity graph chains facts across passages (reproduced at 0.975 across both
the Python and Go harnesses; Go recall@6 0.900 > doc 0.675). For temporal, the lossy LLM
fact-extraction (Graphiti) *loses* to a dependency-free 2-hop retriever.

**Lesson 2 — context quality beats quantity:** on MuSiQue, routed context **ties full-dump answer
accuracy at ~⅓ the tokens** (Δ CR−full-dump = +0.025, 95% CI [−0.050, +0.092]) while **decisively
beating naive** (Δ +0.275, CI [+0.192, +0.358]); Nemotron-Super even wins outright (CR 0.80 >
full-dump 0.70). Model-size-invariant across 24B–284B. Go replication: MuSiQue cr 0.39·565t beats
full-dump 0.30·2442t at 23% of the tokens.

**Lesson 3 — orchestration still matters at scale:** under heavy pollution, `DeepSeek-V4-Flash`
**abstains 42–50%** of the time (returns "NOT FOUND" despite the answer being present); under
Context Runtime's compressed, document-scoped context, abstention drops to **0%**. The proprietary
nutrition corpus reproduces the rescue (89% vs 48% answered, 5× faster).

**Honest negative — the temporal weak spot:** routing is perfect and SimGraph recovers 1.000 recall,
yet full-dump still wins temporal *answer* quality on the small oracle haystack — a context-*quantity*
effect (all ~3 evidence sessions fit), not a retrieval one. The answer-win awaits the large
distractor haystack. This is the same ceiling the ladder's temporal row sits at.

## v5 — reasoning-intensive retrieval (DIVER) + context sizing

**`redevops-rag/benchmarks/eval_v5_ablation.py` · `eval_v5_bandit.py` · `eval_v5_context_size.py`**
— measured on **TEMPO** (temporal reasoning-intensive retrieval, 64.7k docs) and LongMemEval.

**DIVER retrieval** (LLM query-expansion → hybrid retrieve → listwise rerank), TEMPO/workplace vs
gold documents:

| Metric | BM25 (SIM-RAG) | DIVER | Δ |
|---|---|---|---|
| NDCG@10 | 0.197 | **0.448** | ▲ +0.251 |
| Recall@10 | 0.245 | **0.580** | ▲ +0.335 |
| MRR | 0.276 | **0.490** | ▲ +0.214 |

**Ablation — DIVER is the lever, not the embedder** (NDCG@10, matched config): DIVER·bge **0.448**
beats DIVER·ReasonIR-8B **0.337**; a stronger 8B reasoning embedder lifts *plain hybrid* (0.297→0.325)
but *degrades* DIVER (−0.11). So DIVER ships embedder-agnostic over a cheap encoder; ReasonIR and the
combined retriever are documented opt-ins, not defaults.

**Learned online:** streamed through the retriever bandit with an NDCG reward, per-query selection
climbs **+0.148** (0.26 → 0.41) and beats every fixed arm.

**Context size has a sweet spot** (LongMemEval, judge-graded answer accuracy vs tokens of context
passed — the CR-sizer tunable): Qwen rises **0.167 → 0.333** as context grows ~3.5k → ~10.5k, but
DeepSeek-Flash peaks near ~4.5k (0.250) then **dips** at ~10.5k (0.167) — the lost-in-the-middle
tail. The sizer should target a *band*, not dump everything. _(Small n: Qwen n=18, DeepSeek n=12;
DeepSeek is a Q4 CPU build, so this axis shows the **shape**, not a model ranking.)_

---

# Standalone studies

## Retrieval over heterogeneous personal data (financial × medical)

**`examples/heterogeneous_shards.py`** — real **FinanceBench 10-K pages** + a medical corpus (live
demo uses public **PubMedQA**, `deploy/medical/`) deliberately built to collide on vocabulary
(*discharge*, *statement*, *balance*, *chronic*/*acute*, *liability*). 8 medical probes over 3000
financial pages + 16 clinical notes:

| Strategy | Medical recall | Cross-domain noise (top-5) |
|---|---|---|
| flat mixed index | 8/8 | 2.5 finance docs / query |
| sharded + RRF fuse | 8/8 | 2.9 *(fusion makes it worse)* |
| sharded + **coverage-routed** | 8/8 | **0.0** |

It is **not** a burial problem — BM25 keeps the short clinical note at rank #1 (recall 8/8
everywhere). The failure is **context pollution**, and naive RRF fusion injects *more* cross-domain
noise. Coverage routing (score each shard by its best hit's query-term coverage, fuse only the
shard(s) the query belongs to) cuts noise **22 → 0 docs bidirectionally**, recall intact. Same result
on InMemory / DuckDB `fts` / Postgres `tsvector` backends. The `route_margin` threshold is exactly
the kind of policy the chat-memory bandit learns per bucket.

## 3-index chat memory — learning which index to read

**`examples/chat_memory.py`** — recency / semantic / entity recall modes as one `RetrieverPlugin`; a
per-bucket ε-greedy bandit learns the single decisive index per query bucket from `value − read-cost`.

| | Context Runtime (learned) | baseline (read all three) | lift |
|---|---|---|---|
| reward | **1.23** | −1.70 | **+2.93** |

Learned policy: `followup → recency`, `factual → semantic`, `entity → entity`.

## Parallel sharded fusion

**`examples/parallel_fusion.py`** — fan a query to N shards concurrently, RRF-fuse the union (Polars
when installed, else a byte-identical pure-Python fallback). 6 shards: **5.8× faster** parallel
fan-out vs sequential (wall-clock = slowest shard, not the sum); fused top-K **identical**.

## Tenants & governance (Gen-5)

**`examples/hop_routing.py`** — intent-routed: conceptual → hybrid (acc 0.78, $0.06); multi-hop →
graph (acc 0.88, $0.43), surfacing a bridge doc via the ATP→α-synuclein hop.
**`examples/trust_aware.py`** — trust folded into plan selection (trust_weight 0.6 → the relied-upon
local plan wins); abstention gate: expected accuracy 0.80 clears the 0.7 threshold → SERVE.
See `examples/fleet_tenants.py` and the per-app reference-application tenants for the live paths.

---

## Methodology & reproducibility

- **Harnesses:** ladder — `redevops-rag/benchmarks/eval_ladder.py`; v5 — `eval_v5_*.py` (same repo);
  v1→v2/v3/shards/chat-memory/fusion — `examples/*.py` here (`PYTHONPATH=. python examples/<name>.py`);
  v4 routing/graph — `benchmarks/context-vs-model/harness/` + `graph_compare.py`.
- **Models:** answerer(s) Qwen3.6-35B-A3B (GPU/NVFP4) and, for the large-model study, Qwen3.6-Coder-Next
  80B / Nemotron-3-Super 120B / DeepSeek-V4-Flash 284B (CPU/GGUF Q4, bartowski clean quant); router +
  graph/temporal extraction Qwen3.6-27B (NVFP4); embeddings `BAAI/bge-large-en-v1.5` (bge-small for the
  ladder/v5 stores); judge grok-4.5.
- **Retrieval engines:** real HippoRAG (LLM-OpenIE + PPR) and Graphiti (Neo4j bi-temporal) behind the
  `HopRouter` seam for the v4 study; SimGraph is dependency-free; DIVER over redevops-rag hybrid.
- **Settings:** λ=0.15 (cost graph 0.4 / temporal 0.2 / document 0.0), bandit ε=0.15, seed=13.

### Datasets & regimes

| Dataset | Availability | Regime it stresses |
|---|---|---|
| PopQA | public | single-fact lookup |
| MuSiQue | public | multi-hop reasoning (graph) |
| LongMemEval | public | temporal / conversational memory |
| TEMPO | public | temporal reasoning-intensive retrieval (64.7k docs) |
| LiveRAG · FinanceBench · PubMedQA | public | pollution / numeric QA / medical shard |
| Nutrition-consultation corpus | **proprietary** | anti-parametric control (never in training) |

### Caveats

- Ladder n=15/regime (45 q) with a single 35B answerer + grok judge — directional, not a leaderboard;
  lookup saturates, temporal sits at a small-model aggregation ceiling.
- PopQA/MuSiQue are public and likely in training data → answer-quality numbers carry a contamination
  caveat; routing/recall headlines are answer-model-independent, and the nutrition corpus is the
  anti-parametric control.
- v4 temporal answer-quality is on the easy oracle haystack; the large-haystack win is future work.
- The v5 context-size axis is small-n and its DeepSeek arm is Q4-CPU (shows shape, not a model ranking).

### Deep lab reports (appendices, not headline duplicates)

- [`benchmarks/context-vs-model/RESULTS.md`](./benchmarks/context-vs-model/RESULTS.md) — the full
  *Better Context Beats More Context* study (3 lessons, CIs, per-model tables, nutrition rescue).
- [`benchmarks/context-vs-model/reports/`](./benchmarks/context-vs-model/reports) — routing-study,
  large-model-routing, nutri-context-vs-model (learning curves, full CIs).
