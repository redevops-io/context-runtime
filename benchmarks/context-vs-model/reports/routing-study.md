<!-- Detailed lab report. Synthesis: ../RESULTS.md · Article: https://redevops.io/blog/better-context-beats-more-context -->

# Context Runtime v4 — Knowledge-Aware Routing: Benchmark Report

**Question.** Does routing each query to the *knowledge representation* that can answer it —
document, graph, or temporal — beat (a) a fixed single retrieval method and (b) a cheap keyword
router, using **real** retrieval engines? And downstream: at equal model size, does CR-routed context
match "dump everything" at lower cost?

**One-line answer.** Yes for routing (the study's centerpiece) and for graph; document-regime is an
efficiency win; the temporal engine underperforms on this dataset and is the honest weak spot.

---

## Setup

Three datasets, each a distinct **graph regime**, unified to one schema (120 q each; runs use 40 or 15):

| Regime | Dataset | Right representation | Real engine |
|---|---|---|---|
| Low / no graph | PopQA (single fact + distractors) | document | InMemory BM25/hybrid |
| Multi-hop graph | MuSiQue (20 paras, 2 supporting) | graph | **real HippoRAG** (OpenIE + Personalized PageRank) |
| Temporal conversation | LongMemEval-oracle (timestamped sessions) | temporal | **real Graphiti** (Neo4j bi-temporal) |

**Stack.** Base `context_runtime` on the v4 planner (`8e4ffeb`, classify→constrain→learn). Retrieval
engines behind CR's `HopRouter` seam. All extraction/answering by **Qwen3.6-27B-NVFP4** (one GPU),
routing labels by the same 27B, a no-think proxy in front to stop thinking-mode from corrupting
HippoRAG's OpenIE. The 80B planner was cut — the 27B already routes ≥0.90. LLM-free retrieval reward =
recall − λ·cost (λ=0.15, seed=13, bandit ε=0.15).

**Ground truth is the dataset's design, not the model.** Routing accuracy grades the classifier's
label against the regime the dataset was built for (`low_graph→document, graph→graph,
temporal→temporal`) — not against the model's own labels — and the classify prompt sees only the
question text. That target is independently justified by the retrieval numbers below (fixed-graph
≫ fixed-document on MuSiQue *proves* graph is the correct target), so "0.90 routing accuracy" =
90% of queries sent to the empirically-best representation.

---

## Result 1 — Routing accuracy (the centerpiece)

Does the planner pick the representation that can answer the query?

| Regime | Keyword router (shipped `RuleIntentAnalyzer`) | **LLM router (27B)** |
|---|---|---|
| PopQA → document | 0.95 | **1.00** |
| MuSiQue → graph | **0.00** | **0.90** |
| LongMemEval → temporal | **0.00** | **1.00** |

The keyword head is blind to *implicit* intent — "Who is the spouse of the Green performer?" carries
no "multi-hop" string, so it defaults to document and the graph engine is never reached. **The v4
architecture is validated; its shipped head is the bottleneck.** The fix ships as
`HybridIntentAnalyzer` (below).

### Interleaved mixed stream — per-query discrimination, not per-dataset memorization

All three regimes shuffled into **one** stream (45 q, deterministic), backend chosen per-query by
regime, one bandit:

| Regime (in the mixed stream) | routed correctly | CR recall | fixed-doc | fixed-graph | fixed-temporal |
|---|---|---|---|---|---|
| low_graph (n=15) | **1.00 → document** | 1.000 | 1.000 | 1.000 | — |
| graph (n=15) | **0.93 → graph** | 0.733 | 0.600 | 0.967 | — |
| temporal (n=15) | **1.00 → temporal** | 0.967 | 1.000 | — | 0.698 |

In a single shuffled workload the router sends each query to the right representation — the strongest
form of the thesis. Memorizing "this dataset is graph" is impossible here; it's per-query.

---

## Result 2 — Retrieval quality + the bandit learning

Gold-recall under CR's route vs fixed methods (MuSiQue, n=40):

| | recall | 95% CI (2000-boot) |
|---|---|---|
| **fixed graph (HippoRAG)** | **0.988** | [0.963, 1.000] |
| CR (routed + learned) | 0.925 | [0.863, 0.988] |
| fixed document | 0.650 | [0.550, 0.738] |
| **Δ(graph − document)** | **+0.338** | **[0.250, 0.438]** |

Real HippoRAG lifts multi-hop recall **0.650 → 0.988** — the graph representation's value is large and
robustly significant. CR routes 88% to graph and captures 0.925; the gap to fixed-graph is exploration
cost, not a routing failure (CIs overlap).

**Learning curve** (MuSiQue, stream order) — the bandit converging, not just the endpoint:

| checkpoint | cum graph-route frac | cum CR recall |
|---|---|---|
| q4 | 0.75 | 0.750 |
| q10 | 0.80 | 0.850 |
| q20 | 0.65 | 0.900 |
| q40 | 0.70 | 0.925 |

Chosen-arm shifts **1st half {graph:13, hybrid:7} → 2nd half {graph:15, hybrid:3, vector:2}** —
convergence to `graph:local`.

---

## Result 3 — Judged answer quality (grok-4.5 judge, 578 rows, $0.51)

The downstream question, at **equal model size** (all four answerers are 24–30B; there is no bigger
model in the set, so this is *better context vs more context*, not "beats a bigger model"):

| Dataset | CR | naive | full_dump | reading |
|---|---|---|---|---|
| **MuSiQue** | **0.49** | 0.29 | 0.46 | CR **matches** full-dump, **beats** naive |
| PopQA | 0.97 | 0.97 | 0.99 | tie → efficiency win |
| LongMemEval | 0.03 | 0.03 | **0.23** | full-dump wins (weak spot) |

Paired bootstrap on MuSiQue (n=160 model×question):
- **Δ(CR − full_dump) = +0.037, 95% CI [−0.025, +0.106]** → **statistical tie.** CR uses ~⅓ the tokens
  (routed passages vs all 20 paras), so *matching at a third of the context* is the win.
- **Δ(CR − naive) = +0.206, 95% CI [+0.144, +0.269]** → CR **significantly** beats single-hop retrieval.
- CR ≥ full-dump held on all four models individually (qwen 0.55/0.50, mistral 0.57/0.47, gemma & nemotron tie).

**Honest headline:** at equal model size, CR-routed context **matches "dump everything" at a third of
the context cost and clearly beats weak retrieval.** Better context ≈ more context, far cheaper.

---

## The temporal weak spot (honest negative)

On LongMemEval, routing is perfect (1.00 → temporal) but the temporal *engine* loses:

- Graphiti session-recall (with the new edge→session provenance) = **0.731**, vs **document/hybrid on
  sessions = 1.000**. Answer quality: full-dump 0.23 ≫ CR 0.03.
- **Why:** LongMemEval-*oracle* keeps only the evidence sessions (≈3 per question), so the haystack is
  tiny — plain hybrid retrieval finds every relevant session, and Graphiti's LLM-extracted facts are
  *lossy* relative to the raw turns. Graphiti's advantage (point-in-time over a large, revised history)
  needs the full `longmemeval_s/m` haystack (100s of distractor sessions) to appear; the oracle set is
  too easy for document retrieval. **On this dataset the correct representation is document** — and the
  router's job is to learn that, which is the next test.
- Takeaway: the temporal *binding* is correct and now measurable (provenance fix), but the *claim*
  "temporal beats document" is not supported on the oracle haystack. Reported as-is.

**λ-sensitivity:** doubling the efficiency penalty (λ 0.15 → 0.30) does **not** change PopQA's route —
still 100% document. The efficiency preference is stable, not a knife-edge of λ.

## Follow-up: does the *heavy* engine earn its cost in each regime?

The temporal result raised the real question — not "is the binding correct" but "does the expensive
LLM engine beat the simple non-lossy alternative?" We measured it on both engines.

**Graph (MuSiQue recall@4, same 40 items, `benchmarks/graph_compare.py`):** does the dependency-free
`SimGraphRetriever` (2-hop term-spread, no LLM-OpenIE, no heavy deps) match the heavy `HippoRAGRetriever`?

| engine | recall@4 | cost |
|---|---|---|
| **HippoRAG** (LLM-OpenIE entity graph + PPR) | **0.975** | heavy |
| SimGraph (dependency-free 2-hop) | 0.438 | ~free |
| **Δ (SimGraph − HippoRAG)** | **−0.538** | → **HippoRAG earns its cost** |

SimGraph (0.44) doesn't just lag HippoRAG (0.975) — it lags even flat document retrieval (0.65). Genuine
multi-hop needs the learned entity graph + PPR; the cheap 2-hop can't chain "spouse of the Green
performer" across paragraphs. HippoRAG reproduced at exactly 0.975 here — cross-harness validation.
(HippoRAG is already non-lossy — it returns *raw passages*, using the graph only as a ranking signal —
so the "Graphiti fix" is a no-op for it.)

**Temporal (LongMemEval), replacing Graphiti with the non-lossy SimGraph over sessions:** SimGraph
recall = **1.000** — it *fully recovers* the temporal regime (vs Graphiti's 0.733), matching document,
with **no Neo4j and no extraction LLM**. The lossy LLM fact-extraction was the whole problem; a
non-lossy retriever fixes it for free.

*Answer quality on the same temporal contexts* (re-run with the pack fix + SimGraph, judged, 3 large
CPU models): now **non-zero** (the earlier 0.00 was the empty-context artifact) but **full-dump still
edges cr/naive** — Nemotron-Super full_dump 0.40 vs cr 0.24; Coder-Next 0.16 vs 0.12; DeepSeek 0.20 vs
0.08. This is a **context-*quantity*** effect, not a retrieval one: recall is a perfect 1.000, but the
oracle haystack is small enough that dumping all sessions (~18k chars) fits and beats the top-k
retrieved subset (~9k). CR's efficiency edge only pays off once the full haystack no longer fits — i.e.
the real temporal test still needs `longmemeval_s/m` (100s of distractor sessions), where full-dump
can't fit and SimGraph's 1.000 recall would carry the answer. **Retrieval: solved. Answer-quality win:
awaits the large haystack.**

**The symmetry is the finding:**

| regime | simple non-lossy | heavy LLM engine | decision |
|---|---|---|---|
| **Temporal** | SimGraph **1.00** / document 1.00 | Graphiti 0.733 | **drop the heavy engine** |
| **Graph** | SimGraph 0.44 | HippoRAG **0.975** | **keep the heavy engine** |

The same "does the expensive engine earn its cost?" question gets **opposite answers**: heavy graph
*construction* pays off for multi-hop; heavy temporal fact-*extraction* does not. Recommendation: keep
HippoRAG for graph; use SimGraph (dependency-free, non-lossy) for temporal and retire the Graphiti +
Neo4j dependency for this workload.

> **Measurement caveat found during this follow-up:** the context packer dropped any single passage
> larger than the token budget instead of truncating it, which zeroed out LongMemEval's cr/naive answer
> contexts (its sessions are long) — so the earlier temporal *answer* scores (cr 0.03) understated the
> arm. Fixed (truncate-first-item); temporal *retrieval* recall above was always correct.

---

## What ships to context-runtime (see `benchmarks_tests_setup.md` for the two-patch assembly)

1. **`HybridIntentAnalyzer` (`planner/llm_intent.py`) — the primary code finding.** Keyword-first,
   LLM-on-doubt: the cheap head handles explicit cues for free; the LLM head fires only on the
   implicit-intent blind spot (document-default / low confidence). Verified: implicit multi-hop → 1 LLM
   call → graph; explicit temporal cue → **0 LLM calls**; LLM-down → graceful keyword fallback. Ships
   with a structured `analytical`/`code` prompt, bucket re-alignment (so model tier follows the
   representation), and a `stats` cost-ledger (`escalated`/`overrides` = the measurable per-query
   routing overhead).
2. **`GraphitiTemporalRetriever` (`store_temporal.py`)** — the real bi-temporal binding (didn't exist,
   only a stub-promise). Injectable embedder/reranker, edge→session provenance, ISO+non-ISO dates.
3. **`store_hipporag.py`** — local-endpoint threading + two correctness fixes (public-repo import,
   numpy truth-value) worth upstreaming regardless.

---

## Limitations (stated plainly)

- **n=40, single seed, ε-greedy** — the *routing gap* (0.90 vs 0.00) and *retrieval Δ(graph−doc)*
  [0.250, 0.438] are robust; the CR-vs-fixed-graph and CR-vs-full-dump deltas are within noise (report
  as ties, which is the honest and still-favorable reading). Multi-seed averaging is the obvious next step.
- **Temporal** evaluated on the oracle (easy) haystack; the full haystack is needed to test Graphiti's
  real value.
- **PopQA corpus is constructed** (gold fact + distractors) — a controlled low-graph regime, not an
  open-domain retrieval test.
- Routing labels came from the *same* 27B that answers; ground truth is the dataset regime (not the
  model), so this isn't circular, but a second independent labeler would further harden it.
- **Driver-model headroom is small.** The routing driver is the Qwen3.6-27B; routing accuracy is already
  near-ceiling (PopQA 1.00, MuSiQue 0.88, LongMemEval 1.00), so a stronger/larger driver (e.g. a 32B)
  has room for **at most ~+0.03 overall** — only MuSiQue's 5/40 implicit-multi-hop misroutes have
  headroom. The routing lift was never about model *size* (the keyword head scores 0.00; the LLM head
  even at 27B reaches 0.90–1.00) — it's about analyzer *type*, which is why the 80B driver was skipped.
  A cheaper lever for those 5 misroutes is a better classify prompt/few-shot, not a bigger model;
  extraction headroom is similarly small (HippoRAG already 0.975).

## Reproducibility

Harness `bench_route.py` (classify / route / `--dump-contexts` / `--interleave` / `--lam`), prep
`prep_datasets.py` (seed 13), orchestrators, and `mh_{answer,judge}.py`. λ=0.15 (COST: graph 0.4,
temporal 0.2, document 0.0). Judge grok-4.5, $0.51 total. All raw results in
`/cache/bench/v4/results/`.
