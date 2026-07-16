# Results — Better Context Beats More Context

The benchmark results behind the article
[**"Better Context Beats More Context"**](https://redevops.io/blog/better-context-beats-more-context).

Three lessons from stress-testing retrieval architectures against **24B–284B** models — where heavy
infrastructure earned its cost, where it didn't, and why context orchestration and model progress are
complementary, not competing.

Every number here is reproducible from this harness (see [Reproducibility](#reproducibility)); the two
detailed lab reports live in [`reports/`](./reports).

---

## What we tested, and why the datasets matter

Every headline comes from a run on the **same models**, served locally so the comparison is
apples-to-apples. Retrieval and orchestration were driven by **Context Runtime**; a smaller model
handled routing and graph/temporal extraction. Contamination is the quiet threat in any RAG benchmark:
if a model memorized the test set, "does context help" collapses. So we mix **public** datasets
(transparent, reproducible) with a **proprietary** corpus the models could not have seen in training —
the anti-parametric control.

| Role | Model | Notes |
|---|---|---|
| Answerer (large tier, CPU/GGUF Q4) | `Qwen3.6-Coder-Next` (80B-A3B) | MoE, sparse active params |
| Answerer (large tier, CPU/GGUF Q4) | `NVIDIA Nemotron-3-Super` (120B-A12B) | strong long-context handling |
| Answerer (large tier, CPU/GGUF Q4) | `DeepSeek-V4-Flash` (284B-A13B) | heavy in-model context management |
| Answerer (small tier, GPU/NVFP4) | 24–30B set (Qwen / Mistral / Gemma / Nemotron) | the size-invariance control |
| Router · extractor | `Qwen3.6-27B` (NVFP4) | intent routing + graph/temporal extraction |
| Embeddings | `BAAI/bge-large-en-v1.5` | dense retrieval |
| Judge | `grok-4.5` | answer grading |

| Dataset | Availability | Regime it stresses |
|---|---|---|
| PopQA | public | single-fact lookup (low graph) |
| MuSiQue | public | multi-hop reasoning (graph) |
| LongMemEval | public | temporal / conversational memory |
| LiveRAG | public | mixed-domain, adversarial pollution |
| FinanceBench | public | numeric 10-K QA |
| Nutrition-consultation corpus | **proprietary** | anti-parametric — never in training |

### At a glance

| | Finding | Evidence |
|---|---|---|
| **Routing** (foundation) | LLM routing sends each query to the right representation; keyword routing is blind to implicit intent | 1.00 / 0.90 / 1.00 vs 0.95 / **0.00** / **0.00** |
| **Lesson 1** | Heavy infrastructure has to *earn* its cost — same question, opposite answers by regime | temporal: SimGraph 1.00 > Graphiti 0.73 · graph: HippoRAG 0.98 > SimGraph 0.44 |
| **Lesson 2** | Context quality beats context quantity — same answer at ⅓ the context | MuSiQue: CR ≈ full-dump (tie) at ~⅓ tokens; CR ≫ naive (+0.28) |
| **Lesson 3** | Runtime orchestration still matters once context is large or polluted | DeepSeek abstention 42–50% → **0%** under Context Runtime |

---

## Routing — the foundation (contamination-immune)

Routing accuracy grades the classifier's label against the regime each dataset was built for, seeing
only the question text — so it is independent of the answer model and immune to the memorization
concern. Does the planner pick the representation that can answer the query?

| Regime | Keyword router (shipped `RuleIntentAnalyzer`) | **LLM router (Qwen3.6-27B)** |
|---|---|---|
| PopQA → document | 0.95 | **1.00** |
| MuSiQue → graph | **0.00** | **0.90** (0.88 large-model run) |
| LongMemEval → temporal | **0.00** | **1.00** |

The keyword head is blind to *implicit* intent — "Who is the spouse of the Green performer?" carries no
"multi-hop" string, so it defaults to document and the graph engine is never reached. Shuffle all three
regimes into one stream and the LLM router still discriminates per-query (document 1.00 / graph 0.93 /
temporal 1.00), which rules out per-dataset memorization. **The routing lift is a property of the
analyzer *type*, not model size** — the keyword head scores 0.00 where it counts; the LLM head reaches
0.90–1.00 even at 27B.

---

## Lesson 1 — Better architecture beats heavier infrastructure

One of our early assumptions was wrong. We expected Graphiti's LLM-generated temporal graph to
outperform simpler approaches. **It didn't.** We paired each regime with its proper retriever — a real
**HippoRAG** graph for multi-hop, our dependency-free **SimGraph** for temporal memory — then
cross-applied the wrong one as a control. The results ran in opposite directions.

**Temporal recall — LongMemEval** (session recall):

| Retriever | recall | |
|---|---|---|
| Graphiti (heavy LLM fact-extraction) | 0.733 | heavy engine **loses** |
| Document (hybrid) | 1.000 | |
| SimGraph (dependency-free 2-hop) | **1.000** | non-lossy retriever recovers it, no Neo4j / no extraction LLM |

**Multi-hop recall — MuSiQue** (recall@4, same 40 items, `harness`/`graph_compare.py`):

| Retriever | recall@4 | |
|---|---|---|
| SimGraph (dependency-free 2-hop) | 0.438 | heavy engine **wins** |
| Document (hybrid) | 0.650 | |
| HippoRAG (LLM-OpenIE entity graph + PPR) | **0.975** | only the real graph chains facts across passages |

Graphiti's LLM-extracted facts are *lossy* and *lost* to SimGraph. On multi-hop, the cheap 2-hop
retriever collapsed below even flat document retrieval — only HippoRAG's real entity graph could chain
"spouse of the Green performer" across paragraphs. HippoRAG reproduced at exactly 0.975 across both
harnesses (cross-harness validation).

**The symmetry is the finding:**

| Regime | Simple (non-lossy) | Heavy (LLM engine) | Decision |
|---|---|---|---|
| Temporal | **SimGraph 1.00** | Graphiti 0.73 | drop the heavy engine |
| Multi-hop graph | SimGraph 0.44 | **HippoRAG 0.98** | keep the heavy engine |

The same "does the expensive engine earn its cost?" question gets **opposite answers** by regime: heavy
graph *construction* pays off for multi-hop; heavy temporal fact-*extraction* does not (lossy facts vs.
raw turns).

> **The lesson wasn't "graphs are good." It was that heavy infrastructure should earn its cost —
> sometimes it does, sometimes it doesn't.**

---

## Lesson 2 — Context quality beats context quantity

On clean data, all three large models answered about equally well whether they got the full context or
Context Runtime's routed context — each with the retriever proper to the task. The detail that matters:
**Context Runtime reached the same answer quality using roughly one-third of the context tokens.**

**MuSiQue — judged answer accuracy (large models):**

| Model | Context Runtime | naive retrieval | full-dump |
|---|---|---|---|
| Coder-Next (80B) | **0.57** | 0.40 | 0.57 |
| Nemotron-Super (120B) | **0.80** | 0.38 | 0.70 |
| DeepSeek-Flash (284B) | 0.53 | 0.30 | 0.55 |

Paired bootstrap, pooled models (n=120):
- **Δ(CR − full-dump) = +0.025, 95% CI [−0.050, +0.092] → statistical tie**, at **~⅓ the tokens**.
- **Δ(CR − naive) = +0.275, 95% CI [+0.192, +0.358] → CR decisively wins.**
- Notably **Nemotron-Super CR 0.80 > full-dump 0.70** — a clean CR win on the biggest-context-capable model.

**The effect is model-size-invariant.** The same result holds on the small (24–30B) tier:

| MuSiQue | small models (24–30B, GPU/NVFP4) | large models (80–284B, CPU/GGUF) |
|---|---|---|
| CR vs full-dump | +0.037, tie | +0.025, tie |
| CR vs naive | +0.206, wins | +0.275, wins |
| PopQA | tie (efficiency) | tie (efficiency) |
| LongMemEval | full-dump wins | full-dump wins |

This isn't about beating frontier models. It's about reaching the **same answer using one-third of the
context** — lower latency, lower cost, more room for reasoning, and a larger safety margin before the
window saturates. The effect held across an order of magnitude of model size. On PopQA (memorizable
single facts) every arm ties, so routing to the cheapest representation is a pure efficiency win.

> **Context quality beats context quantity. Better context beats bigger context.**

---

## Lesson 3 — Runtime architecture still matters at scale

Frontier models increasingly perform sophisticated internal context management. Our expectation was that
this would *reduce* the value of external context orchestration. Instead, runtime orchestration still
mattered once the context became sufficiently large or polluted.

The evidence: `DeepSeek-V4-Flash` is exactly this kind of model. Yet under heavily polluted retrieval it
began **abstaining — returning "NOT FOUND" despite the answer being present.** This signal is
judge-independent (the model literally declines to answer).

**LiveRAG — DeepSeek-V4-Flash abstention rate ("NOT FOUND"), native `document_ids` scoping:**

| Arm | pollution 0 | pollution 100 |
|---|---|---|
| full-dump (large native ctx) | **42%** (5/12) | 17% (2/12) |
| naive retrieval (polluted) | 17% (2/12) | **50%** (6/12) |
| **Context Runtime** | **0%** (0/12) | **0%** (0/12) |
| Coder-Next / Nemotron-Super (all arms) | 0% | 0% |

Context Runtime eliminated the drowning entirely — the bandit-picked, compressed, document-scoped
context is always digestible; pollution doesn't touch it. The **proprietary nutrition corpus reproduces
the rescue** off the public data: with a ~17–25k-token context DeepSeek returns "NOT FOUND" on 42/80
questions despite 99% retrieval hit; on CR's compressed context it answers **89% vs 48%**, flat under
pollution and 5× faster.

The other two models never abstained. **Models with stronger long-context capabilities see smaller
quality gains on clean data — but Context Runtime still cuts context usage substantially, and keeps them
robust once the workload becomes large or polluted.** Context Runtime isn't competing with modern models
— it's complementing them.

> **As models get better at reasoning, the competitive advantage shifts from model capability to
> context orchestration.**

---

## The temporal weak spot (honest negative)

On LongMemEval, routing is perfect (1.00 → temporal) and SimGraph recovers perfect recall (1.000), but
**full-dump still wins on answer quality** (large models 0.20–0.47 vs CR ~0.00–0.07):

- This is a context-*quantity* effect, **not** a retrieval one — recall is a perfect 1.000. The
  LongMemEval-*oracle* haystack is tiny (≈3 evidence sessions/question), so dumping all sessions
  (~18k chars) fits and beats the top-k retrieved subset (~9k). CR's efficiency edge only pays off once
  the full haystack no longer fits.
- The real temporal answer-win needs the full `longmemeval_s/m` haystack (100s of distractor sessions),
  where full-dump can't fit and SimGraph's 1.000 recall would carry the answer. **Retrieval: solved.
  Answer-quality win: awaits the large haystack.**

A measurement bug found here and fixed — the context packer *dropped* any single passage larger than the
token budget instead of truncating it, zeroing LongMemEval's long-session contexts — was the source of
an earlier artifactual 0.00 (fixed in [`a881e89`](https://github.com/redevops-io/context-runtime),
truncate-first-item). Temporal *retrieval* recall was always correct.

**Cross-study nuance (nutrition corpus):** on that clean, retrievable corpus **Nemotron-Super *preferred*
full context** — CR's compression removed signal it didn't need and lowered accuracy (90→76% under
pollution). Whether context management helps depends on the task and the model's long-context ability,
not on pollution level alone. See [`reports/nutri-context-vs-model.md`](./reports/nutri-context-vs-model.md).

---

## The through-line

Model progress and context orchestration are **complementary technologies, not competing ones.** Better
models *increase* the value of better context — they don't eliminate it. Three findings, one shape:
**heavy infrastructure should earn its cost**, context quality beats context quantity, and a runtime
layer routes signal to the model rather than dumping noise at it. As models improve, context
orchestration doesn't become less important — it becomes more efficient.

---

## Reproducibility

- **Harness:** this directory (`benchmarks/context-vs-model/`) — `harness/run.py`, `harness/report.py`,
  `harness/tuner.py` (the real `EpsilonGreedyBandit` over `redevops_rag.RetrievalConfig`), and
  `graph_compare.py` (SimGraph vs HippoRAG recall@k). See [`README.md`](./README.md) for the run recipe.
- **Serving:** large models via `bartowski` GGUF Q4 on `llama.cpp` (CPU, 96 threads, 377 GB RAM);
  small tier + the Qwen3.6-27B router/extractor on GPU (NVFP4).
- **Retrieval engines:** real HippoRAG (LLM-OpenIE + PPR) and Graphiti (Neo4j bi-temporal) behind
  Context Runtime's `HopRouter` seam; SimGraph is dependency-free. Pins: `graphiti-core` and `HippoRAG`
  fixed to SHAs, `redevops-rag` at the `document_ids` native-scoping commit.
- **Settings:** λ=0.15 (cost: graph 0.4 / temporal 0.2 / document 0.0), seed=13, bandit ε=0.15.
- **Judge:** grok-4.5 (routing/large-model runs $0.51 + $0.27; abstention results are judge-free).

### Detailed lab reports

- [`reports/routing-study.md`](./reports/routing-study.md) — the 3-regime knowledge-aware routing study
  (small-model tier + SimGraph/HippoRAG + temporal follow-up), full CIs and learning curves.
- [`reports/large-model-routing.md`](./reports/large-model-routing.md) — large-model (80–284B) replication,
  size-invariance, and the LiveRAG polluted-context run.
- [`reports/nutri-context-vs-model.md`](./reports/nutri-context-vs-model.md) — the proprietary nutrition
  corpus "can context management beat a better model?" result (DeepSeek rescue; Nemotron prefers full context).

### Known caveats

- n=40 (small) / n=12–120 (large) per cell — the routing gap (0.90 vs 0.00) and the retrieval Δ(graph −
  document) [0.250, 0.438] are robust; the CR-vs-full-dump deltas are within noise (reported as ties).
- PopQA/MuSiQue are public and likely in training data → answer quality carries a contamination caveat;
  the routing/retrieval headline is immune, and the proprietary nutrition corpus is the anti-parametric control.
- Temporal evaluated on the easy oracle haystack; the large haystack is needed to test Graphiti's real value.
- LiveRAG prose accuracy awaits a `--judge` re-run; the abstention result above is judge-free.
