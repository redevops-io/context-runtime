<!-- Detailed lab report. Synthesis: ../RESULTS.md · Article: https://redevops.io/blog/better-context-beats-more-context -->

# CR v4 Knowledge-Aware Routing — Large-Model Replication (CPU / GGUF)

**What this run is.** The 3-regime routing study (PopQA / MuSiQue / LongMemEval), re-run with the
**large models** as answerers — **Qwen3.6-Coder-Next (80B-A3B), Nemotron-3-Super (120B-A12B),
DeepSeek-V4-Flash (284B-A13B)** served on **CPU/llama.cpp (GGUF Q4)** — through the **verified**
`context-runtime-bench` stack (vendored `context_runtime` with G1 source-hydration, `HybridIntentAnalyzer`,
structured routing; graphiti@main, HippoRAG, redevops-rag), replacing the earlier hand-wired adapters.
GPU-side Qwen3.6-27B did the HippoRAG/Graphiti extraction; the large models only answered. Judge:
grok-4.5 (310 rows, **$0.27**).

> **Bottom line:** the routing thesis is **model-size-invariant.** The same result we saw on 24–30B
> models holds on 80B–284B models: routing is correct, CR-routed context **matches "dump everything"
> at ~⅓ the tokens and beats naive retrieval** on multi-hop, and temporal remains the honest weak spot.

---

## Routing + retrieval (contamination-immune — the headline)

These metrics don't depend on the answer model, so they're immune to the memorization concern. Produced
by the verified `context_runtime` (identical to the small-model run, as expected):

| Regime | routing accuracy | CR recall | fixed document | fixed graph (HippoRAG) |
|---|---|---|---|---|
| PopQA (low-graph) | **1.00 → document** | 1.000 | 1.000 | 1.000 |
| MuSiQue (multi-hop) | **0.88 → graph** | 0.925 | 0.650 | 0.975 |
| LongMemEval (temporal) | **1.00 → temporal** | — | 1.000 | 0.733 (Graphiti) |

Real HippoRAG lifts MuSiQue recall **0.650 → 0.975**; the bandit learned `unknown→graph:local`. PopQA
routes to the cheapest representation (efficiency). Same as the small-model run — the routing layer is
independent of the answerer.

## Judged answer quality (large models)

| Dataset | model | CR | naive | full_dump |
|---|---|---|---|---|
| **MuSiQue** | Coder-Next | **0.57** | 0.40 | 0.57 |
| | Nemotron-Super | **0.80** | 0.38 | 0.70 |
| | DeepSeek-Flash | 0.53 | 0.30 | 0.55 |
| PopQA | (all three) | ~0.99 | ~0.98 | 1.00 |
| **LongMemEval** | Coder-Next / Nemotron / DeepSeek | 0.00 / 0.00 / 0.07 | 0.00 / 0.00 / 0.07 | **0.20 / 0.47 / 0.27** |

**MuSiQue (paired bootstrap, pooled models, n=120):**
- **Δ(CR − full_dump) = +0.025, 95% CI [−0.050, +0.092] → statistical tie.** CR-routed graph context
  matches dumping all 20 paragraphs while using **~⅓ the tokens.**
- **Δ(CR − naive) = +0.275, 95% CI [+0.192, +0.358] → CR significantly wins.**
- Notably **Nemotron-Super CR 0.80 > full_dump 0.70** — a clean CR win on the biggest-context-capable model.

**PopQA:** ties across the board → efficiency win (and contamination-transparent: these are memorizable
facts and every model aces all arms — exactly the caveat below).

**LongMemEval:** full_dump wins (0.20–0.47 vs ~0.00). Temporal stays the honest negative.

## Small vs large — the pattern is invariant

| | small models (24–30B, GPU/NVFP4) | large models (80–284B, CPU/GGUF) |
|---|---|---|
| MuSiQue: CR vs full_dump | +0.037, tie | +0.025, tie |
| MuSiQue: CR vs naive | +0.206, wins | +0.275, wins |
| PopQA | tie (efficiency) | tie (efficiency) |
| LongMemEval | full_dump wins | full_dump wins |

The "better context ≈ more context at a fraction of the cost, and far better than weak retrieval"
result reproduces across an **order of magnitude of model size** — evidence it's a property of the
routing design, not of a particular model tier.

**Cross-study nuance:** in the earlier FinanceBench/nutri `context-vs-model` study, Nemotron-Super
*preferred* full context (compression hurt it). Here, on multi-hop MuSiQue, CR *helps* it (0.80 vs
0.70) — because routing to the graph representation is a retrieval-structure win that's orthogonal to
the model's long-context ability. Consistent with that study's own conclusion: whether context
management helps depends on the task, not just the model.

## Honest notes / caveats

- **Contamination (as agreed, proceed-as-is):** PopQA/MuSiQue are public Wikipedia-derived sets these
  large models likely saw in training, so **answer quality is reported with that caveat** — the
  routing/retrieval headline above is immune to it. The `naive`-vs-`full_dump` gap (0.30 vs 0.55 on
  MuSiQue) shows context still drove answers rather than pure memory, but the effect shrinks with size.
- **G1 source hydration:** in this deployment Graphiti's hydration returned *empty* content (episodes
  hydrated but raw content not persisted on that path), so the temporal CR context used a **raw-turn
  fallback** (the retrieved sessions) rather than G1's hydrated turns. Worth a look on the fork side —
  the G1 mechanism itself unit-passes; the gap is content persistence in this indexing path. Temporal
  answer quality would likely improve once hydration returns real turns.
- **redevops-io/HippoRAG** still pins `openai==1.91.1` (nonexistent on PyPI) — clashes with
  graphiti's `openai>=1.91.0`; patched locally to install. Fork-hygiene fix.

## SimGraph vs HippoRAG — the efficiency test (the true graph analog of the temporal decision)

Temporal showed the *simple non-lossy* approach (document) beat the *heavy LLM engine* (Graphiti). The
honest parallel for graph isn't "fix HippoRAG" (it's already non-lossy — returns raw passages, graph is
only a ranking signal) — it's: does the **dependency-free `SimGraphRetriever`** (2-hop term-spreading,
no LLM-OpenIE, no heavy install, no fork-pin) match the heavy `HippoRAGRetriever`? Measured on the same
40 MuSiQue items (`benchmarks/graph_compare.py`, recall@4):

| engine | mean recall@4 | cost |
|---|---|---|
| **HippoRAG** (LLM-OpenIE entity graph + PPR) | **0.975** | heavy (LLM per item, deps) |
| SimGraph (dependency-free 2-hop) | 0.438 | ~free |
| **Δ (SimGraph − HippoRAG)** | **−0.538** | → **HippoRAG earns its cost** |

**Verdict: keep HippoRAG.** SimGraph (0.44) doesn't just lag HippoRAG — it lags even flat document
retrieval (0.65). Genuine multi-hop needs the learned entity graph + PPR; the cheap 2-hop can't chain
"spouse of the Green performer" across paragraphs. (HippoRAG's 0.975 here reproduces the study's 0.975
exactly — cross-harness validation.)

**The symmetry is the finding:**

| regime | simple non-lossy | heavy LLM engine | decision |
|---|---|---|---|
| Temporal | document **1.00** | Graphiti 0.733 | **drop the heavy engine** |
| Graph | SimGraph 0.44 | HippoRAG **0.975** | **keep the heavy engine** |

Heavy graph *construction* pays off for multi-hop; heavy temporal *fact-extraction* does not (lossy
facts vs. raw turns). Same "does the expensive engine earn its cost" question — opposite answers.

## LiveRAG — does CR still help DeepSeek under *large polluted* context?

The prior CPU study found CR *rescues* DeepSeek-V4-Flash from drowning in a large context (89% vs 48%).
This tests whether that holds when the large context is also **polluted** — on **LiveRAG** (895
DataMorgana Q&A / 970 docs, mixed-domain), 12k context, pollution 0 vs 100, 3 arms, 12 Q/cell, on the
**corrected native `document_ids` SQL-scoping path** (redevops-rag `9f9176a` — the candidate pools are
scoped before RRF fusion, the valid pollution measurement).

**The clean, judge-independent signal — abstention ("NOT FOUND" = the model drowned):**

| model | arm | pol=0 | pol=100 |
|---|---|---|---|
| **DeepSeek-Flash** | full_dump (large native ctx) | **5/12 (42%)** | 2/12 |
| | naive_rag (polluted retrieval) | 2/12 | **6/12 (50%)** |
| | **context_runtime** | **0/12** | **0/12** |
| Coder-Next | all arms | 0/12 | 0/12 |
| Nemotron-Super | all arms | 0/12 | 0/12 |

**Answer: CR still helps DeepSeek — decisively — and its in-model context management does *not* make CR
redundant.**
- **Only DeepSeek drowns.** Its own context handling fails in *both* regimes: the large clean context
  (full_dump → 42% "NOT FOUND") *and* the polluted retrieval (naive → 50% "NOT FOUND"). It literally
  returns "NOT FOUND" despite the answer being present.
- **CR eliminates the drowning entirely — 0% abstention at every pollution level.** The bandit-picked
  compressed + document-scoped context is always digestible; pollution doesn't touch it.
- **Coder-Next and Nemotron never drown (0/12 everywhere)** → they don't need the rescue, consistent
  with the study's thesis: CR helps exactly the models that can't exploit a large (or polluted) context.

Graded accuracy is consistent (DeepSeek CR 0.50/0.42 vs full_dump 0.17/0.42, naive 0.50/0.17) but noisier
and *underestimated* — the harness grades numeric-match first and defers prose to an LLM judge that
wasn't run this pass, so prose LiveRAG answers score False. The **abstention rate above is the crisp,
judge-free result**; fair absolute accuracy needs a `--judge` re-run.

**Caveats:** n=12/cell (coarse, 0.083 granularity); trimmed to pollution {0,100} + 12 Q for CPU
tractability; prose grading pending the judge. The DeepSeek drowning / CR-rescue pattern is robust to
all of these (abstention is unambiguous).

## Reproducibility

Verified stack: `context-runtime-bench@benchmarks` (3c8c22c), `graphiti@main` (G1), `HippoRAG`,
`redevops-rag`; venv `/cache/crbench-venv`. Models: `bartowski` GGUF Q4 (Coder-Next / Nemotron-Super /
DeepSeek-Flash) on llama.cpp CPU (96 threads, 377 GB RAM). Extraction: Qwen3.6-27B-NVFP4 (GPU).
λ=0.15, seed=13, ε=0.15. Judge grok-4.5, $0.27. Raw: `/cache/bench/v4/results_large/`.
