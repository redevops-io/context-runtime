# Can better context management beat a better model?

A reproducible benchmark: at a **fixed memory budget on one CPU box**, is it better to
spend it on a **bigger model that manages context natively**, or a **smaller model +
Context Runtime** (execution planning over retrieval)? We hold the machine, the dataset,
and the question constant and move only the *context strategy* — then watch what happens
as the retrieval corpus gets polluted.

The interesting outcome isn't "Model X wins." It's whether execution planning is becoming
as important as the model.

## Results

**→ [`RESULTS.md`](./RESULTS.md)** — the headline synthesis, organized by the three lessons of the
article [**"Better Context Beats More Context"**](https://redevops.io/blog/better-context-beats-more-context):
(1) heavy infrastructure must earn its cost, (2) context quality beats context quantity, and
(3) runtime orchestration still matters once context is large or polluted.

Detailed lab reports in [`reports/`](./reports): the 3-regime routing study, the large-model
(80–284B) replication with the LiveRAG polluted-context run, and the proprietary nutrition-corpus study.

## The experiment

**Dataset — FinanceBench** (Patronus AI, public): 150 expert Q&A over 84 SEC 10-K/10-Q
filings from 32 companies. Real gold answers (mostly numeric) *and* gold evidence pages,
so both **answer quality** and **retrieval quality** are ground-truth. The multi-company
corpus is the pollution axis: a question targets one filing, but 83 others share the same
financial vocabulary with different numbers — adversarial distractors by construction.

**Three arms** (per question):

| arm | what it does | tests |
|---|---|---|
| `full_dump` | stuff the whole scoped pool (target + distractor filings) up to a token budget | the model's **native** long-context handling |
| `naive_rag` | fixed top-K BM25 (Context Runtime **OFF** — the library default) | untuned retrieval |
| `context_runtime` | intent-keyed bandit picks a retrieval config (gating threshold / limit / rerank) and prunes | Context Runtime **ON** |

**Pollution sweep:** `--pollution 0,2,8,…` = how many distractor filings get mixed into
the candidate pool. 0 = clean (target filing only); higher = noisier.

**Six metrics**, all logged per (model, arm, pollution, question): answer accuracy
(numeric-match + LLM-judge), retrieval precision/recall/hit vs gold pages, context
pollution % (share of off-target passages that reached the model), latency, token usage,
and Context Runtime's execution decisions (which config the bandit chose).

## Models under test (CPU, comparable memory footprint)

Everything runs on `llama.cpp` (CPU, GGUF) so the comparison is apples-to-apples and
reproducible without a GPU. Default set:

| model | total / active | quant | ~RAM |
|---|---|---|---|
| DeepSeek-V4-Flash | 284B / 13B | Q4_K_XL | ~155 GB |
| Nemotron-3-Super-120B-A12B | 121B / 13B | Q4_K_M | ~87 GB |
| Qwen3.6-35B-A3B | 35B / 3B | Q8_0 | ~37 GB |

The 37 GB Qwen is the deliberately-small model that gets Context Runtime. If it matches or
beats the 87–155 GB models as pollution rises, the small model won using *less* memory.

## Reproduce

```bash
# 0. deps: a recent llama.cpp (CPU build) + Python 3.11+. From the repo root:
pip install -e .                       # the importable core of context-runtime (no torch)

# 1. data — fetches the 84 public FinanceBench filings + builds the passage corpus
benchmarks/context-vs-model/scripts/download_data.sh

# 2. serve a model on CPU  (--reasoning off is required: see the script header)
benchmarks/context-vs-model/scripts/serve_model.sh \
    /models/Qwen3.6-35B-A3B-Q8_0.gguf Qwen3.6-35B-A3B 8080

# 3. run the sweep (resumable; each row appended to the JSONL)
cd benchmarks/context-vs-model
PYTHONPATH=../..:. python -m harness.run \
    --model-name qwen3.6 --base-url http://127.0.0.1:8080/v1 --model Qwen3.6-35B-A3B \
    --pollution 0,2,8 --out results/qwen.jsonl \
    --judge-base-url $JUDGE_URL --judge-model $JUDGE_MODEL   # judge optional

# 4. aggregate → table + crossover plot
PYTHONPATH=../..:. python -m harness.report results/*.jsonl \
    --md results/summary.md --plot results/crossover.png
```

## Design notes

- **No torch, no vector DB.** The retriever is a compact BM25 (`harness/retriever.py`) so
  the harness runs anywhere. BM25 is a fair substrate precisely because cross-company
  distractors share line-item vocabulary — lexical retrieval is what struggles, and gating
  is what Context Runtime adds.
- **Context Runtime is the real thing.** `harness/tuner.py` drives the actual
  `EpsilonGreedyBandit` over `redevops_rag.RetrievalConfig` arms, keyed by planner intent —
  the same mechanism as the shipped tenants, just over the lean BM25 backend so the
  benchmark stays dependency-light.
- **Grading is ground-truth-first.** Numeric/exact-match is authoritative when it fires
  (`$1,577 million` ≈ `$1577.00`); a neutral frontier judge handles prose and unit slips.
- **`--reasoning off`.** Reasoning models otherwise burn the token budget thinking and
  never answer on hard long-context questions; disabling it is uniform, cheaper, and fair.

This benchmark lives on the `benchmarks` branch of
[redevops-io/context-runtime](https://github.com/redevops-io/context-runtime). Reproduce
the results — or extend them with your own models and datasets.
