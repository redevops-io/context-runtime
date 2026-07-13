# Benchmarks

Every result below is produced by a runnable example in [`examples/`](./examples) — no
invented numbers. Reproduce with `PYTHONPATH=. python examples/<name>.py`.

> **Better Context Beats More Context** — the knowledge-aware routing study (3 graph regimes ×
> 24B–284B models, SimGraph vs HippoRAG, and the LiveRAG polluted-context run) behind the
> [article of the same name](https://redevops.io/blog/better-context-beats-more-context) lives in
> [`benchmarks/context-vs-model/RESULTS.md`](./benchmarks/context-vs-model/RESULTS.md), organized by its
> three lessons: heavy infrastructure must earn its cost, context quality beats quantity, and runtime
> orchestration still matters at scale. Detailed lab reports in
> [`benchmarks/context-vs-model/reports/`](./benchmarks/context-vs-model/reports).

---

## v1 → v2, measured in both runtimes (headline)

**`examples/consolidated_benchmark.py`** — runs the *same* seeded, ground-truth retrieval
simulation in the Python source-of-truth **and** the Go port, and merges the results into one
table. v2 adds the DSpark-inspired upgrades: calibrated per-passage relevance folded into the
reward, grounded abstention, and the load-aware expensive-stage sizer. Reproduce with
`PYTHONPATH=. python examples/consolidated_benchmark.py` (add `--html out.html` for the site card).

| Metric | Py v1 | Py v2 | Δ | Go v1 | Go v2 | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| Learned-policy precision | 67.6% | 82.2% | ▲ +14.6 pts | 84.6% | 95.9% | ▲ +11.3 pts |
| Abstention recall (unanswerable caught) | 0.0% | 100.0% | ▲ +100.0 pts | 0.0% | 100.0% | ▲ +100.0 pts |
| False-abstain rate (answerable dropped) | 0.0% | 0.0% | — | 0.0% | 0.0% | — |
| Expensive-stage depth (passages) | 8.00 | 3.00 | ▼ −62% | 8.00 | 3.00 | ▼ −63% |
| Precision after the sizer | 37.5% | 100.0% | ▲ +62.5 pts | 37.5% | 100.0% | ▲ +62.5 pts |

_40-seed average; precision headlined at β=0.9 (the calibration-trust knob; shipped default 0.5).
Go is an **independent re-implementation on identical methodology** — the two runtimes agree
directionally on every effect, which is the point: the policy improvement is a property of the
design, not of one language's simulation. Absolute precision differs because each runtime uses its
own strategy set (the achievable ceiling differs), but both show a double-digit lift, full
abstention that v1 lacks, and a ~62% depth cut with precision rising to 100%._

---

## Retrieval over heterogeneous personal data (financial × medical)

**`examples/heterogeneous_shards.py`** — the interesting, non-obvious one.

A real user's local files are a mix of very different data. This runs against **real
FinanceBench 10-K pages** plus a small **medical corpus deliberately built to collide on
vocabulary** — words that mean different things in each domain: *discharge* (hospital vs
debt), *statement* (patient vs financial), *balance* (fluid vs sheet), *chronic* / *acute*
(condition vs distress), *liability*.

Three strategies over the same data — a flat mixed index, sharded + RRF fusion, and
sharded + coverage routing — measured on 8 medical probes over **3000 financial pages + 16
clinical notes**:

| Strategy | Medical recall | Cross-domain noise (top-5) |
|---|---|---|
| flat mixed index | 8/8 | **2.5** finance docs / query |
| sharded + RRF fuse | 8/8 | **2.9** *(fusion makes it worse)* |
| sharded + **coverage-routed** | 8/8 | **0.0** |

**The surprise:** it is *not* a burial problem. BM25's length-normalization keeps the short,
focused clinical note at **rank #1** in every strategy — recall is 8/8 across the board. The
real failure is **context pollution**: a collision query like *"discharge summary"* drags
10-K *"discharge of liability"* pages into the top-k, and naive RRF fusion (which pulls from
every shard) injects **more** cross-domain noise, not less.

**Coverage routing** fixes it. Instead of comparing raw BM25 scores across shards — which is
meaningless when one shard has 16 docs and another has 3000 — it scores each shard by its best
hit's **query-term coverage** (a corpus-statistics-independent signal) and fuses only the
shard(s) the query actually belongs to. That cut cross-domain noise **22 → 0 docs across the 8
queries, bidirectionally** (financial queries return zero medical docs too), with recall
intact.

The routing threshold (`route_margin`) is exactly the kind of policy a Context Runtime bandit
learns per query bucket — see the chat-memory tenant below.

```
corpus: 3000 financial 10-K pages + 16 medical notes  (k=5)

MEDICAL probes  (recall of the right note  |  cross-domain noise in top-5)
  strategy                recall   avg-noise
  mixed flat index        8/8      2.50 fin-docs/query
  sharded + RRF fuse      8/8      2.88 fin-docs/query
  sharded + routed        8/8      0.00 fin-docs/query

verdict: coverage routing cut cross-domain noise 22 -> 0 docs across 8 medical
         queries while keeping recall 8/8.
```

Backends: the shards are `InMemoryStore` by default; `adapters/store_duckdb.py` gives the same
retrieval over a persistent DuckDB `fts` index (one file per shard / per user's local data),
and `adapters/store_postgres.py` over a Postgres `tsvector` index — the *same* flat-vs-routed
result holds on all three.

---

## 3-index chat memory — learning which index to read

**`examples/chat_memory.py`** — an Elastic-Atlas-style agent memory with three recall modes
(**recency / semantic / entity**) exposed as one `RetrieverPlugin`. A per-bucket
ε-greedy bandit learns which single index to read per query bucket, from `value − read-cost`.

| | Context Runtime (learned) | baseline (read all three) | lift |
|---|---|---|---|
| reward | **1.23** | −1.70 | **+2.93** |

Learned policy: `followup → recency`, `factual → semantic`, `entity → entity`. Reading all
three indices always answers but pays the full read cost; CR learns the cheap decisive index
per bucket. The offline example simulates the reward; `ChatMemoryTenant.record_feedback()` is
the live path (a real thumbs / task-success signal).

---

## Parallel sharded fusion

**`examples/parallel_fusion.py`** — the scheduler fans a query out to N shards concurrently,
then RRF-fuses the union (Polars when installed, else a pure-Python fallback that is
byte-identical).

| | value |
|---|---|
| shards | 6 |
| parallel fan-out vs sequential | **5.8× faster** (wall-clock = slowest shard, not the sum) |
| fusion top-K parallel vs sequential | **identical** |

Polars parallelizes the *fusion*; the concurrent fan-out (optionally planned by the
`SchedulerPlugin`) is the separate win.

## Calibrated, load-aware retrieval (v1 vs v2) — the DSpark-inspired additions

**`examples/dspark_calibration_bench.py`** — a seeded simulation of the LibreChat
self-learning loop over a stub corpus with **ground-truth per-passage relevance**, so we
can score what was actually served. The per-query judge models the real heuristic judge: it
scores **term coverage / recall**, so it rewards dumping more passages and is blind to
precision — the coarse signal the per-passage calibrated `P(relevant)` corrects. Because the
v2 features are opt-in, *v2 with them off is byte-for-byte v1* — the A/B is a flag toggle in
one process. Each effect is isolated (precision measured on answerable, non-abstained only).

The reward comparison is **seed-averaged over 40 seeds**: a single run is dominated by
bandit exploration luck (every high-coverage arm ties on the coverage judge, so which one
the policy locks onto is random), so a lone run can read anywhere from +0 to +30 pts. The
systematic effect grows with how much the reward trusts calibrated relevance over the coarse
judge — the `beta` sweep:

| effect | v1 baseline | v2 | delta |
|---|---|---|---|
| **reward** — served true-precision, β=0.5 | 67.6% | 68.9% | **+1.3 pts** |
| **reward** — served true-precision, β=0.7 | 67.6% | 74.3% | **+6.7 pts** |
| **reward** — served true-precision, β=0.9 | 67.6% | 82.2% | **+14.6 pts** |
| **abstention** — unanswerable queries caught | 0% (can't) | 100% | — |
| **abstention** — answerable wrongly dropped | — | 0.0% | — |
| **sizer** — passages to the expensive stage (deep k=8 arm) | 8.00 | 3.00 | **−62%** |
| **sizer** — precision of that served set | 38% | 100% | pruned the low-relevance tail |

```
(1) reward     v1 67.6%  →  v2 68.9% (β0.5) / 74.3% (β0.7) / 82.2% (β0.9)   [40-seed avg]
(2) abstention v2 catches 100% of unanswerable queries, 0% false abstentions (v1 cannot)
(3) sizer      deep k=8 arm: 8.00 → 3.00 passages (−62%), precision 38% → 100%
```

**Why each moves:** (1) the coarse coverage judge makes judge-only (v1) chase deep,
low-precision arms; the reward now blends the mean calibrated relevance of the *served*
passages (`reward_beta`), and the more it trusts that over the judge, the better the learned
policy — monotone in β (β=0.3 is actually −2 pts, so the live default is 0.5). (2) a
calibrated `P(relevant)` floor lets the runtime say "not enough context" and skip the
upstream call — impossible on v1, whose scores aren't probabilities. (3) the sizer admits
passages by DSpark's cumulative survival product and stops when it decays, so a deep arm's
irrelevant tail never reaches synthesis. All default off; `v1`/`v2` git branches capture the
same toggle for an out-of-process A/B. (2) and (3) are deterministic; only (1) needs seed
averaging.
