# Benchmarks

Every result below is produced by a runnable example in [`examples/`](./examples) — no
invented numbers. Reproduce with `PYTHONPATH=. python examples/<name>.py`.

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
can score what was actually served. The per-query judge is deliberately noisy (one coarse
scalar, like the real judge); the per-passage calibrated `P(relevant)` is the cleaner
signal the reward used to throw away. Because the v2 features are opt-in, *v2 with them off
is byte-for-byte v1* — the A/B is a flag toggle in one process. Each effect is isolated to
avoid confounds (precision is measured on answerable, non-abstained queries only).

| effect | v1 baseline | v2 | delta |
|---|---|---|---|
| **reward** — served true-precision (abstention off) | 73.8% | 82.2% | **+8.4 pts** |
| **abstention** — unanswerable queries caught | 0% (can't) | 100% | — |
| **abstention** — answerable wrongly dropped | — | 0.0% | — |
| **sizer** — passages to the expensive stage (deep k=8 arm) | 8.00 | 3.00 | **−62%** |
| **sizer** — precision of that served set | 38% | 100% | pruned the low-relevance tail |

```
(1) reward       v1 judge-only 73.8%  →  v2 judge + calibrated relevance 82.2%   (+8.4 pts)
(2) abstention   v2 catches 100% of unanswerable queries, 0% false abstentions   (v1 cannot)
(3) sizer        deep k=8 arm: 8.00 → 3.00 passages (−62%), precision 38% → 100%
```

**Why each moves:** (1) the bandit reward finally includes the mean calibrated relevance of
the *served* passages (`reward_beta`), not just the noisy per-query judge — lower-variance
signal ⇒ a better learned policy. (2) a calibrated `P(relevant)` floor lets the runtime say
"not enough context" and skip the upstream call — impossible on v1, whose scores aren't
probabilities. (3) the sizer admits passages by DSpark's cumulative survival product and
stops when it decays, so a deep arm's irrelevant tail never reaches synthesis. All default
off; `v1` and `v2` git branches capture the same toggle for an out-of-process A/B.
