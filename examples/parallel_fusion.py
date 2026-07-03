"""Parallel sharded retrieval — scheduler fans out, Polars fuses.

Splits a corpus across N shards (each with simulated per-shard I/O latency), then:
  * fans out the query to all shards CONCURRENTLY (wall-clock ≈ slowest shard), and
  * RRF-fuses the union with Polars if installed, else the pure-Python fallback.

Shows the parallel fan-out beating sequential, that fusion is order-independent
(parallel and sequential produce identical top-K), and which fuse engine ran.

    python examples/parallel_fusion.py            # polars if `pip install context-runtime[polars]`
"""
from __future__ import annotations

import re
import time

from context_runtime.scheduler.parallel_fusion import (
    ShardedRetriever,
    fuse,
    polars_available,
)
from context_runtime.types import Hit

_WORD = re.compile(r"[a-z0-9]+")
SHARD_LATENCY_S = 0.05   # simulate per-shard I/O (network / disk)


class SlowShard:
    """A tiny token-overlap retriever over a slice of the corpus, with simulated I/O."""

    def __init__(self, name: str, docs: list[tuple[str, str]]):
        self.name = name
        self.docs = docs   # [(chunk_id, text)]

    def search(self, query: str, k: int, method: str = "hybrid") -> list[Hit]:
        time.sleep(SHARD_LATENCY_S)   # stand-in for a real shard round-trip
        q = set(_WORD.findall(query.lower()))
        scored = []
        for cid, text in self.docs:
            toks = set(_WORD.findall(text.lower()))
            overlap = len(q & toks)
            if overlap:
                scored.append((overlap / (len(toks) ** 0.5), cid, text))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [Hit(chunk_id=cid, filename=self.name, text=text, score=round(sc, 4))
                for sc, cid, text in scored[:k]]


CORPUS = [
    ("d01", "Context Runtime is a query planner for LLM context."),
    ("d02", "The runtime decides what a model sees before it answers."),
    ("d03", "Reciprocal rank fusion merges rankings from multiple retrievers."),
    ("d04", "Polars is a fast multi-threaded dataframe library in Rust."),
    ("d05", "Sharding splits a corpus across independent indices for scale."),
    ("d06", "A bandit learns which context bundle to retrieve per query."),
    ("d07", "DuckDB supports full-text search and vector similarity."),
    ("d08", "Parallel fan-out makes wall-clock the slowest shard, not the sum."),
    ("d09", "Semantic retrieval bridges synonyms that lexical search misses."),
    ("d10", "The scheduler fires shard queries concurrently, then fuses results."),
    ("d11", "Postgres uses tsvector and pgvector for hybrid retrieval."),
    ("d12", "Reranking reorders the fused candidates before the model reads them."),
]


def shard(corpus, n):
    return [SlowShard(f"shard{i}", corpus[i::n]) for i in range(n)]


def main() -> None:
    n_shards = 6
    shards = shard(CORPUS, n_shards)
    query = "how does the scheduler fuse shard results in parallel"
    k = 5

    retr = ShardedRetriever(shards, engine="auto")

    # parallel (fan-out) — the plugin
    fused_parallel = retr.search(query, k=k)
    st = retr.last_stats

    # sequential baseline — same shards, one after another, same fusion
    t0 = time.perf_counter()
    ranked_seq = [s.search(query, k * 3, "hybrid") for s in shards]
    seq_fanout_ms = round((time.perf_counter() - t0) * 1000, 2)
    fused_seq = fuse(ranked_seq, k=k, engine="python")

    print(f"shards={st['shards']}  candidates={st['candidates']}  fuse-engine={st['engine']}"
          f"  (polars installed: {polars_available()})")
    print(f"fan-out wall-clock:  parallel {st['fanout_ms']:>7.2f} ms   "
          f"vs sequential {seq_fanout_ms:>7.2f} ms   "
          f"({seq_fanout_ms / max(st['fanout_ms'], 0.01):.1f}x faster)")
    print(f"fusion:              {st['fuse_ms']:.2f} ms")

    ids_parallel = [h.chunk_id for h in fused_parallel]
    ids_seq = [h.chunk_id for h in fused_seq]
    print(f"\ntop-{k} (parallel):   {ids_parallel}")
    print(f"top-{k} (sequential): {ids_seq}")
    print(f"identical fusion: {ids_parallel == ids_seq}")
    for h in fused_parallel:
        print(f"  {h.chunk_id} [{h.filename}]  {h.text}")

    # correctness: fusion is order-independent, so parallel == sequential top-K
    assert ids_parallel == ids_seq, "parallel and sequential fusion diverged"
    # the fan-out should genuinely overlap the shard latencies
    assert st["fanout_ms"] < seq_fanout_ms, "parallel fan-out was not faster than sequential"
    print("\nOK — parallel fan-out + RRF fusion verified.")


if __name__ == "__main__":
    main()
