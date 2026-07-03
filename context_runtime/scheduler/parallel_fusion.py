"""Parallel sharded retrieval — scheduler fans out, Polars fuses/reranks.

At scale a corpus is split across shards/sources (per-tenant DuckDB files, a remote
index, an API). Two costs dominate: (1) the I/O of hitting every shard, and (2) the
merge/rerank over the combined candidate set. They want different tools:

    fan-out  — a concurrent scheduler (ThreadPoolExecutor here; in the full runtime the
               SchedulerPlugin/TopoScheduler drives it) fires all shard searches at once,
               so wall-clock is the slowest shard, not the sum.
    fuse     — reciprocal-rank fusion over the union of candidates. Vectorized with
               **Polars** (multi-threaded group-by) when the extra is installed; falls
               back to the pure-Python RRF so the core keeps its zero-dependency promise.

`ShardedRetriever` implements the RetrieverPlugin surface (search/info), so it composes:
each shard can itself be any retriever (bm25/semantic/hybrid/chat-memory index).
Polars parallelizes the FUSION, not the I/O — the two together are the win.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from ..types import Hit, PluginInfo


class _Shard(Protocol):
    def search(self, query: str, k: int, method: str) -> list[Hit]: ...


def _key(h: Hit) -> str:
    return f"{h.filename}\x00{h.chunk_id}"


def _rrf_python(ranked_lists: list[list[Hit]], k: int, c: int = 60) -> list[Hit]:
    """Pure-Python reciprocal-rank fusion (the zero-dep fallback)."""
    score: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits):
            kk = _key(h)
            score[kk] = score.get(kk, 0.0) + 1.0 / (c + rank + 1)
            best.setdefault(kk, h)
    fused = sorted(best.values(), key=lambda h: (-score[_key(h)], h.filename, h.chunk_id))
    return fused[:k] if k > 0 else fused


def _rrf_polars(ranked_lists: list[list[Hit]], k: int, c: int = 60) -> list[Hit]:
    """Vectorized RRF via Polars: one DataFrame of (key, rrf-contribution), a threaded
    group-by sum, sort, top-k. Wins when the candidate union is large (many shards ×
    many hits). Raises ImportError if polars is absent — the caller falls back."""
    import polars as pl

    rows = []
    best: dict[str, Hit] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits):
            kk = _key(h)
            rows.append((kk, 1.0 / (c + rank + 1)))
            best.setdefault(kk, h)
    if not rows:
        return []
    df = (
        pl.DataFrame(rows, schema=["key", "rrf"], orient="row")
        .group_by("key")
        .agg(pl.col("rrf").sum().alias("score"))
        # deterministic tiebreak on key (filename\x00chunk_id) so this matches the
        # pure-Python engine byte-for-byte — the two must be interchangeable.
        .sort(["score", "key"], descending=[True, False])
    )
    keys = (df.head(k) if k > 0 else df)["key"].to_list()
    return [best[kk] for kk in keys]


def polars_available() -> bool:
    try:
        import polars  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def fuse(ranked_lists: list[list[Hit]], k: int, c: int = 60, engine: str = "auto") -> list[Hit]:
    """RRF-fuse ranked lists. engine: 'polars' | 'python' | 'auto' (polars if installed)."""
    if engine == "python":
        return _rrf_python(ranked_lists, k, c)
    if engine == "polars":
        return _rrf_polars(ranked_lists, k, c)
    try:
        return _rrf_polars(ranked_lists, k, c)
    except Exception:  # noqa: BLE001 — polars missing / any failure → deterministic fallback
        return _rrf_python(ranked_lists, k, c)


class ShardedRetriever:
    """Fan out a query to N shards concurrently, RRF-fuse the union. Drop-in
    RetrieverPlugin whose shards are themselves retrievers."""

    def __init__(self, shards: list[_Shard], *, max_workers: int | None = None,
                 pool_per_shard: int = 3, engine: str = "auto"):
        if not shards:
            raise ValueError("ShardedRetriever needs at least one shard")
        self.shards = shards
        self.max_workers = max_workers or min(16, len(shards))
        self.pool_per_shard = pool_per_shard   # over-fetch per shard before fusion
        self.engine = engine
        self.last_stats: dict = {}

    def search(self, query: str, k: int, method: str = "hybrid") -> list[Hit]:
        pool = max(k * self.pool_per_shard, k)
        t0 = time.perf_counter()
        # fan-out: every shard search runs concurrently (the scheduler).
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            ranked = list(ex.map(lambda s: _safe_search(s, query, pool, method), self.shards))
        t1 = time.perf_counter()
        fused = fuse(ranked, k=k, engine=self.engine)
        t2 = time.perf_counter()
        self.last_stats = {
            "shards": len(self.shards),
            "candidates": sum(len(r) for r in ranked),
            "fanout_ms": round((t1 - t0) * 1000, 2),
            "fuse_ms": round((t2 - t1) * 1000, 2),
            "engine": "polars" if (self.engine != "python" and polars_available()) else "python",
        }
        return fused

    def info(self) -> PluginInfo:
        caps = {"parallel", "rrf"}
        if polars_available():
            caps.add("polars")
        return PluginInfo(name="sharded-parallel-fusion", kind="retriever", version="0.1",
                          capabilities=frozenset(caps))


def _safe_search(shard: _Shard, query: str, k: int, method: str) -> list[Hit]:
    """A dead/erroring shard drops to an empty list rather than failing the whole fan-out."""
    try:
        return list(shard.search(query, k, method))
    except Exception:  # noqa: BLE001
        return []
