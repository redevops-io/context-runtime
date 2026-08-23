"""Bounded, order-preserving parallel execution for independent retrieval legs (LLM-parallelization
audit, P1). Hybrid retrieval runs its BM25 and vector legs, and two-stage runs its stage-1 methods, as
independent siblings that today evaluate serially (positional-arg / loop order). ``run_parallel`` overlaps
them when opted in via ``CR_RETRIEVAL_CONCURRENCY`` (default 1 = unchanged serial), preserving result
order so the downstream RRF fusion is identical."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor


def retrieval_concurrency() -> int:
    try:
        return max(1, int(os.getenv("CR_RETRIEVAL_CONCURRENCY", "1")))
    except ValueError:
        return 1


def run_parallel(thunks) -> list:
    """Run a list of 0-arg callables, returning results in input order. Serial when concurrency<=1
    (default) or a single thunk; otherwise a bounded ThreadPool. The thunks must be independent."""
    thunks = list(thunks)
    c = retrieval_concurrency()
    if c <= 1 or len(thunks) <= 1:
        return [t() for t in thunks]
    with ThreadPoolExecutor(max_workers=min(c, len(thunks))) as pool:
        futures = [pool.submit(t) for t in thunks]
        return [f.result() for f in futures]        # input order preserved
