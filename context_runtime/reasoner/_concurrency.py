"""Bounded, order-preserving fork-join for the reasoners (LLM-parallelization audit, P0).

Several reasoners issue K/N *independent* model calls (self-consistency samples, debate answers, plan
workers) in a serial list-comp/`for`, each multiplying wall-clock by K/N. `fanout` overlaps them when
opted in, while keeping results in **input order** so any consensus/index-dependent downstream is
identical. Default is serial (unchanged behaviour); set ``CR_REASONER_CONCURRENCY`` > 1 to enable — the
same opt-in shape as the Mission Runtime executor.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor


def reasoner_concurrency() -> int:
    try:
        return max(1, int(os.getenv("CR_REASONER_CONCURRENCY", "1")))
    except ValueError:
        return 1


def fanout(fn, items):
    """Map ``fn`` over ``items``, order-preserving. Serial when concurrency<=1 (default) or a single item;
    otherwise a bounded ThreadPool. The calls must be independent (no shared mutable state) — which is
    exactly why these sites were flagged parallelizable."""
    items = list(items)
    c = reasoner_concurrency()
    if c <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(c, len(items))) as pool:
        return list(pool.map(fn, items))       # map preserves input order
