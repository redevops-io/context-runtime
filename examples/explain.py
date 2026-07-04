#!/usr/bin/env python3
"""EXPLAIN — the runtime's answer to "why did it retrieve *that*?"

The database analogue of EXPLAIN ANALYZE: given a request, show the plan's decision (every candidate
arm with its learned value + quality/cost decomposition + why it won/lost), the per-method retrieval
trace with **calibrated P(relevant)** (so a high-raw-score-but-irrelevant hit is obvious), what was
served, the abstention decision, and how the reward is computed.

This runs against the same seeded, ground-truth stub as the DSpark benchmark (real method-quality
and per-method score scales), fits a calibration map, learns a policy over a query stream with
quality-routing on, then EXPLAINs a query. Deterministic. Exits 0.

Run:  PYTHONPATH=. python examples/explain.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from context_runtime.explain import render_explain
from context_runtime.integrations.librechat import DEFAULT_STRATEGIES, LibreChatTenant
from context_runtime.quality import QualityLedger

from examples.dspark_calibration_bench import (  # noqa: E402
    StubRetriever, coverage_judge, fit_calibration, _rng,
)


def main() -> int:
    tmp = tempfile.mkdtemp()
    cmap = fit_calibration(tmp)
    ledger = QualityLedger()   # in-memory; learns quality apart from cost
    tenant = LibreChatTenant(
        retriever=StubRetriever(), strategies=DEFAULT_STRATEGIES,
        calibration=cmap, reward_beta=0.9,
        quality_ledger=ledger, quality_routing=True, quality_min_samples=3,
    )
    # learn a policy over a seeded query stream (native-signal / judge bootstrap)
    noise = _rng(7)
    for i in range(400):
        q = f"user question {i}"
        ctx = tenant.retrieve(q)
        tenant.record_judgment(q, coverage_judge(ctx.hits, (noise() - 0.5) * 0.4))

    # explain a query of the same form the policy trained on, so learned values are populated
    print(render_explain(tenant.explain("user question 999", k=5)))
    print()
    print("Reproduce:  PYTHONPATH=. python examples/explain.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
