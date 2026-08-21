"""Slice 5 — freshness as a scored dimension, REFRESH verdict, and evidence lineage in EXPLAIN.

Freshness/confidence are optimizer inputs; a REFRESH abstention outcome fires when a confident plan is
backed by stale evidence; EXPLAIN surfaces content-hash / source-version / capability-version. All
additive: a fully-fresh plan scores exactly as before, and the REFRESH gate is off by default.
"""
from __future__ import annotations

from context_runtime.types import PlanScore
from context_runtime.costmodel.score import total, finalize, DEFAULT_WEIGHTS
from context_runtime.abstention import AbstentionGate
from context_runtime.explain import render_explain


# ── freshness as a scored dimension ──

def test_fresher_plan_scores_higher_and_fresh_is_unchanged():
    fresh = PlanScore(expected_accuracy=0.8, freshness=1.0)
    stale = PlanScore(expected_accuracy=0.8, freshness=0.2)
    assert total(fresh) > total(stale)
    # a fully-fresh plan (default freshness) is scored EXACTLY as before Slice 5 — staleness term is 0
    assert total(fresh) == total(PlanScore(expected_accuracy=0.8))
    # the penalty magnitude is w[stale] * (1 - freshness)
    assert abs((total(fresh) - total(stale)) - DEFAULT_WEIGHTS["stale"] * (1 - 0.2)) < 1e-9


# ── REFRESH abstention outcome ──

def test_refresh_when_confident_but_stale():
    gate = AbstentionGate(min_confidence=0.5, min_freshness=0.5)
    v = gate.evaluate(PlanScore(expected_accuracy=0.9, freshness=0.2))
    assert v.action == "refresh" and v.refresh
    assert not v.abstained and v.freshness == 0.2
    # confident AND fresh → serve
    assert gate.evaluate(PlanScore(expected_accuracy=0.9, freshness=1.0)).action == "serve"


def test_refresh_gate_is_off_by_default():
    gate = AbstentionGate(min_confidence=0.5)          # min_freshness defaults to 0.0
    v = gate.evaluate(PlanScore(expected_accuracy=0.9, freshness=0.0))
    assert v.action == "serve"                          # freshness ignored → no behavior change


def test_stale_and_unsure_still_abstains_not_refreshes():
    gate = AbstentionGate(min_confidence=0.8, min_freshness=0.0)   # freshness gate off
    v = gate.evaluate(PlanScore(expected_accuracy=0.2, freshness=0.1))
    assert v.action == "abstain"


# ── evidence lineage in EXPLAIN ──

def _exp():
    return {
        "request": "what did revenue do after Q2", "intent_bucket": "analytical", "context_key": "k",
        "decision": {"candidates": [{"key": "retrieval:hybrid", "chosen": True,
                                     "bandit": {"value": 0.5, "n": 3}, "cost_units": 1.0, "reason": "best arm"}]},
        "retrieval": {"hybrid": [{"served": True, "chunk_id": "kb.md::0", "score": 1.2, "p_rel": 0.7,
                                  "content_hash": "sha256:deadbeefcafe0000", "version": "v7"}]},
        "served": {"n": 1, "method": "hybrid", "citations": ["kb.md::0"],
                   "freshness": 0.35, "refresh": True, "refresh_reason": "source revised since retrieval"},
        "reward": {"policy": "reward v1", "note": "learned"},
        "lineage": [{"ref": "crm://acct/42", "version": "v7", "content_hash": "sha256:deadbeefcafe0000",
                     "source": "crm", "capability_version": "readers@3", "freshness": 0.9}],
    }


def test_explain_surfaces_versioned_lineage_and_refresh():
    txt = render_explain(_exp())
    # a dedicated lineage section citing the exact revision, its source, capability version, freshness
    assert "evidence lineage" in txt
    assert "crm://acct/42@v7" in txt
    assert "sha256:deadbeef" in txt          # content hash surfaced
    assert "cap=readers@3" in txt            # capability/model version surfaced
    assert "fresh=0.90" in txt
    # per-hit revision annotation + REFRESH in the served line
    assert "⟨@v7" in txt
    assert "REFRESH — source revised since retrieval" in txt
    assert "freshness 0.35" in txt


# ── Benchmark C: freshness arm ──

def test_benchmark_c_freshness_arm():
    # fixed retrieval binds a stale source; adaptive routing picks a fresh alternative of equal quality.
    fixed_stale = finalize(PlanScore(expected_accuracy=0.8, freshness=0.2))
    adaptive_fresh = finalize(PlanScore(expected_accuracy=0.8, freshness=1.0))
    # the optimizer prefers the fresher plan on total utility ...
    assert adaptive_fresh.total > fixed_stale.total
    # ... and at serve time the stale plan REFRESHes while the fresh one serves
    gate = AbstentionGate(min_confidence=0.5, min_freshness=0.5)
    assert gate.evaluate(fixed_stale).refresh
    assert gate.evaluate(adaptive_fresh).action == "serve"
