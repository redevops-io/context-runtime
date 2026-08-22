"""Freshness sourced from actual evidence, on the real serving path (v0.2.x stabilization).

The Slice-5 primitives (scored freshness, REFRESH gate, EXPLAIN lineage renderer) already exist and
are covered by test_freshness_slice5.py. This suite closes the audited seam: freshness is now derived
from the retrieved evidence's own source timestamp/version, reaches ``PlanScore.freshness`` on the
normal ``ContextRuntime.run`` path, drives REFRESH there, is named exactly in EXPLAIN, and — critically
— is a no-op unless a FreshnessPolicy is configured. These are plan tests 5–9.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from context_runtime import ContextRuntime, FreshnessPolicy
from context_runtime import freshness as F
from context_runtime.explain import render_explain

NOW = "2026-08-21T00:00:00Z"


def _docs(observed_at: str, version: str):
    """One-doc corpus carrying evidence identity (as the RAG binding now supplies)."""
    return [{
        "chunk_id": "kb.md::0", "filename": "kb.md",
        "text": "quarterly revenue growth accelerated after the pricing change",
        "observed_at": observed_at, "version": version, "content_hash": "rcv1:deadbeefcafe0000",
        "source_ref": "crm://acct/42",
    }]


def _iso(days_ago: float) -> str:
    return (datetime.fromisoformat(NOW.replace("Z", "+00:00")) - timedelta(days=days_ago)).isoformat()


# ── Test 5: retrieved evidence age/version actually changes PlanScore.freshness ──

def test_5_evidence_age_changes_planscore_freshness():
    policy = FreshnessPolicy(enabled=True, mode="age_decay", half_life_days=30.0, min_freshness=0.0)
    q = "what did revenue do"

    fresh_rt = ContextRuntime.default(_docs(_iso(0), "v9"), freshness_policy=policy)
    stale_rt = ContextRuntime.default(_docs(_iso(120), "v1"), freshness_policy=policy)

    r_fresh = fresh_rt.run(q, as_of=NOW)
    r_stale = stale_rt.run(q, as_of=NOW)

    # freshness is sourced from the evidence timestamp and lands on the result + plan score
    assert r_fresh.freshness > 0.9          # ~0 days old → ~1.0
    assert r_stale.freshness < 0.2          # 120 days at 30-day half-life → 0.5**4 = 0.0625
    assert r_fresh.freshness > r_stale.freshness
    assert r_stale.plan.score.freshness == r_stale.freshness


# ── Test 6: stale evidence crosses configured threshold → REFRESH ──

def test_6_stale_evidence_triggers_refresh_on_serving_path():
    policy = FreshnessPolicy(enabled=True, mode="age_decay", half_life_days=30.0, min_freshness=0.5)
    rt = ContextRuntime.default(_docs(_iso(120), "v1"), freshness_policy=policy)
    r = rt.run("what did revenue do", as_of=NOW)
    assert r.refresh is True                 # REFRESH reached on the normal run() path
    assert r.answer == ""                    # declines to serve stale evidence
    assert "freshness" in r.refresh_reason and r.freshness < 0.5


# ── Test 7: fresh evidence does not REFRESH ──

def test_7_fresh_evidence_serves():
    policy = FreshnessPolicy(enabled=True, mode="age_decay", half_life_days=30.0, min_freshness=0.5)
    rt = ContextRuntime.default(_docs(_iso(1), "v9"), freshness_policy=policy)
    r = rt.run("what did revenue do", as_of=NOW)
    assert r.refresh is False
    assert r.answer != ""                     # serves normally
    assert r.freshness > 0.9


# ── Test 8: EXPLAIN names the exact evidence ref/version/hash used ──

def test_8_explain_names_exact_evidence_ref_version_hash():
    policy = FreshnessPolicy(enabled=True, mode="age_decay", half_life_days=30.0, min_freshness=0.5)
    rt = ContextRuntime.default(_docs(_iso(120), "v1"), freshness_policy=policy)
    hits = rt.retriever.search("what did revenue do", k=5)
    assert hits and hits[0].version == "v1" and hits[0].content_hash == "rcv1:deadbeefcafe0000"

    # the generic producer names the exact evidence; the existing renderer surfaces it
    lineage = F.lineage_from_hits(hits, as_of=NOW, policy=policy, capability_version="readers@3")
    exp = {
        "request": "what did revenue do", "intent_bucket": "analytical", "context_key": "k",
        "decision": {"candidates": [{"key": "retrieval:hybrid", "chosen": True,
                                     "bandit": {"value": 0.5, "n": 1}, "cost_units": 1.0, "reason": "arm"}]},
        "retrieval": {"hybrid": [F.explain_hit_row(hits[0])]},
        "served": {"n": 1, "method": "hybrid", "freshness": lineage[0]["freshness"], "refresh": True,
                   "refresh_reason": "evidence too stale"},
        "reward": {"policy": "reward v1", "note": "learned"},
        "lineage": lineage,
    }
    txt = render_explain(exp)
    assert "crm://acct/42@v1" in txt            # exact source ref + version
    assert "rcv1:deadbeef" in txt               # exact content hash
    assert "cap=readers@3" in txt               # capability version
    assert "⟨@v1" in txt                        # per-hit revision annotation
    assert "REFRESH" in txt


# ── Test 9: legacy configuration with freshness enforcement disabled behaves exactly as before ──

def test_9_disabled_freshness_is_byte_for_byte_legacy():
    q = "what did revenue do"
    # identical corpus and query; one runtime has no policy, one has an explicitly-disabled policy,
    # one has a policy that WOULD refresh if it were enforced — with enforcement off all three serve.
    docs = _docs(_iso(3650), "v1")           # 10 years stale
    base = ContextRuntime.default([dict(d) for d in docs])
    disabled = ContextRuntime.default([dict(d) for d in docs],
                                      freshness_policy=FreshnessPolicy(enabled=False, min_freshness=0.9))

    rb = base.run(q, as_of=NOW)
    rd = disabled.run(q, as_of=NOW)
    assert rb.refresh is False and rd.refresh is False
    assert rb.freshness == 1.0 and rd.freshness == 1.0     # no sourcing when disabled
    assert rb.answer == rd.answer and rb.answer != ""      # identical served answer
    # and the freshness scoring term is a no-op at freshness=1.0 (Slice-5 invariant still holds)
    assert rb.plan.score.freshness == 1.0
