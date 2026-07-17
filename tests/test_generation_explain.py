"""Generation-strategy layer (Phase 3) — transparency parity in compare + EXPLAIN.

The generation decision is a first-class block alongside the retrieval decision: the strategy ladder
for the intent, each arm's config + learned value, and the entry point. Rendered in the EXPLAIN text.
"""
from __future__ import annotations

from context_runtime.explain import render_explain
from context_runtime.reasoner import strategies as gs


def test_explain_block_off_reports_legacy(monkeypatch):
    monkeypatch.delenv("CR_GENSTRATEGY", raising=False)
    blk = gs.explain_block("multi_hop")
    assert blk == {"enabled": False, "strategy": "single_shot",
                   "note": "generation-strategy layer off (set CR_GENSTRATEGY=1)"}


def test_explain_block_on_shows_ladder_and_entry_point(monkeypatch):
    monkeypatch.setenv("CR_GENSTRATEGY", "1")
    blk = gs.explain_block("multi_hop", method="hybrid", tier="cheap")
    assert blk["enabled"] and blk["ladder"] == ["decompose", "reason"]
    entry = [c for c in blk["candidates"] if c["entry_point"]]
    assert len(entry) == 1 and entry[0]["strategy"] == "decompose"    # cheapest-capable = entry point
    d = next(c for c in blk["candidates"] if c["strategy"] == "decompose")
    assert d["thinking"] is True and d["max_tokens"] == gs.get("decompose").max_tokens
    assert all(c["bandit"] == {"n": 0, "value": 0.0} for c in blk["candidates"])   # unlearned yet


def test_explain_block_surfaces_learned_values(monkeypatch):
    monkeypatch.setenv("CR_GENSTRATEGY", "1")

    class _Bandit:
        def value(self, ctx, arm):
            return (5, 0.87) if (ctx == "multi_hop" and arm == "hybrid:decompose:cheap") else (0, 0.0)

    blk = gs.explain_block("multi_hop", method="hybrid", tier="cheap", bandit=_Bandit())
    d = next(c for c in blk["candidates"] if c["strategy"] == "decompose")
    assert d["bandit"] == {"n": 5, "value": 0.87}


def _min_exp(generation):
    return {
        "request": "how does auth relate to the billing outage", "intent_bucket": "multi_hop",
        "query_type": None, "context_key": "multi_hop",
        "decision": {"chosen": {"key": "hybrid:k5:rr", "method": "hybrid", "final_k": 5, "rerank": True},
                     "candidates": [{"key": "hybrid:k5:rr", "method": "hybrid", "final_k": 5, "rerank": True,
                                     "cost_units": 1.0, "bandit": {"n": 3, "value": 0.7}, "quality": None,
                                     "chosen": True, "reason": "highest learned reward (0.7)"}]},
        "generation": generation,
        "retrieval": {"hybrid": []},
        "served": {"method": "hybrid", "n": 0, "citations": [], "max_p_rel": None,
                   "abstain": False, "abstain_reason": None},
        "reward": {"policy": "native implicit signal", "calibrated": False, "reward_beta": 0.5,
                   "quality_routing": False, "note": "reward = quality − λ·cost"},
    }


def test_render_explain_includes_the_generation_section(monkeypatch):
    monkeypatch.setenv("CR_GENSTRATEGY", "1")
    gen = gs.explain_block("multi_hop", method="hybrid", tier="cheap")
    text = render_explain(_min_exp(gen))
    assert "reasoning — strategy ladder for multi_hop" in text
    assert "decompose" in text and "reason" in text
    assert "entry point" in text


def test_render_explain_notes_when_generation_layer_off(monkeypatch):
    monkeypatch.delenv("CR_GENSTRATEGY", raising=False)
    text = render_explain(_min_exp(gs.explain_block("multi_hop")))
    assert "generation-strategy layer off" in text
