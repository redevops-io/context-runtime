"""Generation-strategy layer (Phase 1) — the answer-plane arms.

Offline (stub model): the layer is opt-in via CR_GENSTRATEGY. Off → plans are byte-identical to
before (legacy single_shot). On → the `reason` step becomes a bandit arm: one candidate per intent
strategy, carrying its thinking/budget, and the bandit arm key distinguishes strategies.
"""
from __future__ import annotations

from context_runtime import ContextRuntime
from context_runtime.optimizer.online import plan_key
from context_runtime.reasoner import strategies as gs
from context_runtime.types import Candidate, Constraints, Goal, StepSpec


def _rt():
    return ContextRuntime.default([
        {"chunk_id": "a::0", "filename": "a.md", "text": "The auth service issues tokens.", "created_at": None},
        {"chunk_id": "b::0", "filename": "b.md", "text": "The billing outage was caused by token expiry.", "created_at": None},
    ])


def _reason_strategy(cand) -> str:
    return next((s.params.get("strategy") for s in cand.steps if s.type == "reason"), None)


# ─────────────────────────── off: exact back-compat ───────────────────────────
def test_disabled_keeps_legacy_single_shot(monkeypatch):
    monkeypatch.delenv("CR_GENSTRATEGY", raising=False)
    rt = _rt()
    g = Goal(text="how does the auth service relate to the billing outage", constraints=Constraints())
    cands = rt.candidates.generate(rt.intent.analyze(g), g)
    assert cands and all(_reason_strategy(c) == "single_shot" for c in cands)
    # the reason step carries no thinking/budget params when off (byte-identical plan)
    rstep = next(s for s in cands[0].steps if s.type == "reason")
    assert "thinking" not in rstep.params and "max_tokens" not in rstep.params


# ─────────────────────────── on: strategy arms ───────────────────────────
def test_enabled_generates_per_intent_strategy_arms(monkeypatch):
    monkeypatch.setenv("CR_GENSTRATEGY", "1")
    rt = _rt()
    g = Goal(text="how does the auth service relate to the billing outage", constraints=Constraints())
    intent = rt.intent.analyze(g)
    assert intent.bucket == "multi_hop"
    cands = rt.candidates.generate(intent, g)
    strats = {_reason_strategy(c) for c in cands}
    # multi_hop's warm-start strategies are offered as distinct candidates
    assert strats == set(gs.strategies_for("multi_hop")) == {"decompose", "reason"}
    # each reason step now carries its strategy's thinking flag + token budget
    for c in cands:
        rstep = next(s for s in c.steps if s.type == "reason")
        spec = gs.get(rstep.params["strategy"])
        assert rstep.params["thinking"] == spec.thinking
        assert rstep.params["max_tokens"] == spec.max_tokens


def test_lookup_bucket_offers_only_terse(monkeypatch):
    monkeypatch.setenv("CR_GENSTRATEGY", "1")
    rt = _rt()
    g = Goal(text="ERR-500 status code", constraints=Constraints())
    intent = rt.intent.analyze(g)
    assert intent.bucket == "exact_lookup"
    cands = rt.candidates.generate(intent, g)
    assert {_reason_strategy(c) for c in cands} == {"terse"}   # cheapest arm only


# ─────────────────────────── the bandit arm key ───────────────────────────
def _cand(method, strategy, tier="cheap"):
    steps = (StepSpec("retrieve", {"method": method}), StepSpec("reason", {"strategy": strategy}))
    return Candidate(steps=steps, model_tier=tier)


def test_plan_key_folds_in_strategy_but_not_single_shot():
    # single_shot → legacy key (no strategy segment): existing learned values are preserved
    assert plan_key(_cand("hybrid", "single_shot")) == "hybrid:cheap"
    assert plan_key(_cand("hybrid", "")) == "hybrid:cheap"
    # a real strategy becomes a distinct arm
    assert plan_key(_cand("hybrid", "decompose")) == "hybrid:decompose:cheap"
    assert plan_key(_cand("graph", "reason", "premium")) == "graph:reason:premium"
    assert plan_key(_cand("hybrid", "decompose")) != plan_key(_cand("hybrid", "reason"))


# ─────────────────────────── strategy registry + extraction ───────────────────────────
def test_registry_covers_the_arms_and_orders_by_cost():
    for name in ("terse", "reason", "decompose", "mapreduce", "single_shot"):
        assert gs.get(name).name == name
    # cheapest → most expensive prior (the escalation-ladder ordering)
    assert gs.get("terse").cost_units < gs.get("reason").cost_units < gs.get("mapreduce").cost_units
    assert gs.get("terse").thinking is False and gs.get("reason").thinking is True


def test_extract_final_pulls_answer_and_strips_think():
    out = "<think>the auth service relates via tokens; billing failed on expiry</think>\n" \
          "Reasoning: token expiry cascaded.\nAnswer: token expiry"
    assert gs.extract_final(out) == "token expiry"
    assert gs.extract_final("just a direct answer") == "just a direct answer"


# ─────────────────────────── end-to-end dispatch (stub model) ───────────────────────────
def test_runtime_runs_a_strategy_plan(monkeypatch):
    monkeypatch.setenv("CR_GENSTRATEGY", "1")
    rt = _rt()
    res = rt.run("how does the auth service relate to the billing outage")
    # a generation strategy (not the legacy single_shot) was chosen and executed without error
    assert _reason_strategy(res.plan.chosen) in {"decompose", "reason"}
    assert isinstance(res.answer, str)
