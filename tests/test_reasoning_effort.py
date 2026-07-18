"""Reasoning-effort control (A–D) — self-consistency arm, refine depth, effort menu/Pareto. Parity
with context-runtime-go/reasoner/effort_test.go. Model-free (a fake ModelPlugin)."""
from types import SimpleNamespace

import pytest

from context_runtime.reasoner import strategies as S
from context_runtime.reasoner.verify import consensus_index
from context_runtime.reasoner.strategy_reasoner import StrategyReasoner
from context_runtime.optimizer.online import plan_key
from context_runtime.types import Candidate, ModelResult, ReasonRequest, StepSpec


# ── A: consensus + gating + plan_key ──────────────────────────────────────────────────────────
def test_consensus_index_skips_abstention_and_picks_majority():
    assert consensus_index(["NOT FOUND", "alpha beta", "alpha beta"], "alpha beta gamma") in (1, 2)
    assert consensus_index(["x y z", "alpha beta", "alpha beta"], "alpha beta") in (1, 2)


def test_offers_self_consistency_gated(monkeypatch):
    monkeypatch.setenv("CR_GENSTRATEGY", "1")
    monkeypatch.delenv("CR_SELFCONSISTENCY", raising=False)
    assert not S.offers_self_consistency("multi_hop")
    monkeypatch.setenv("CR_SELFCONSISTENCY", "1")
    assert S.offers_self_consistency("multi_hop")
    assert not S.offers_self_consistency("exact_lookup")


def test_plan_key_encodes_sc_and_verify():
    c = Candidate(steps=(StepSpec("retrieve", {"method": "hybrid"}),
                         StepSpec("reason", {"strategy": "decompose", "self_consistency": 5, "verify": True})),
                  model_tier="premium")
    assert plan_key(c) == "hybrid:decompose+sc+v:premium"


class FakeModel:
    """Records temperature per call; returns scripted texts so consensus/refine are observable."""
    def __init__(self, texts):
        self.texts, self.calls, self.temps = texts, 0, []

    def complete(self, req):
        if req.temperature is not None:
            self.temps.append(req.temperature)
        txt = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return ModelResult(text=txt, model="fake", tier="cheap",
                           prompt_tokens=10, completion_tokens=5, est_cost_usd=0.01)

    def capabilities(self, model):  # pragma: no cover - interface shim
        from context_runtime.types import ModelCapabilities
        return ModelCapabilities()

    def count_tokens(self, text, model=""):  # pragma: no cover
        return len(text.split())

    def info(self):
        from context_runtime.types import PluginInfo
        return PluginInfo(name="fake", kind="model")


def _req(ctx_text, question):
    ctx = SimpleNamespace(assembled_text=ctx_text,
                          plan=SimpleNamespace(id="p1", intent=SimpleNamespace(normalized=question)))
    return ReasonRequest(context=ctx, capability="synthesis")


def test_self_consistency_samples_and_rolls_up_cost():
    m = FakeModel(["Answer: alpha beta", "Answer: alpha beta", "Answer: zzz qqq"])
    r = StrategyReasoner(m, "reason", samples=3)
    res = r.reason(_req("alpha beta gamma", "q?"))
    assert m.calls == 3 and len(m.temps) == 3 and m.temps[0] > 0     # K stochastic samples
    assert res.text == "alpha beta"                                  # consensus of the agreeing pair
    assert res.prompt_tokens == 30 and res.est_cost_usd == pytest.approx(0.03)  # K-sample cost
    assert "+sc" in r.info().name


# ── C: refine depth ───────────────────────────────────────────────────────────────────────────
def test_refine_depth_retries_up_to_budget(monkeypatch):
    monkeypatch.setenv("CR_REFINE_DEPTH", "3")
    assert S.refine_depth() == 3
    m = FakeModel(["zzz", "zzz", "zzz", "zzz"])           # never grounded → retries to the budget
    StrategyReasoner(m, "reason", verify=True).reason(_req("alpha beta gamma", "q?"))
    assert m.calls == 4                                    # 1 initial + 3 refine rounds


def test_verify_stops_when_grounded(monkeypatch):
    monkeypatch.setenv("CR_REFINE_DEPTH", "3")
    m = FakeModel(["alpha beta gamma"])                   # grounded first attempt → no retry
    StrategyReasoner(m, "reason", verify=True).reason(_req("alpha beta gamma delta", "q?"))
    assert m.calls == 1


# ── D: effort menu / Pareto ───────────────────────────────────────────────────────────────────
def test_effort_menu_marks_pareto(monkeypatch):
    monkeypatch.setenv("CR_GENSTRATEGY", "1")
    monkeypatch.setenv("CR_SELFCONSISTENCY", "1")
    blk = S.explain_block("multi_hop", method="hybrid", tier="premium")
    menu = blk["effort_menu"]
    assert menu and any("+sc" in a["arm"] for a in menu)
    cheapest = min(menu, key=lambda a: a["cost_units"])
    assert cheapest["pareto_optimal"] is True            # at zero learning the cheapest dominates
