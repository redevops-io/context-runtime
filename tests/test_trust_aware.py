"""Generation 5 — Trust-Aware Execution: trust as a ranking objective (not just a tie-breaker) and a
calibrated abstention gate (serve / escalate / abstain).
"""
from __future__ import annotations

from context_runtime.abstention import AbstentionGate
from context_runtime.optimizer.online import BanditOptimizer
from context_runtime.types import Candidate, Goal, PlanScore, StepSpec


def _cand(tier: str, method: str = "bm25") -> Candidate:
    return Candidate(steps=(StepSpec(type="retrieve", params={"method": method}),), model_tier=tier)


def _score(total: float, acc: float = 0.9) -> PlanScore:
    return PlanScore(total=total, expected_accuracy=acc, feasible=True)


class _Trust:
    def __init__(self, scores):
        self.scores = scores

    def score(self, cand: Candidate, goal: Goal) -> float:
        return self.scores.get(cand.model_tier, 0.5)


# ── trust as an objective ──

def test_trust_weight_zero_is_pure_cost_ranking():
    trust = _Trust({"premium": 0.1, "local": 0.9})
    opt = BanditOptimizer(None, epsilon=0.0, trust=trust, trust_weight=0.0)
    hi, lo = _cand("premium"), _cand("local")
    plan = opt.select([(hi, _score(0.9)), (lo, _score(0.5))], Goal(text="q"), context="synthesis")
    assert plan.chosen is hi   # trust only tie-breaks; cost total still wins


def test_trust_weight_folds_trust_into_the_objective():
    """A less costly-attractive but more trusted plan wins once trust is weighted into the objective."""
    trust = _Trust({"premium": 0.1, "local": 0.9})
    opt = BanditOptimizer(None, epsilon=0.0, trust=trust, trust_weight=0.7)
    hi, lo = _cand("premium"), _cand("local")
    # premium: 0.3*0.9 + 0.7*0.1 = 0.34 ; local: 0.3*0.5 + 0.7*0.9 = 0.78 → local wins
    plan = opt.select([(hi, _score(0.9)), (lo, _score(0.5))], Goal(text="q"), context="synthesis")
    assert plan.chosen is lo


def test_trust_on_bandit_no_longer_errors():
    """Regression: _arm_value now receives the goal, so a TrustProvider on the bandit works."""
    opt = BanditOptimizer(None, epsilon=0.0, trust=_Trust({"local": 0.9}))
    plan = opt.select([(_cand("local"), _score(0.5))], Goal(text="q"), context="c")
    assert plan.chosen.model_tier == "local"


# ── abstention gate ──

def test_gate_serves_when_confident():
    v = AbstentionGate(min_confidence=0.6).evaluate(_score(0.5, acc=0.8))
    assert v.action == "serve" and not v.abstained


def test_gate_abstains_when_below_bar_and_no_escalation():
    v = AbstentionGate(min_confidence=0.6).evaluate(_score(0.5, acc=0.3))
    assert v.action == "abstain" and v.abstained and "abstain" in v.reason


def test_gate_escalates_when_a_stronger_option_exists():
    gate = AbstentionGate(min_confidence=0.6, can_escalate=lambda s, g: True)
    assert gate.evaluate(_score(0.5, acc=0.3)).action == "escalate"


def test_gate_uses_calibration_map():
    """A raw score above the bar can be calibrated below it → abstain (calibrated P(correct))."""
    gate = AbstentionGate(min_confidence=0.6, calibrate=lambda c: c * 0.5)
    v = gate.evaluate(_score(0.5, acc=0.9))       # raw 0.9 → calibrated 0.45 < 0.6
    assert v.action == "abstain" and abs(v.confidence - 0.45) < 1e-9


def test_optimizer_records_abstention_verdict_in_plan_extra():
    gate = AbstentionGate(min_confidence=0.7)
    opt = BanditOptimizer(None, epsilon=0.0, abstain_gate=gate)
    plan = opt.select([(_cand("local"), _score(0.5, acc=0.3))], Goal(text="q"), context="c")
    ab = plan.extra["abstention"]
    assert ab["action"] == "abstain" and ab["confidence"] == 0.3
