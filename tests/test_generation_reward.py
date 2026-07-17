"""Generation-strategy layer (Phase 2) — the measured-reward loop + escalation.

Offline + deterministic: the cheap verifier scores an answer, the reward folds into the SAME bandit
loop retrieval uses (keyed by the strategy-aware arm), and the escalation ladder picks the next rung
when a strategy underperforms.
"""
from __future__ import annotations

from context_runtime.optimizer.online import BanditOptimizer, plan_key
from context_runtime.reasoner import verify
from context_runtime.reasoner.verify import GenerationVerifier, apply_feedback, next_strategy, should_escalate
from context_runtime.types import Candidate, Goal, Intent, Plan, PlanScore, StepSpec

_CTX = "The auth service issues tokens. The billing outage was caused by token expiry."


# ─────────────────────────── the verifier ───────────────────────────
def test_reward_grounded_beats_hallucinated_beats_abstention():
    v = GenerationVerifier()
    grounded, _ = v.reward("token expiry", _CTX)
    hallucinated, _ = v.reward("a disk controller firmware bug", _CTX)
    abst, sig = v.reward("NOT FOUND", _CTX)
    assert grounded > hallucinated
    assert abst == verify.ABSTAIN_FLOOR and sig["abstained"] is True
    assert grounded >= 0.9   # every content token is in the context


def test_faithfulness_and_consistency_signals():
    assert verify.faithfulness("token expiry", _CTX) == 1.0
    assert verify.faithfulness("", _CTX) == 0.0
    assert verify.self_consistency(["token expiry", "expiry of the token"]) == 1.0
    assert verify.self_consistency(["token expiry", "disk failure"]) < 1.0


def test_pluggable_judge_overrides_the_proxy():
    v = GenerationVerifier(judge=lambda a, c, q: 0.42)
    r, sig = v.reward("anything at all", _CTX)
    assert r == 0.42 and sig["faithfulness"] == 0.42


# ─────────────────────────── escalation ladder ───────────────────────────
def test_ladder_walks_cheapest_to_costliest_then_stops():
    # multi_hop ladder = ("decompose", "reason")
    assert next_strategy("multi_hop", "decompose") == "reason"
    assert next_strategy("multi_hop", "reason") is None          # top of the ladder
    assert next_strategy("exact_lookup", "terse") is None        # single-rung ladder
    assert should_escalate(0.2) and not should_escalate(0.8)


# ─────────────────────────── reward → bandit (the self-optimization) ───────────────────────────
def _cand(method, strategy, tier="cheap"):
    return Candidate(steps=(StepSpec("retrieve", {"method": method}),
                            StepSpec("reason", {"strategy": strategy})), model_tier=tier)


class _Est:
    def estimate(self, c, g): return PlanScore(total=0.5, feasible=True)
    def statistics(self, bucket=None): raise NotImplementedError
    def observe(self, p, t): raise NotImplementedError


def test_apply_feedback_folds_generation_reward_into_the_strategy_arm():
    opt = BanditOptimizer(_Est(), epsilon=0.0)
    reason = _cand("hybrid", "reason")        # prior winner (0.9)
    decompose = _cand("hybrid", "decompose")  # prior 0.4
    scored = [(reason, PlanScore(total=0.9, feasible=True)),
              (decompose, PlanScore(total=0.4, feasible=True))]
    goal = Goal(text="how does auth relate to the billing outage")

    assert opt.select(scored, goal, context="multi_hop").chosen is reason   # prior wins first
    # decompose actually produces the grounded answer → high measured reward, fed via the verifier
    plan_d = Plan(intent=Intent(bucket="multi_hop"), chosen=decompose, score=PlanScore(),
                  extra={"bandit": {"context": "multi_hop", "arm": plan_key(decompose)}})
    for _ in range(3):
        out = apply_feedback(opt, plan_d, "token expiry", _CTX, bucket="multi_hop")
        assert out["reward"] >= 0.9 and out["escalate_to"] is None
    # the learned reward on the decompose arm now beats reason's prior
    assert opt.select(scored, goal, context="multi_hop").chosen is decompose
    n, mean = opt.bandit.value("multi_hop", plan_key(decompose))
    assert n == 3 and mean >= 0.9


def test_apply_feedback_flags_escalation_on_a_weak_answer():
    opt = BanditOptimizer(_Est(), epsilon=0.0)
    decompose = _cand("hybrid", "decompose")
    plan = Plan(intent=Intent(bucket="multi_hop"), chosen=decompose, score=PlanScore(),
                extra={"bandit": {"context": "multi_hop", "arm": plan_key(decompose)}})
    out = apply_feedback(opt, plan, "NOT FOUND", _CTX, bucket="multi_hop")
    assert out["reward"] == verify.ABSTAIN_FLOOR
    assert out["escalate_to"] == "reason"          # the next (costlier) rung to shadow
    assert out["event"].arm == plan_key(decompose) and out["event"].reward == verify.ABSTAIN_FLOOR
