"""Generation-4 online-learning optimizer: contextual-bandit selection with exploration, learning
from reward, off-policy evaluation, and composition with the Phase-0 governance seam.
"""
from __future__ import annotations

from context_runtime.optimizer.online import BanditOptimizer, offpolicy_values, plan_key
from context_runtime.types import Candidate, Goal, PlanScore, StepSpec


def _cand(tier: str, method: str = "bm25") -> Candidate:
    return Candidate(steps=(StepSpec(type="retrieve", params={"method": method}),), model_tier=tier)


def _score(total: float) -> PlanScore:
    return PlanScore(total=total, feasible=True)


class _Est:  # unused by select(); present so BanditOptimizer(estimator=...) is realistic
    def estimate(self, c, g): return _score(0.5)
    def statistics(self, bucket=None): raise NotImplementedError
    def observe(self, p, t): raise NotImplementedError


def test_exploit_picks_highest_prior_when_epsilon_zero():
    opt = BanditOptimizer(_Est(), epsilon=0.0)
    goal = Goal(text="q")
    hi, lo = _cand("premium", "hybrid"), _cand("local", "bm25")
    plan = opt.select([(lo, _score(0.4)), (hi, _score(0.9))], goal, context="synthesis")
    assert plan.chosen is hi
    assert plan.extra["bandit"]["mode"] == "exploit"
    assert plan.extra["bandit"]["p"] == 1.0          # greedy under ε=0 has propensity 1


def test_exploration_serves_a_non_greedy_arm():
    opt = BanditOptimizer(_Est(), epsilon=1.0)        # always explore
    goal = Goal(text="q")
    hi, lo = _cand("premium", "hybrid"), _cand("local", "bm25")
    plan = opt.select([(hi, _score(0.9)), (lo, _score(0.4))], goal, context="synthesis")
    assert plan.chosen is lo                          # the non-greedy arm
    assert plan.extra["bandit"]["mode"] == "explore"
    assert plan.extra["bandit"]["p"] == 0.5           # ε/m with m=2 arms


def test_learning_shifts_the_exploit_choice():
    opt = BanditOptimizer(_Est(), epsilon=0.0)
    goal = Goal(text="q")
    a, b = _cand("premium", "hybrid"), _cand("local", "graph")   # a prior 0.9 > b prior 0.4
    scored = [(a, _score(0.9)), (b, _score(0.4))]
    assert opt.select(scored, goal, context="multi_hop").chosen is a   # prior wins first

    for _ in range(3):                                # b turns out to be worth 1.0 in practice
        opt.learn("multi_hop", b, 1.0)
    assert opt.select(scored, goal, context="multi_hop").chosen is b   # learned value beats a's prior


def test_learn_from_plan_uses_recorded_context_and_arm():
    opt = BanditOptimizer(_Est(), epsilon=0.0)
    goal = Goal(text="q")
    b = _cand("local", "graph")
    plan = opt.select([(_cand("premium", "hybrid"), _score(0.9)), (b, _score(0.4))], goal, context="mh")
    # feed reward back for the *served* arm via the plan's recorded (context, arm)
    for _ in range(3):
        opt.learn_from_plan(
            opt.select([(b, _score(0.4))], goal, context="mh"), 1.0
        )
    n, mean = opt.bandit.value("mh", plan_key(b))
    assert n == 3 and mean == 1.0


class _LocalOnly:
    def feasible(self, cand, goal, score):
        return None if cand.model_tier == "local" else "local models only"


def test_policy_filters_before_any_exploration():
    """Even with ε=1 (always explore), a policy-infeasible arm is never in the pool → never served."""
    opt = BanditOptimizer(_Est(), epsilon=1.0, policy=_LocalOnly())
    goal = Goal(text="q")
    ext, loc = _cand("premium", "hybrid"), _cand("local", "bm25")
    plan = opt.select([(ext, _score(0.9)), (loc, _score(0.4))], goal, context="synthesis")
    assert plan.chosen is loc
    assert any(c is ext and r.startswith("policy:") for c, r in plan.rejected)


def test_offpolicy_evaluation_surfaces_a_rare_but_better_arm():
    """Tier 1: a high-reward arm that was rarely served (low p) still scores highest, because IPS
    up-weights it — so the planner can prefer it without ever running it on the live path."""
    logs = [
        {"context": "mh", "arm": "hybrid:cheap", "reward": 0.2, "p": 0.9},   # served often, mediocre
        {"context": "mh", "arm": "hybrid:cheap", "reward": 0.3, "p": 0.9},
        {"context": "mh", "arm": "graph:cheap", "reward": 0.9, "p": 0.1},    # served rarely, great
    ]
    vals = offpolicy_values(logs)["mh"]
    assert vals["graph:cheap"][1] > vals["hybrid:cheap"][1]
    assert abs(vals["hybrid:cheap"][1] - 0.25) < 1e-9   # self-normalized IPS = mean reward here
