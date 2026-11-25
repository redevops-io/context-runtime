"""A candidate policy rejects is never the plan - not as a favourite, not by exploration, and not through the fallback that
produces a plan when nothing meets the cost or latency limits. Found by asking both optimizers with a policy that denies
everything: each returned a denied plan, marked feasible."""
import pytest

from context_runtime.optimizer.knapsack import KnapsackOptimizer
from context_runtime.optimizer.online import BanditOptimizer
from context_runtime.plugins.base import PolicyDenied
from context_runtime.types import Candidate, Constraints, Goal, PlanScore, StepSpec


def _cand(tier):
    return Candidate(steps=(StepSpec(type="retrieve", params={"method": "bm25"}),), model_tier=tier)


class _Deny:
    def __init__(self, *tiers):
        self.tiers = set(tiers)

    def feasible(self, candidate, goal, score):
        return f"{candidate.model_tier} is not allowed" if candidate.model_tier in self.tiers else None


class _Est:
    def estimate(self, candidate, goal):
        return PlanScore(total=0.5, feasible=True)

    def statistics(self, bucket=None):
        raise NotImplementedError

    def observe(self, plan, trace):
        raise NotImplementedError


def _optimizers(policy, **kw):
    return (KnapsackOptimizer(_Est(), policy=policy), BanditOptimizer(_Est(), policy=policy, **kw))


FRONTIER, LOCAL = _cand("frontier"), _cand("local")
SCORED = [(FRONTIER, PlanScore(total=0.9, feasible=True, cost_usd=5.0)), (LOCAL, PlanScore(total=0.2, feasible=True, cost_usd=5.0))]
TOO_CHEAP = Goal(text="q", constraints=Constraints(max_cost_usd=0.01))          # nothing meets this limit


def test_when_policy_rejects_everything_there_is_no_plan():
    for opt in _optimizers(_Deny("frontier", "local")):
        with pytest.raises(PolicyDenied) as e:
            opt.select(SCORED, Goal(text="q"))
        assert len(e.value.rejected) == 2 and "not allowed" in str(e.value)


@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_the_limit_fallback_stays_inside_what_policy_permits(epsilon):
    """Both candidates fail the cost limit, so policy was never asked about them on the way to the fallback."""
    for opt in _optimizers(_Deny("frontier"), epsilon=epsilon):
        for _ in range(25):
            plan = opt.select(SCORED, TOO_CHEAP)
            assert plan.chosen is LOCAL, type(opt).__name__
            assert plan.score.feasible is False                                  # it does not meet the limit, and says so
    for opt in _optimizers(_Deny("frontier", "local"), epsilon=epsilon):
        with pytest.raises(PolicyDenied):
            opt.select(SCORED, TOO_CHEAP)


def test_without_a_policy_a_plan_is_still_produced_when_nothing_meets_the_limits():
    for opt in _optimizers(None, epsilon=0.0):
        plan = opt.select(SCORED, TOO_CHEAP)
        assert plan.chosen in (FRONTIER, LOCAL) and plan.score.feasible is False


def test_with_a_permitted_candidate_inside_the_limits_nothing_changes():
    for opt in _optimizers(_Deny("frontier"), epsilon=0.0):
        plan = opt.select(SCORED, Goal(text="q"))
        assert plan.chosen is LOCAL and plan.score.feasible is True
