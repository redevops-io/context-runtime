"""The optimizer's governance seam — an injected PolicyProvider narrows the feasible space before
cost ranking, and a TrustProvider breaks ties. Uses in-file fakes (no enterprise dependency): the
same Protocols the commercial PolicyEngine / TrustLedger adapters satisfy.
"""
from __future__ import annotations

from context_runtime.optimizer.knapsack import KnapsackOptimizer
from context_runtime.types import Candidate, Goal, PlanScore, StepSpec


def _cand(tier: str, method: str = "bm25") -> Candidate:
    return Candidate(steps=(StepSpec(type="retrieve", params={"method": method}),), model_tier=tier)


def _score(total: float) -> PlanScore:
    return PlanScore(total=total, feasible=True)


class _LocalOnlyPolicy:
    """Rejects any candidate that isn't a local model — the canonical 'no external egress' policy."""

    def feasible(self, candidate: Candidate, goal: Goal, score: PlanScore) -> str | None:
        if candidate.model_tier != "local":
            return "local models only"
        return None


class _PreferMethod:
    """Trust score keyed on the retrieval method — used only to break cost ties."""

    def __init__(self, favored: str):
        self.favored = favored

    def score(self, candidate: Candidate, goal: Goal) -> float:
        method = candidate.steps[0].params.get("method")
        return 1.0 if method == self.favored else 0.5


def _estimator():  # minimal CostEstimator returning a fixed score per candidate tier
    class _E:
        def estimate(self, candidate: Candidate, goal: Goal) -> PlanScore:
            return _score(0.9 if candidate.model_tier != "local" else 0.4)
        def statistics(self, bucket=None):  # unused by these tests
            raise NotImplementedError
        def observe(self, plan, trace):
            raise NotImplementedError
    return _E()


def test_policy_narrows_feasible_space_before_cost():
    """The external plan scores higher, but policy makes it infeasible → the local plan is chosen,
    and the external plan appears in rejected with the policy reason (observable in EXPLAIN)."""
    goal = Goal(text="q")
    ext, loc = _cand("frontier"), _cand("local")
    scored = [(ext, _score(0.9)), (loc, _score(0.4))]
    opt = KnapsackOptimizer(_estimator(), policy=_LocalOnlyPolicy())

    plan = opt.select(scored, goal)

    assert plan.chosen is loc, "policy-infeasible high scorer must not win"
    ext_reasons = [r for c, r in plan.rejected if c is ext]
    assert ext_reasons and "policy: local models only" in ext_reasons[0]


def test_policy_marks_score_infeasible():
    goal = Goal(text="q")
    opt = KnapsackOptimizer(_estimator(), policy=_LocalOnlyPolicy())
    assert opt.score(_cand("local"), goal).feasible is True
    assert opt.score(_cand("frontier"), goal).feasible is False


def test_trust_breaks_ties():
    """Two feasible candidates with equal cost total; trust prefers the graph one → it wins."""
    goal = Goal(text="q")
    a = _cand("local", method="bm25")
    b = _cand("local", method="graph")
    scored = [(a, _score(0.5)), (b, _score(0.5))]
    opt = KnapsackOptimizer(_estimator(), trust=_PreferMethod("graph"))
    assert opt.select(scored, goal).chosen is b


def test_no_providers_is_unchanged_behavior():
    """Without policy/trust the optimizer is the pre-seam engine: highest total wins, ties keep order."""
    goal = Goal(text="q")
    hi, lo = _cand("frontier"), _cand("local")
    plan = KnapsackOptimizer(_estimator()).select([(lo, _score(0.4)), (hi, _score(0.9))], goal)
    assert plan.chosen is hi
