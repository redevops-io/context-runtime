"""Through the whole runtime, with canaries on retrieval and every model: what policy rejects is never executed - not as the
selection, not through the limit fallback, not replayed from the plan cache, not from a cache two identities share. When
policy permits nothing, nothing runs at all."""
import pytest

from context_runtime.costmodel.estimators import HeuristicEstimator
from context_runtime.optimizer.knapsack import KnapsackOptimizer
from context_runtime.optimizer.online import BanditOptimizer
from context_runtime.plancache.cache import SnapshotPlanCache
from context_runtime.plugins.base import PolicyDenied
from context_runtime.runtime.runtime import ContextRuntime
from context_runtime.types import Constraints, Hit, ModelResult

QUESTION = "what is our refund policy"


class _Calls:
    def __init__(self):
        self.retrieved, self.generated = 0, []


class _Retriever:
    def __init__(self, calls):
        self.calls = calls

    def search(self, query, k=5, method="hybrid"):
        self.calls.retrieved += 1
        return [Hit(chunk_id="c1", filename="kb.md", text="refunds are accepted for 30 days", score=1.0)]

    def index(self, path):
        return {}


class _Model:
    def __init__(self, tier, calls):
        self.tier, self.calls = tier, calls

    def complete(self, req):
        self.calls.generated.append(self.tier)
        return ModelResult(text="30 days [c1]", model="stub", tier=self.tier)

    def count_tokens(self, text, model):
        return len(text) // 4


class _Models(dict):
    def __init__(self, calls):
        super().__init__()
        self.calls = calls

    def get(self, tier, default=None):
        return self.setdefault(tier, _Model(tier, self.calls))

    def __getitem__(self, tier):
        return self.get(tier)

    def __bool__(self):
        return True


class _Policy:
    def __init__(self):
        self.denied: set = set()
        self.deny_all = False

    def feasible(self, candidate, goal, score):
        return "revoked" if self.deny_all or candidate.model_tier in self.denied else None


def _runtime(policy, kind, **kw):
    calls = _Calls()
    est = HeuristicEstimator()
    if kind == "knapsack":
        rt = ContextRuntime(models=_Models(calls), retriever=_Retriever(calls), estimator=est,
                            optimizer=KnapsackOptimizer(est, policy=policy), **kw)
    else:                                                      # the learning path: no plan cache unless one is injected
        rt = ContextRuntime(models=_Models(calls), retriever=_Retriever(calls), estimator=est, learning=True,
                            optimizer=BanditOptimizer(est, policy=policy, epsilon=1.0), **kw)
    return rt, calls


KINDS = ("knapsack", "bandit")


@pytest.mark.parametrize("kind", KINDS)
def test_when_policy_permits_nothing_nothing_is_retrieved_or_generated(kind):
    policy = _Policy()
    policy.deny_all = True
    rt, calls = _runtime(policy, kind)
    for entry in (lambda: rt.run(QUESTION), lambda: rt.plan(QUESTION), lambda: rt.explain(QUESTION, analyze=True),
                  lambda: rt.explain(QUESTION), lambda: rt.simulate(QUESTION)):
        with pytest.raises(PolicyDenied):
            entry()
    assert calls.retrieved == 0 and calls.generated == []


@pytest.mark.parametrize("kind", KINDS)
def test_a_plan_permitted_a_moment_ago_is_not_replayed_after_policy_revokes_it(kind):
    """The default runtime caches plans. The cached plan was served after policy revoked it."""
    policy = _Policy()
    rt, calls = _runtime(policy, kind, plan_cache=SnapshotPlanCache())
    first = rt.run(QUESTION)
    assert first.plan.chosen.model_tier in calls.generated
    for _ in range(4):
        revoked = rt.plan(QUESTION).chosen.model_tier
        policy.denied.add(revoked)
        calls.generated.clear()
        try:
            again = rt.run(QUESTION)
        except PolicyDenied:
            break
        assert again.plan.chosen.model_tier not in policy.denied and not set(calls.generated) & policy.denied, (revoked, calls.generated)
    policy.deny_all = True
    calls.retrieved, calls.generated = 0, []
    with pytest.raises(PolicyDenied):
        rt.run(QUESTION)                                                 # a cached plan exists, and is not an answer
    assert calls.retrieved == 0 and calls.generated == []


def test_two_identities_that_share_a_cache_and_a_key_do_not_share_permission():
    shared = SnapshotPlanCache()
    open_policy, strict = _Policy(), _Policy()
    a, _ = _runtime(open_policy, "knapsack", plan_cache=shared)
    served = a.run(QUESTION).plan.chosen.model_tier
    strict.denied.add(served)
    b, calls = _runtime(strict, "knapsack", plan_cache=shared)
    out = b.run(QUESTION)
    assert out.plan.chosen.model_tier != served and served not in calls.generated and out.plan.cache != "hit"
    c, calls_c = _runtime(_Policy(), "knapsack", plan_cache=shared)
    strict_all = _Policy()
    strict_all.deny_all = True
    d, calls_d = _runtime(strict_all, "knapsack", plan_cache=shared)
    with pytest.raises(PolicyDenied):
        d.run(QUESTION)
    assert calls_d.retrieved == 0 and calls_d.generated == []


@pytest.mark.parametrize("kind", KINDS)
def test_the_limit_fallback_never_executes_what_policy_rejects(kind):
    policy = _Policy()
    rt, calls = _runtime(policy, kind)
    tiers = {c.model_tier for c in rt.candidates.generate(rt.intent.analyze(rt._coerce_goal(QUESTION, None, None)), rt._coerce_goal(QUESTION, None, None))}
    keep = sorted(tiers)[0]
    policy.denied = tiers - {keep}
    nothing_fits = Constraints(max_cost_usd=0.0000001, max_latency_seconds=0.0000001)
    for _ in range(12):
        out = rt.run(QUESTION, constraints=nothing_fits)
        assert out.plan.chosen.model_tier == keep and set(calls.generated) <= {keep}
    policy.deny_all = True
    calls.retrieved, calls.generated = 0, []
    with pytest.raises(PolicyDenied):
        rt.run(QUESTION, constraints=nothing_fits)
    assert calls.retrieved == 0 and calls.generated == []


def test_without_a_policy_the_cache_still_replays():
    rt, _ = _runtime(None, "knapsack")
    a, b = rt.plan(QUESTION), rt.plan(QUESTION)
    assert a.cache != "hit" and b.cache == "hit" and a.chosen == b.chosen
