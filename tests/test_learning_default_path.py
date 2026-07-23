"""Item 6: the online optimizer + reward loop on the DEFAULT serving path (opt-in via learning=True).

The audit's finding was that the learning machinery existed but the runtime defaulted to the static
knapsack optimizer and never called learn(). Here: learning=True selects the BanditOptimizer, and
execute() folds a shared-contract reward back so the bandit actually updates.
"""
from context_runtime import ContextRuntime
from context_runtime.adapters.model_stub import StubModel
from context_runtime.adapters.store_inmemory import InMemoryStore
from context_runtime.learning.reward import DefaultReward
from context_runtime.optimizer.online import BanditOptimizer, plan_key


class FakeTrace:
    def __init__(self, cost, verified, citations):
        self.actual_cost_usd = cost
        self.verified = verified
        self.citations = citations


def test_default_reward_quality_and_cost():
    r = DefaultReward()
    # grounded + verified, ~free → near the quality ceiling
    hi = r.reward(FakeTrace(0.0, True, ("c1",)), None)
    assert hi == 1.0
    # ungrounded, unverified → just the base
    lo = r.reward(FakeTrace(0.0, False, ()), None)
    assert abs(lo - 0.5) < 1e-9
    # cost drags reward down
    costly = r.reward(FakeTrace(0.05, True, ("c1",)), None)   # full cost penalty (lam=0.3)
    assert abs(costly - (1.0 - 0.3)) < 1e-9


def test_verdict_overrides_trace_flag():
    r = DefaultReward()
    class V:  # a failing verdict beats a truthy trace.verified
        passed = False
    assert r.reward(FakeTrace(0.0, True, ("c1",)), V()) < 1.0


def _rt(learning):
    return ContextRuntime(
        models={t: StubModel(tier=t) for t in ("local", "cheap", "premium")},
        retriever=InMemoryStore([{"chunk_id": "d1", "filename": "d1",
                                  "text": "reciprocal rank fusion blends rankings", "created_at": None}]),
        learning=learning)


def test_learning_flag_selects_bandit_optimizer():
    assert isinstance(_rt(True).optimizer, BanditOptimizer)
    # default stays static
    from context_runtime.optimizer.knapsack import KnapsackOptimizer
    assert isinstance(_rt(False).optimizer, KnapsackOptimizer)


def test_default_path_learns_reward_folds_into_bandit():
    rt = _rt(True)
    q = "What is reciprocal rank fusion?"
    # before any run, no arm observed
    res = rt.run(q)
    arm = res.plan.extra["bandit"]["arm"]
    ctx = res.plan.extra["bandit"]["context"]
    n_after_1, _ = rt.optimizer.bandit.value(ctx, arm)
    assert n_after_1 >= 1                       # execute() folded a reward in
    rt.run(q); rt.run(q)
    n_after_3, mean = rt.optimizer.bandit.value(ctx, arm)
    assert n_after_3 > n_after_1                 # keeps learning across runs
    assert 0.0 <= mean <= 1.0                    # bounded reward contract


def test_learning_off_does_not_learn():
    rt = _rt(False)
    rt.run("What is reciprocal rank fusion?")
    assert not hasattr(rt.optimizer, "bandit")   # static optimizer, nothing to learn into
