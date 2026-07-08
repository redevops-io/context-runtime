"""Phase 4 — the async learning loop: execution → outcome events → aggregator → snapshot → replicas.
Learning happens off the serving path; snapshots distribute it; replays are idempotent."""
from __future__ import annotations

from context_runtime.integrations.bandit import EpsilonGreedyBandit
from context_runtime.learning import (
    InMemoryBus, LearnedStateSnapshot, LearningAggregator, OutcomeEvent,
)
from context_runtime.optimizer.online import BanditOptimizer
from context_runtime.types import Candidate, Goal, PlanScore, StepSpec


def _cand(tier, method):
    return Candidate(steps=(StepSpec(type="retrieve", params={"method": method}),), model_tier=tier)


def _score(total, acc=0.9):
    return PlanScore(total=total, expected_accuracy=acc, feasible=True)


def test_event_from_plan_reads_optimizer_metadata():
    opt = BanditOptimizer(None, epsilon=0.0)
    plan = opt.select([(_cand("cheap", "graph"), _score(0.7))], Goal(text="q"), context="multi_hop")
    ev = OutcomeEvent.from_plan(plan, reward=0.9, seq=1, accepted=True)
    assert ev.context == "multi_hop" and ev.arm == "graph:cheap" and ev.reward == 0.9
    assert ev.accepted is True and OutcomeEvent.from_dict(ev.to_dict()) == ev


def test_aggregator_folds_events_off_the_hot_path():
    bus, bandit = InMemoryBus(), EpsilonGreedyBandit(arms=())
    agg = LearningAggregator(bandit)
    for i, (arm, r) in enumerate([("graph:cheap", 0.9), ("graph:cheap", 0.9), ("hybrid:cheap", 0.2)]):
        bus.publish("outcomes", OutcomeEvent(context="mh", arm=arm, reward=r, seq=i + 1))
    assert agg.drain(bus) == 3 and agg.version == 1
    assert bandit.value("mh", "graph:cheap")[1] > bandit.value("mh", "hybrid:cheap")[1]


def test_snapshot_distributes_learning_to_a_replica():
    bus = InMemoryBus()
    canonical = EpsilonGreedyBandit(arms=())
    agg = LearningAggregator(canonical)
    for i in range(3):
        bus.publish("outcomes", OutcomeEvent(context="mh", arm="graph:cheap", reward=0.9, seq=i + 1))
    agg.drain(bus)
    agg.publish(bus, "snapshots")

    # a stateless replica reconciles from the published snapshot — it never processed an event itself
    replica = EpsilonGreedyBandit(arms=())
    snap = bus.poll("snapshots")[-1]
    LearnedStateSnapshot.from_dict(snap.to_dict()).apply_to(replica)   # survives serialization
    assert replica.value("mh", "graph:cheap") == canonical.value("mh", "graph:cheap")

    # and a replica optimizer now exploits what the aggregator learned, without learning locally
    ropt = BanditOptimizer(None, bandit=replica, epsilon=0.0)
    goal = Goal(text="q")
    plan = ropt.select([(_cand("cheap", "graph"), _score(0.1)), (_cand("cheap", "hybrid"), _score(0.9))],
                       goal, context="mh")
    assert plan.chosen.model_tier == "cheap" and plan.chosen.steps[0].params["method"] == "graph"


def test_replay_is_idempotent_by_seq():
    bandit = EpsilonGreedyBandit(arms=())
    agg = LearningAggregator(bandit)
    ev = OutcomeEvent(context="mh", arm="graph:cheap", reward=0.9, seq=5)
    assert agg.apply(ev) is True
    assert agg.apply(ev) is False           # same seq → ignored
    assert bandit.value("mh", "graph:cheap")[0] == 1   # counted exactly once


def test_trust_sink_receives_every_event_and_abstentions_skip_the_arm():
    seen = []
    bandit = EpsilonGreedyBandit(arms=())
    agg = LearningAggregator(bandit, on_trust=seen.append)
    agg.apply(OutcomeEvent(context="mh", arm="graph:cheap", reward=0.0, seq=1, abstained=True))
    assert len(seen) == 1 and seen[0].abstained
    assert "graph:cheap" not in bandit.stats.get("mh", {})   # abstention rewards no arm; trust still hears it
