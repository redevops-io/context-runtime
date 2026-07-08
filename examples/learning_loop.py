"""Phase 4 — the asynchronous learning loop that makes the planner scale.

A stateless planner replica selects fast against a LOCAL snapshot and never learns on the serving path.
Executions publish outcome events to a bus; a single aggregator folds them into the shared bandit off
the hot path and republishes a versioned snapshot; replicas reconcile. Same seam works in-process
(shown here) or over Kafka across a fleet — neither stream touches the model's context window.

    python examples/learning_loop.py
"""
from __future__ import annotations

from context_runtime.integrations.bandit import EpsilonGreedyBandit
from context_runtime.learning import InMemoryBus, LearningAggregator, OutcomeEvent
from context_runtime.optimizer.online import BanditOptimizer
from context_runtime.types import Candidate, Goal, PlanScore, StepSpec


def cand(method):
    return Candidate(steps=(StepSpec(type="retrieve", params={"method": method}),), model_tier="cheap")


def score(total):
    return PlanScore(total=total, expected_accuracy=0.9, feasible=True)


def main():
    bus = InMemoryBus()
    goal = Goal(text="how does the auth change relate to the billing outage")
    candidates = [(cand("hybrid"), score(0.80)), (cand("graph"), score(0.55))]  # cost model prefers hybrid

    # A serving replica: selects against its own (initially empty) bandit; ε=0 so it just exploits.
    replica_bandit = EpsilonGreedyBandit(arms=())
    replica = BanditOptimizer(None, bandit=replica_bandit, epsilon=0.0)

    def served(opt):
        return next(s.params["method"] for s in opt.select(candidates, goal, context="multi_hop").chosen.steps
                    if s.type == "retrieve")

    print("Replica serves (before any learning):", served(replica), "— the cost-model favorite\n")

    # Executions happen (somewhere in the fleet) and publish outcome events. graph earns more reward.
    print("Fleet executions publish outcome events to the bus (graph=0.9, hybrid=0.3)...")
    seq = 0
    for _ in range(4):
        for arm, r in (("graph:cheap", 0.9), ("hybrid:cheap", 0.3)):
            seq += 1
            bus.publish("outcomes", OutcomeEvent(context="multi_hop", arm=arm, reward=r, seq=seq))

    # The aggregator (one writer, off the hot path) folds them in and publishes a snapshot.
    aggregator = LearningAggregator(EpsilonGreedyBandit(arms=()))
    n = aggregator.drain(bus)
    snap = aggregator.publish(bus, "snapshots")
    print(f"Aggregator drained {n} events off the serving path → snapshot v{snap.version}\n")

    # The replica reconciles from the published snapshot — it never processed an event itself.
    bus.poll("snapshots")[-1].apply_to(replica_bandit)
    print("Replica serves (after reconciling the snapshot):", served(replica),
          "— adapted past the stale estimate, with zero learning on the hot path")


if __name__ == "__main__":
    main()
