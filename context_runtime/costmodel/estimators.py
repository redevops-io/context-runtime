"""Cost estimators (SPEC §3, §3.1). v0.1 = heuristic; path → learned → neural.

``estimate()`` predicts the PlanScore fields for a candidate; ``observe()`` records
estimate-vs-actual into the statistics store (the trust layer); ``statistics()``
exposes calibration. The learner that *consumes* observations to improve estimates is
v0.3 — v0.1 just collects honestly.
"""
from __future__ import annotations

from pathlib import Path

from ..planner import rules
from ..types import Candidate, CostModelStatistics, Goal, Plan, PlanScore, Trace
from . import score as score_mod
from .statistics import StatisticsStore

# rough per-tier cost/latency priors ($ per call, seconds per call)
TIER_COST = {"local": 0.0, "cheap": 0.05, "premium": 0.4}
TIER_LATENCY = {"local": 4.0, "cheap": 8.0, "premium": 18.0}
# rough per-tier quality priors
TIER_ACCURACY = {"local": 0.62, "cheap": 0.78, "premium": 0.9}
TIER_HALLUCINATION = {"local": 0.18, "cheap": 0.1, "premium": 0.05}

METHOD_RECALL = {"bm25": 0.6, "vector": 0.7, "hybrid": 0.85, "code": 0.8, "graph": 0.75}


class HeuristicEstimator:
    version = "heuristic-0.1"

    def __init__(self, stats_path: str | Path | None = None, profile=None):
        self.stats = StatisticsStore(
            estimator_version=self.version,
            path=Path(stats_path) if stats_path else None,
        )
        # Optional CostProfile: measured stage latency replacing the TIER_LATENCY prior
        # when a cell has samples (DSpark's profiled cost table). None ⇒ pure heuristic.
        self.profile = profile

    def estimate(self, candidate: Candidate, goal: Goal) -> PlanScore:
        tier = candidate.model_tier
        steps = {s.type: s.params for s in candidate.steps}

        method = steps.get("retrieve", {}).get("method", "hybrid")
        recall = METHOD_RECALL.get(method, 0.7)
        reranked = "rerank" in steps
        verified = "verify" in steps

        # Intent-aware retrieval: for a multi-hop question the answer lives in the
        # connections between documents — graph retrieval reaches it, single-hop
        # structurally cannot. For a single-hop question, graph is just pricier with no
        # recall edge, so hybrid wins. This is how the planner routes HippoRAG vs redevops-rag.
        bucket, _risk = rules.classify(goal.text)
        if bucket == "multi_hop":
            recall = 0.93 if method == "graph" else recall * 0.55
        is_graph = method == "graph"

        base_acc = TIER_ACCURACY.get(tier, 0.7)
        accuracy = min(0.99, base_acc * (0.7 + 0.3 * recall) + (0.04 if reranked else 0.0))

        # graph retrieval pays for the KG build + Personalized PageRank hop
        cost = TIER_COST.get(tier, 0.1) + (0.01 if reranked else 0.0) + (0.03 if is_graph else 0.0)
        # Base model latency: measured (profiled) when available, else the tier prior.
        base_lat = TIER_LATENCY.get(tier, 8.0)
        if self.profile is not None:
            measured = self.profile.latency(f"synthesis:{tier}", 1)
            if measured is not None:
                base_lat = measured
        latency = (base_lat + (2.0 if reranked else 0.0)
                   + (3.0 if verified else 0.0) + (5.0 if is_graph else 0.0))

        hallucination = TIER_HALLUCINATION.get(tier, 0.12) * (0.5 if verified else 1.0)
        risk = {"low": 0.1, "medium": 0.3, "high": 0.6}.get(goal.constraints.sensitivity, 0.1)
        context_loss = 0.0 if "compress" not in steps else 0.15
        verify_conf = 0.8 if verified else 0.0
        cache_hit = 0.0   # Plan Cache is v0.2

        raw = PlanScore(
            expected_accuracy=round(accuracy, 4),
            cache_hit_probability=cache_hit,
            verification_confidence=verify_conf,
            cost_usd=round(cost, 4),
            latency_seconds=round(latency, 2),
            risk=risk,
            hallucination_probability=round(hallucination, 4),
            context_loss=context_loss,
        )
        return score_mod.finalize(raw, goal.constraints.weight_overrides)

    def observe(self, plan: Plan, trace: Trace) -> None:
        est = plan.score
        self.stats.observe(
            estimates={
                "cost_usd": est.cost_usd,
                "latency_seconds": est.latency_seconds,
                "expected_accuracy": est.expected_accuracy,
            },
            actuals={
                "cost_usd": trace.actual_cost_usd,
                "latency_seconds": trace.actual_latency_seconds,
                # proxy: verification pass ≈ accuracy signal we can observe cheaply
                "expected_accuracy": 1.0 if trace.verification_passed else (
                    est.expected_accuracy if trace.verification_passed is None else 0.0
                ),
            },
        )

    def statistics(self, bucket: str | None = None) -> CostModelStatistics:
        return self.stats.snapshot(bucket)
