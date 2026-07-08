"""The Context-Runtime decision layer — an intent-keyed bandit over retrieval configs.

This is the actual Context Runtime mechanism (planner intent → per-bucket
``EpsilonGreedyBandit`` over ``RetrievalConfig`` arms → cost-aware reward), the same
shape as ``examples/rag_tuning.py`` — just applied to the harness's lean BM25 backend
instead of ``redevops_rag.hybrid_search`` so the benchmark stays torch-free. The bandit
learns ONLINE across the question stream: cheaper gating configs win on clean corpora,
thorough/gated configs win as pollution rises.
"""
from __future__ import annotations

import re

from context_runtime.integrations.bandit import EpsilonGreedyBandit
from context_runtime.integrations.redevops_rag import (
    DEFAULT_ARMS, RetrievalConfig, reward_from_quality,
)

# Map FinanceBench question flavor → a coarse intent bucket the bandit keys on.
_METRIC_HINTS = re.compile(r"\b(fy\d{2,4}|capex|capital expenditure|revenue|margin|ratio|"
                           r"eps|amount|how much|what is the|total|net|cash flow)\b", re.I)


def bucket_for(question) -> str:
    qt = getattr(question, "qtype", "")
    if qt == "metrics-generated" or _METRIC_HINTS.search(question.question):
        return "exact_lookup"
    if qt == "novel-generated":
        return "synthesis"
    return "conceptual"


class ContextRuntimeTuner:
    """Wraps the real ``EpsilonGreedyBandit`` over ``DEFAULT_ARMS``; ``choose`` selects a
    config per intent bucket, ``record`` feeds cost-aware reward back."""

    def __init__(self, arms=DEFAULT_ARMS, *, epsilon: float = 0.15,
                 discount: float = 0.0, persist_path: str | None = None):
        self.arms = {a.key: a for a in arms}
        self.bandit = EpsilonGreedyBandit(arms, epsilon=epsilon, discount=discount,
                                          persist_path=persist_path)

    def choose(self, question) -> tuple:
        bucket = bucket_for(question)
        arm = self.bandit.select(bucket)
        cfg = arm if isinstance(arm, RetrievalConfig) else self.arms[arm.key]
        return cfg, bucket

    def record(self, question, cfg: RetrievalConfig, quality: float) -> None:
        """quality in 0..1 (e.g. answer-correct → 1.0, retrieval MRR, or a blend)."""
        self.bandit.update(bucket_for(question), cfg, reward_from_quality(quality, cfg))

    def policy(self) -> dict:
        return self.bandit.policy()
