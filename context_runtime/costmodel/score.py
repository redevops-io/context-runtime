"""PlanScore — the soft objective the Cost Optimizer maximizes (SPEC §3).

A weighted utility, NOT a bare sum: each term is normalized to [0,1] before weighting
(the terms have different units). Hard constraints are evaluated elsewhere
(``constraints/``); this only ranks the feasible set.
"""
from __future__ import annotations

from dataclasses import replace

from ..types import PlanScore

DEFAULT_WEIGHTS: dict[str, float] = {
    "acc": 1.0, "cache": 0.2, "vrf": 0.5,
    "cost": 0.6, "lat": 0.3, "risk": 0.8, "hall": 0.9, "loss": 0.4,
}

# scales for raw-valued terms (min–max fallback when a candidate set is degenerate)
COST_SCALE_USD = 2.0       # $2 maps to 1.0
LAT_SCALE_S = 60.0         # 60s maps to 1.0


def _n(x: float, scale: float) -> float:
    return max(0.0, min(1.0, x / scale)) if scale else 0.0


def total(score: PlanScore, weights: dict[str, float] | None = None) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    return (
        + w["acc"] * score.expected_accuracy
        + w["cache"] * score.cache_hit_probability
        + w["vrf"] * score.verification_confidence
        - w["cost"] * _n(score.cost_usd, COST_SCALE_USD)
        - w["lat"] * _n(score.latency_seconds, LAT_SCALE_S)
        - w["risk"] * score.risk
        - w["hall"] * score.hallucination_probability
        - w["loss"] * score.context_loss
    )


def finalize(score: PlanScore, weights: dict[str, float] | None = None) -> PlanScore:
    """Return the score with ``total`` filled in from the weighted utility."""
    return replace(score, total=round(total(score, weights), 6))
