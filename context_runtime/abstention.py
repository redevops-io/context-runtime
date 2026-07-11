"""Honest abstention — Generation 5 (Whitepaper v3, Trust-Aware Execution).

Production AI systems fail when operators stop trusting them, and the fastest way to burn trust is a
confident, wrong answer. So a trust-aware planner must be willing to **decline**: when the best available
plan's calibrated confidence is below the bar, abstain (or escalate to a stronger tier) rather than serve.
Abstaining beats guessing — an honest "I don't know" protects trust, and the TrustLedger credits it.

This gate is optimizer-agnostic: it reads a ``PlanScore`` (the estimator's plan-time confidence, optionally
run through a calibration map so the number means P(correct), not a raw score) and returns a verdict. The
online optimizer records the verdict in ``Plan.extra["abstention"]``; a consumer serves, escalates, or
declines accordingly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .types import Goal, PlanScore


@dataclass(frozen=True)
class AbstentionVerdict:
    action: str          # "serve" | "escalate" | "abstain"
    confidence: float    # calibrated confidence of the evaluated plan, in [0, 1]
    reason: str

    @property
    def abstained(self) -> bool:
        return self.action == "abstain"


class AbstentionGate:
    """Decide whether a plan is confident enough to serve.

    ``min_confidence``  — the bar; below it we do not serve.
    ``calibrate``       — optional map from the raw plan-time confidence to a calibrated P(correct)
                          (e.g. the DSpark ``CalibrationMap``); identity if omitted.
    ``can_escalate``    — optional predicate ``(score, goal) -> bool``: when confidence is below the bar
                          but a stronger option exists, prefer escalation over a flat abstention.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        *,
        calibrate: Callable[[float], float] | None = None,
        can_escalate: Callable[[PlanScore, Goal | None], bool] | None = None,
    ):
        self.min_confidence = min_confidence
        self.calibrate = calibrate
        self.can_escalate = can_escalate

    def confidence(self, score: PlanScore) -> float:
        """Plan-time confidence = the estimator's expected accuracy, calibrated if a map is provided."""
        raw = score.expected_accuracy
        return self.calibrate(raw) if self.calibrate else raw

    def evaluate(self, score: PlanScore, goal: Goal | None = None) -> AbstentionVerdict:
        c = self.confidence(score)
        if c >= self.min_confidence:
            return AbstentionVerdict("serve", c, f"confidence {c:.2f} ≥ {self.min_confidence:.2f}")
        if self.can_escalate is not None and self.can_escalate(score, goal):
            return AbstentionVerdict(
                "escalate", c, f"confidence {c:.2f} < {self.min_confidence:.2f} — escalate to a stronger tier"
            )
        return AbstentionVerdict(
            "abstain", c, f"confidence {c:.2f} < {self.min_confidence:.2f} — abstain (honest, protects trust)"
        )
