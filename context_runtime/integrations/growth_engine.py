# SPDX-License-Identifier: AGPL-3.0-or-later
"""growth-engine × Context Runtime — marketing attribution tenant."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from .bandit import EpsilonGreedyBandit
from ..runtime.runtime import ContextRuntime
from ..tools.base import ToolRegistry, ToolResult, function_tool
from ..types import Goal, Plan, Trace


# ──────────────────────────── attribution arms ────────────────────────────


@dataclass(frozen=True)
class AttributionArm:
    """One attribution window + source bundle (a bandit arm)."""

    window: str
    decisive: str
    ancillary: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        ancillary = ",".join(self.ancillary)
        return f"{self.window}:{self.decisive}:{ancillary}"

    def cost_units(self) -> float:
        window_cost = {"24h": 1.6, "7d": 2.1, "30d": 2.9, "session": 1.2}.get(self.window, 2.0)
        decisive_cost = {
            "utm": 0.8,
            "referrer": 0.7,
            "session": 0.6,
            "first_touch": 1.1,
        }.get(self.decisive, 0.9)
        ancillary_cost = sum(0.4 for _ in self.ancillary)
        return window_cost + decisive_cost + ancillary_cost


DEFAULT_GROWTH: tuple[AttributionArm, ...] = (
    AttributionArm("24h", "utm", ("session",)),
    AttributionArm("24h", "referrer", ("utm",)),
    AttributionArm("7d", "utm", ("first_touch",)),
    AttributionArm("7d", "referrer", ("session",)),
    AttributionArm("30d", "first_touch", ("utm", "referrer")),
    AttributionArm("session", "session", ("utm",)),
)


# ──────────────────────────── helpers: bucket + reward ────────────────────────────


def growth_engine_bucket(text: str) -> str:
    q = text.lower()
    if any(k in q for k in ("paid", "campaign", "cpc", "ad", "utm")):
        return "paid"
    if any(k in q for k in ("community", "partner", "affiliate", "referral")):
        return "referral"
    if any(k in q for k in ("newsletter", "blog", "seo", "organic", "search")):
        return "organic"
    return "direct"


def reward_from_paid(value: float, arm: AttributionArm, cost: float | None = None) -> float:
    return value - (cost if cost is not None else arm.cost_units())


def reward_from_organic(value: float, arm: AttributionArm, cost: float | None = None) -> float:
    return value - (cost if cost is not None else arm.cost_units())


def reward_from_referral(value: float, arm: AttributionArm, cost: float | None = None) -> float:
    return value - (cost if cost is not None else arm.cost_units())


def reward_from_direct(value: float, arm: AttributionArm, cost: float | None = None) -> float:
    return value - (cost if cost is not None else arm.cost_units())


_REWARD_FN = {
    "paid": reward_from_paid,
    "organic": reward_from_organic,
    "referral": reward_from_referral,
    "direct": reward_from_direct,
}


def _growth_bandit(epsilon: float = 0.15, optimistic: float = 1.0,
                   arms: tuple[AttributionArm, ...] = DEFAULT_GROWTH) -> EpsilonGreedyBandit:
    return EpsilonGreedyBandit(arms, epsilon=epsilon, optimistic=optimistic)


# ──────────────────────────── tool (attribution report, simulated) ────────────────────────────


def _draft_attribution_report(args: dict) -> ToolResult:
    question = args.get("goal", "")
    bucket = args.get("bucket", "bucket")
    window = args.get("window", "window")
    decisive = args.get("decisive", "decisive")
    ancillary = args.get("ancillary", ())
    if not isinstance(ancillary, (list, tuple)):
        ancillary = (str(ancillary),) if ancillary else ()
    ancillary_text = ", ".join(ancillary) if ancillary else "none"
    text = (
        f"{question} — attribution bucket {bucket} via {decisive} signal "
        f"with a {window} window (ancillary: {ancillary_text})."
    ).strip()
    return ToolResult(ok=True, text=text, data={
        "bucket": bucket,
        "window": window,
        "decisive": decisive,
        "ancillary": tuple(ancillary),
    })


# ──────────────────────────── tenant ────────────────────────────


class GrowthEngineTenant:
    """Context Runtime tenant for growth-engine attribution decisions."""

    def __init__(self, runtime: ContextRuntime | None = None,
                 arms: tuple[AttributionArm, ...] = DEFAULT_GROWTH,
                 bandit: EpsilonGreedyBandit | None = None, epsilon: float = 0.15,
                 bucket_fn: Callable[[str], str] = growth_engine_bucket,
                 report_tool_factory: Callable[[dict], ToolResult] | None = None):
        self.runtime = runtime or ContextRuntime.default([])
        self.arms = arms
        self.bandit = bandit or _growth_bandit(epsilon=epsilon, arms=arms)
        self.bucket_fn = bucket_fn
        self.registry = ToolRegistry()
        report_fn = report_tool_factory or _draft_attribution_report
        self.registry.register(function_tool(
            name="draft_attribution_report",
            description="Draft an attribution breakdown for the chosen arm (simulated).",
            fn=report_fn,
        ))
        self._pending: dict[str, tuple[Plan, AttributionArm, str, str]] = {}

    def choose(self, attribution_question: str, bucket: str | None = None) -> AttributionArm:
        """Choose an attribution arm for the question (and draft a report via the tool)."""
        plan = self.runtime.plan(Goal(text=attribution_question))
        ctx_bucket = bucket or self.bucket_fn(attribution_question)
        arm = self.bandit.select(ctx_bucket)
        report = self.registry.run("draft_attribution_report", {
            "goal": attribution_question,
            "bucket": ctx_bucket,
            "window": arm.window,
            "decisive": arm.decisive,
            "ancillary": arm.ancillary,
        }).text or ""
        self._pending[self._key(attribution_question)] = (plan, arm, ctx_bucket, report)
        return arm

    def record_outcome(self, attribution_question: str, value: float, cost: float | None = None) -> float:
        key = self._key(attribution_question)
        entry = self._pending.pop(key, None)
        if entry is None:
            return 0.0
        plan, arm, bucket, report = entry
        reward_fn = _REWARD_FN.get(bucket, reward_from_direct)
        reward = reward_fn(value, arm, cost)
        self.bandit.update(bucket, arm, reward)
        tokens = max(len(report.split()) * 4, 32) if report else 32
        actual_cost_units = cost if cost is not None else arm.cost_units()
        actual_cost_usd = actual_cost_units * 0.015
        self.runtime.estimator.observe(plan, Trace(
            plan_id=plan.id,
            goal_text=attribution_question,
            actual_tokens=tokens,
            actual_cost_usd=actual_cost_usd,
            actual_latency_seconds=0.0,
            verification_passed=value >= actual_cost_units,
        ))
        return reward

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
