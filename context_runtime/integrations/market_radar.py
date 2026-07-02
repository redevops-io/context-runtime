"""market-radar × Context Runtime — tenant for competitive intel sweeps.

Clones social_autopilot line-for-line so Context Runtime can optimize which product
watch feeds (pricing page, changelog, careers, blog) to check per request. The tenant
is fully offline/simulated, ready for benchmarking via ``examples/market_radar.py``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from ..runtime.runtime import ContextRuntime
from ..tools.base import ToolRegistry, ToolResult, function_tool
from ..types import Goal, Trace
from .bandit import EpsilonGreedyBandit


@dataclass(frozen=True)
class RadarArm:
    """One concrete market radar sweep (watch/feed set to check)."""

    pricing_page: bool = False
    changelog: bool = False
    careers: bool = False
    blog: bool = False

    @property
    def key(self) -> str:
        parts: list[str] = []
        if self.pricing_page:
            parts.append("pricing")
        if self.changelog:
            parts.append("product")
        if self.careers:
            parts.append("hiring")
        if self.blog:
            parts.append("news")
        return "+".join(parts) or "idle"

    def cost_units(self) -> float:
        base = 0.8
        if self.pricing_page:
            base += 1.6
        if self.changelog:
            base += 1.4
        if self.careers:
            base += 1.2
        if self.blog:
            base += 1.1
        return base


DEFAULT_RADAR: tuple[RadarArm, ...] = (
    RadarArm(pricing_page=True),
    RadarArm(changelog=True),
    RadarArm(careers=True),
    RadarArm(blog=True),
    RadarArm(pricing_page=True, changelog=True),
    RadarArm(pricing_page=True, careers=True),
    RadarArm(pricing_page=True, blog=True),
    RadarArm(changelog=True, careers=True),
    RadarArm(changelog=True, blog=True),
    RadarArm(careers=True, blog=True),
    RadarArm(pricing_page=True, changelog=True, careers=True),
    RadarArm(pricing_page=True, changelog=True, blog=True),
    RadarArm(pricing_page=True, careers=True, blog=True),
    RadarArm(changelog=True, careers=True, blog=True),
    RadarArm(pricing_page=True, changelog=True, careers=True, blog=True),
)


_BUCKET_KEY = {
    "pricing": "pricing_page",
    "product": "changelog",
    "hiring": "careers",
    "news": "blog",
}


def market_radar_bucket(text: str) -> str:
    lowered = text.lower()
    if "price" in lowered or "pricing" in lowered or "plan" in lowered:
        return "pricing"
    if "release" in lowered or "feature" in lowered or "product" in lowered:
        return "product"
    if "hiring" in lowered or "headcount" in lowered or "job" in lowered:
        return "hiring"
    return "news"


def reward_from_pricing(value: float, arm: RadarArm, cost: float | None = None) -> float:
    fetch_cost = cost if cost is not None else arm.cost_units()
    return value - fetch_cost


def reward_from_product(value: float, arm: RadarArm, cost: float | None = None) -> float:
    fetch_cost = cost if cost is not None else arm.cost_units()
    return value - fetch_cost


def reward_from_hiring(value: float, arm: RadarArm, cost: float | None = None) -> float:
    fetch_cost = cost if cost is not None else arm.cost_units()
    return value - fetch_cost


def reward_from_news(value: float, arm: RadarArm, cost: float | None = None) -> float:
    fetch_cost = cost if cost is not None else arm.cost_units()
    return value - fetch_cost


def _radar_bandit(epsilon: float = 0.15, optimistic: float = 1.0,
                  arms: tuple[RadarArm, ...] = DEFAULT_RADAR) -> EpsilonGreedyBandit:
    return EpsilonGreedyBandit(arms, epsilon=epsilon, optimistic=optimistic)


def _scrape_watch(args: dict) -> ToolResult:
    text = args.get("goal", "")
    data = {
        "pricing_page": bool(args.get("pricing_page")),
        "changelog": bool(args.get("changelog")),
        "careers": bool(args.get("careers")),
        "blog": bool(args.get("blog")),
    }
    summary_bits = [name for name, flag in data.items() if flag]
    summary = ", ".join(summary_bits) if summary_bits else "no sweep"
    return ToolResult(ok=True, text=f"Simulated market radar sweep for '{text}' — {summary}.", data=data)


class MarketRadarTenant:
    """Context Runtime tenant for the market radar sweep."""

    def __init__(self, runtime: ContextRuntime | None = None,
                 arms: tuple[RadarArm, ...] = DEFAULT_RADAR,
                 bandit: EpsilonGreedyBandit | None = None, epsilon: float = 0.15,
                 scrape_tool_factory: Callable[[dict], ToolResult] | None = None):
        self.runtime = runtime or ContextRuntime.default([])
        self.arms = arms
        self.bandit = bandit or _radar_bandit(epsilon=epsilon, arms=arms)
        self.registry = ToolRegistry()
        scrape_fn = scrape_tool_factory or _scrape_watch
        self.registry.register(function_tool(
            name="perform_sweep",
            description="Perform the selected market radar sweep (simulated).",
            fn=scrape_fn,
        ))
        self._pending: dict[str, tuple[Plan, RadarArm, str, ToolResult]] = {}

    def choose(self, goal_text: str, bucket: str | None = None) -> RadarArm:
        plan = self.runtime.plan(Goal(text=goal_text))
        ctx_bucket = bucket or market_radar_bucket(goal_text)
        arm = self.bandit.select(ctx_bucket)
        result = self.registry.run("perform_sweep", {
            "goal": goal_text,
            "pricing_page": arm.pricing_page,
            "changelog": arm.changelog,
            "careers": arm.careers,
            "blog": arm.blog,
        })
        self._pending[self._key(goal_text)] = (plan, arm, ctx_bucket, result)
        return arm

    def record_outcome(self, goal_text: str, value: float, cost: float | None = None) -> float:
        key = self._key(goal_text)
        entry = self._pending.pop(key, None)
        if entry is None:
            return 0.0
        plan, arm, bucket, result = entry
        if bucket == "pricing":
            reward = reward_from_pricing(value, arm, cost)
        elif bucket == "product":
            reward = reward_from_product(value, arm, cost)
        elif bucket == "hiring":
            reward = reward_from_hiring(value, arm, cost)
        else:
            reward = reward_from_news(value, arm, cost)
        self.bandit.update(bucket, arm, reward)
        sweep_cost = (cost if cost is not None else arm.cost_units()) * 0.02
        summary_text = (result.text or "") if isinstance(result, ToolResult) else ""
        self.runtime.estimator.observe(plan, Trace(
            plan_id=plan.id,
            goal_text=goal_text,
            actual_tokens=max(len(summary_text.split()) * 4, 24) if summary_text else 24,
            actual_cost_usd=sweep_cost,
            actual_latency_seconds=0.0,
            verification_passed=value >= (cost if cost is not None else arm.cost_units()),
        ))
        return reward

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()

    @staticmethod
    def _key(goal_text: str) -> str:
        return hashlib.sha256(goal_text.encode()).hexdigest()[:16]
