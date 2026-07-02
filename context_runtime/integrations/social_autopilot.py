"""social-autopilot × Context Runtime — Postiz-style tenant for channel scheduling.

Maps a social posting goal ("Launch announcement", "Webinar reminder", …) onto the
fleet pattern: the decision point is **which channel/timing/content strategy** to run
for each goal bucket, and the reward is an engagement proxy MINUS the posting cost.
The contextual bandit learns, per goal bucket, which strategy maximizes that reward —
the cheapest strategy that still earns engagement, exactly like the other tenants.

Everything is simulated/offline-ready: strategies are discrete arms (LinkedIn morning
article, Twitter thread, TikTok short, …); a caption-drafting ToolPlugin shows where
post material would be produced; ``record_outcome`` feeds back engagement so the policy
improves. ``examples/social_autopilot.py`` drives a 72-round benchmark proving the
loop learns a better-than-naive schedule.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from ..runtime.runtime import ContextRuntime
from ..tools.base import ToolRegistry, ToolResult, function_tool
from ..types import Goal, Trace
from .bandit import EpsilonGreedyBandit

# ──────────────────────────── strategies (bandit arms) ────────────────────────────


@dataclass(frozen=True)
class SocialStrategy:
    """One concrete channel/timing/content strategy (a bandit arm)."""

    channel: str        # linkedin · twitter · instagram · tiktok
    timing: str         # morning · midday · evening
    format: str         # article · thread · carousel · reel · short
    tone: str = "educational"   # educational · announcement · community · promotional

    @property
    def key(self) -> str:
        return f"{self.channel}:{self.timing}:{self.format}:{self.tone}"

    def cost_units(self) -> float:
        channel_cost = {
            "linkedin": 2.2,
            "twitter": 1.4,
            "instagram": 1.8,
            "tiktok": 2.0,
        }.get(self.channel, 1.6)
        timing_cost = {"morning": 0.9, "midday": 1.1, "evening": 1.0}.get(self.timing, 1.0)
        format_cost = {
            "article": 1.4,
            "thread": 1.1,
            "carousel": 1.3,
            "reel": 1.6,
            "short": 1.5,
        }.get(self.format, 1.2)
        tone_cost = {
            "educational": 0.5,
            "announcement": 0.6,
            "community": 0.4,
            "promotional": 0.7,
        }.get(self.tone, 0.5)
        return channel_cost + timing_cost + format_cost + tone_cost


DEFAULT_STRATEGIES: tuple[SocialStrategy, ...] = (
    SocialStrategy("linkedin", "morning", "article", "educational"),
    SocialStrategy("linkedin", "midday", "carousel", "announcement"),
    SocialStrategy("twitter", "midday", "thread", "announcement"),
    SocialStrategy("instagram", "evening", "reel", "community"),
    SocialStrategy("tiktok", "evening", "short", "community"),
    SocialStrategy("twitter", "morning", "thread", "educational"),
)


# ──────────────────────────── helpers: bucket, reward, bandit ────────────────────────────


def social_bucket(goal: str) -> str:
    """Classify a goal into a coarse bucket (launch · education · community · promo)."""
    g = goal.lower()
    if any(k in g for k in ("launch", "announce", "release", "ship")):
        return "launch"
    if any(k in g for k in ("webinar", "event", "meetup", "community", "customer")):
        return "community"
    if any(k in g for k in ("tutorial", "guide", "how to", "deep dive", "insight")):
        return "education"
    if any(k in g for k in ("discount", "promo", "offer", "sale", "upgrade")):
        return "promotion"
    return "evergreen"


def reward_from_engagement(engagement: float, strategy: SocialStrategy,
                           cost: float | None = None) -> float:
    """Efficiency frontier: reward = engagement − posting_cost (higher = better & cheaper)."""
    posting_cost = cost if cost is not None else strategy.cost_units()
    return engagement - posting_cost


def _social_bandit(epsilon: float = 0.15, optimistic: float = 1.0,
                   strategies: tuple[SocialStrategy, ...] = DEFAULT_STRATEGIES) -> EpsilonGreedyBandit:
    return EpsilonGreedyBandit(strategies, epsilon=epsilon, optimistic=optimistic)


# ──────────────────────────── tool (caption drafting, simulated) ────────────────────────────


def _draft_caption(args: dict) -> ToolResult:
    goal = args.get("goal", "")
    channel = args.get("channel", "channel")
    tone = args.get("tone", "tone").title()
    format_ = args.get("format", "post")
    hashtags = {
        "linkedin": "#B2B #SaaS",
        "twitter": "#Startups #Product",
        "instagram": "#Community #BehindTheScenes",
        "tiktok": "#ProductTips #Automation",
    }
    tag_line = hashtags.get(channel, "#ContextRuntime")
    text = f"{goal} — {tone} take for {channel.title()} {format_}. {tag_line}".strip()
    return ToolResult(ok=True, text=text, data={"channel": channel, "tone": tone.lower(), "format": format_})


# ──────────────────────────── tenant ────────────────────────────


class SocialAutopilotTenant:
    """Context Runtime tenant for Postiz/social autopilot scheduling."""

    def __init__(self, runtime: ContextRuntime | None = None,
                 strategies: tuple[SocialStrategy, ...] = DEFAULT_STRATEGIES,
                 bandit: EpsilonGreedyBandit | None = None, epsilon: float = 0.15,
                 caption_tool_factory: Callable[[dict], ToolResult] | None = None):
        self.runtime = runtime or ContextRuntime.default([])
        self.strategies = strategies
        self.bandit = bandit or _social_bandit(epsilon=epsilon, strategies=strategies)
        self.registry = ToolRegistry()
        caption_fn = caption_tool_factory or _draft_caption
        self.registry.register(function_tool(
            name="draft_caption",
            description="Draft a caption for the chosen social strategy (simulated).",
            fn=caption_fn,
        ))
        self._pending: dict[str, tuple[Plan, SocialStrategy, str, str]] = {}

    def choose(self, goal_text: str, bucket: str | None = None) -> SocialStrategy:
        """Choose a strategy for the goal (and draft copy via the tool)."""
        plan = self.runtime.plan(Goal(text=goal_text))
        ctx_bucket = bucket or social_bucket(goal_text)
        strategy = self.bandit.select(ctx_bucket)
        caption = self.registry.run("draft_caption", {
            "goal": goal_text,
            "channel": strategy.channel,
            "tone": strategy.tone,
            "format": strategy.format,
        }).text or ""
        self._pending[self._key(goal_text)] = (plan, strategy, ctx_bucket, caption)
        return strategy

    def record_outcome(self, goal_text: str, engagement: float, cost: float | None = None) -> float:
        key = self._key(goal_text)
        entry = self._pending.pop(key, None)
        if entry is None:
            return 0.0
        plan, strategy, bucket, caption = entry
        reward = reward_from_engagement(engagement, strategy, cost)
        self.bandit.update(bucket, strategy, reward)
        token_estimate = max(len(caption.split()) * 4, 24) if caption else 24
        actual_cost_usd = (cost if cost is not None else strategy.cost_units()) * 0.02
        self.runtime.estimator.observe(plan, Trace(
            plan_id=plan.id,
            goal_text=goal_text,
            actual_tokens=token_estimate,
            actual_cost_usd=actual_cost_usd,
            actual_latency_seconds=0.0,
            verification_passed=engagement >= (cost if cost is not None else strategy.cost_units()),
        ))
        return reward

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()

    @staticmethod
    def _key(goal_text: str) -> str:
        return hashlib.sha256(goal_text.encode()).hexdigest()[:16]
