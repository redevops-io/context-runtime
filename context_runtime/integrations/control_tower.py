"""control-tower × Context Runtime — Metabase query tuner tenant.

Simulates the internal control tower dashboard problem: each revenue/pipeline/ops/cash
bucket maps to a *Metabase query set* (which dashboards/questions to refresh). The
bandit learns which query set gives the best delta minus compute cost per bucket,
mirroring the social-autopilot tenant, but for BI workloads.

Offline harness: ``examples/control_tower.py`` runs 72 rounds of simulated decisions
and shows the learned policy outperforming a fixed baseline arm.
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
class ControlTowerArm:
    """One dashboard query set (bandit arm)."""

    name: str
    compute_minutes: float

    @property
    def key(self) -> str:
        return self.name

    def cost_units(self) -> float:
        return self.compute_minutes


DEFAULT_TOWER: tuple[ControlTowerArm, ...] = (
    ControlTowerArm("daily_revenue_core", 7.0),
    ControlTowerArm("growth_pipeline_full", 8.5),
    ControlTowerArm("ops_latency_focus", 5.5),
    ControlTowerArm("cashflow_variance", 6.2),
)


def control_tower_bucket(text: str) -> str:
    lower = text.lower()
    if "pipeline" in lower or "lead" in lower or "mql" in lower:
        return "pipeline"
    if "latency" in lower or "oncall" in lower or "escalation" in lower:
        return "ops"
    if "cash" in lower or "burn" in lower or "runway" in lower:
        return "cash"
    return "revenue"


def reward_from_revenue(value: float, arm: ControlTowerArm, cost: float | None = None) -> float:
    compute = cost if cost is not None else arm.cost_units()
    return value - compute


def reward_from_pipeline(value: float, arm: ControlTowerArm, cost: float | None = None) -> float:
    compute = cost if cost is not None else arm.cost_units()
    return value - compute


def reward_from_ops(value: float, arm: ControlTowerArm, cost: float | None = None) -> float:
    compute = cost if cost is not None else arm.cost_units()
    return value - compute


def reward_from_cash(value: float, arm: ControlTowerArm, cost: float | None = None) -> float:
    compute = cost if cost is not None else arm.cost_units()
    return value - compute


_REWARD_BY_BUCKET: dict[str, Callable[[float, ControlTowerArm, float | None], float]] = {
    "revenue": reward_from_revenue,
    "pipeline": reward_from_pipeline,
    "ops": reward_from_ops,
    "cash": reward_from_cash,
}


def _tower_bandit(epsilon: float = 0.15, optimistic: float = 1.0,
                  arms: tuple[ControlTowerArm, ...] = DEFAULT_TOWER) -> EpsilonGreedyBandit:
    return EpsilonGreedyBandit(arms, epsilon=epsilon, optimistic=optimistic)


def _refresh_queries(args: dict) -> ToolResult:
    name = args.get("arm", "control_tower_arm")
    return ToolResult(ok=True, text=f"Refreshed Metabase query set: {name}")


class ControlTowerTenant:
    """Context Runtime tenant that tunes Metabase refresh plans."""

    def __init__(self, runtime: ContextRuntime | None = None,
                 arms: tuple[ControlTowerArm, ...] = DEFAULT_TOWER,
                 bandit: EpsilonGreedyBandit | None = None, epsilon: float = 0.15,
                 refresh_tool_factory: Callable[[dict], ToolResult] | None = None):
        self.runtime = runtime or ContextRuntime.default([])
        self.arms = arms
        self.bandit = bandit or _tower_bandit(epsilon=epsilon, arms=arms)
        self.registry = ToolRegistry()
        refresh_fn = refresh_tool_factory or _refresh_queries
        self.registry.register(function_tool(
            name="refresh_metabase",
            description="Refresh the Metabase query set (simulated).",
            fn=refresh_fn,
        ))
        self._pending: dict[str, tuple[Plan, ControlTowerArm, str, str]] = {}

    def choose(self, goal_text: str, bucket: str | None = None) -> ControlTowerArm:
        plan = self.runtime.plan(Goal(text=goal_text))
        ctx_bucket = bucket or control_tower_bucket(goal_text)
        arm = self.bandit.select(ctx_bucket)
        result = self.registry.run("refresh_metabase", {"arm": arm.key})
        note = result.text or ""
        self._pending[self._key(goal_text)] = (plan, arm, ctx_bucket, note)
        return arm

    def record_outcome(self, goal_text: str, value: float, cost: float | None = None) -> float:
        key = self._key(goal_text)
        entry = self._pending.pop(key, None)
        if entry is None:
            return 0.0
        plan, arm, bucket, note = entry
        reward_fn = _REWARD_BY_BUCKET.get(bucket, reward_from_revenue)
        reward = reward_fn(value, arm, cost)
        self.bandit.update(bucket, arm, reward)
        token_estimate = max(len(note.split()) * 4, 24) if note else 24
        actual_cost_usd = (cost if cost is not None else arm.cost_units()) * 0.03
        self.runtime.estimator.observe(plan, Trace(
            plan_id=plan.id,
            goal_text=goal_text,
            actual_tokens=token_estimate,
            actual_cost_usd=actual_cost_usd,
            actual_latency_seconds=0.0,
            verification_passed=value >= (cost if cost is not None else arm.cost_units()),
        ))
        return reward

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()

    @staticmethod
    def _key(goal_text: str) -> str:
        return hashlib.sha256(goal_text.encode()).hexdigest()[:16]
