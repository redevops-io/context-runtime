"""agentic-billing × Context Runtime — collections signal tuning tenant.

Clone of ``social_autopilot``'s structure: the tenant chooses among discrete signal
bundles (bandit arms) keyed by a billing health bucket and learns which bundle yields
high repayment value minus data-fetch cost. ``examples/agentic_billing.py`` drives a
72-round offline benchmark proving Context Runtime learns a better policy than a
fixed bundle.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from ..runtime.runtime import ContextRuntime
from ..tools.base import ToolRegistry, ToolResult, function_tool
from ..types import Goal, Trace
from .bandit import EpsilonGreedyBandit


# ──────────────────────────── signal bundles (bandit arms) ────────────────────────────


@dataclass(frozen=True)
class BillingSignalBundle:
    """One concrete bundle of signals to pull before acting."""

    include_usage: bool
    include_invoice: bool
    include_dunning: bool
    include_payment_history: bool
    name: str

    @property
    def key(self) -> str:
        return self.name

    def cost_units(self) -> float:
        cost = 1.0
        if self.include_usage:
            cost += 0.6
        if self.include_invoice:
            cost += 0.8
        if self.include_dunning:
            cost += 0.9
        if self.include_payment_history:
            cost += 0.7
        return cost


DEFAULT_BILLING: tuple[BillingSignalBundle, ...] = (
    BillingSignalBundle(True, True, True, True, "full_stack"),
    BillingSignalBundle(True, False, True, False, "usage_dunning"),
    BillingSignalBundle(False, True, True, False, "invoice_dunning"),
    BillingSignalBundle(True, True, False, False, "usage_invoice"),
    BillingSignalBundle(False, True, False, True, "invoice_history"),
    BillingSignalBundle(True, False, False, True, "usage_history"),
)


# ──────────────────────────── buckets and rewards ────────────────────────────


def agentic_billing_bucket(text: str) -> str:
    lowered = text.lower()
    if "overdue" in lowered or "past due" in lowered or "delinquent" in lowered:
        return "delinquent"
    if "churn" in lowered or "warning" in lowered or "risk" in lowered:
        return "at_risk"
    return "healthy"


def reward_from_value(value: float, bundle: BillingSignalBundle, cost: float | None = None) -> float:
    return value - (cost if cost is not None else bundle.cost_units())


def reward_from_delinquent(value: float, bundle: BillingSignalBundle, cost: float | None = None) -> float:
    return reward_from_value(value, bundle, cost)


def reward_from_at_risk(value: float, bundle: BillingSignalBundle, cost: float | None = None) -> float:
    return reward_from_value(value, bundle, cost)


def reward_from_healthy(value: float, bundle: BillingSignalBundle, cost: float | None = None) -> float:
    return reward_from_value(value, bundle, cost)


# ──────────────────────────── tenant ────────────────────────────


def _billing_bandit(*, epsilon: float = 0.15, arms: tuple[BillingSignalBundle, ...] = DEFAULT_BILLING,
                    bandit: EpsilonGreedyBandit | None = None) -> EpsilonGreedyBandit:
    return bandit or EpsilonGreedyBandit(arms, epsilon=epsilon)


def _simulate_fetch(inputs: dict) -> str:
    return (f"Signal bundle {inputs.get('bundle')} fetched: "
            f"usage={inputs.get('usage')} invoice={inputs.get('invoice')} "
            f"dunning={inputs.get('dunning')} history={inputs.get('history')}")


class AgenticBillingTenant:
    def __init__(self, runtime: ContextRuntime | None = None,
                 arms: tuple[BillingSignalBundle, ...] = DEFAULT_BILLING,
                 bandit: EpsilonGreedyBandit | None = None, epsilon: float = 0.15,
                 fetch_tool_factory: Callable[[dict], ToolResult] | None = None):
        self.runtime = runtime or ContextRuntime.default([])
        self.arms = arms
        self.bandit = _billing_bandit(epsilon=epsilon, arms=arms, bandit=bandit)
        self.registry = ToolRegistry()
        fetch_fn = fetch_tool_factory or _simulate_fetch
        self.registry.register(function_tool(
            name="fetch_signals",
            description="Fetch the selected billing signal bundle (simulated).",
            fn=fetch_fn,
        ))
        self._pending: dict[str, tuple] = {}

    def choose(self, account: str, bucket: str | None = None) -> BillingSignalBundle:
        plan = self.runtime.plan(Goal(text=account))
        ctx_bucket = bucket or agentic_billing_bucket(account)
        bundle = self.bandit.select(ctx_bucket)
        _ = self.registry.run("fetch_signals", {
            "bundle": bundle.key,
            "usage": bundle.include_usage,
            "invoice": bundle.include_invoice,
            "dunning": bundle.include_dunning,
            "history": bundle.include_payment_history,
        })
        self._pending[self._key(account)] = (plan, bundle, ctx_bucket)
        return bundle

    def record_outcome(self, account: str, value: float, cost: float | None = None) -> float:
        key = self._key(account)
        entry = self._pending.pop(key, None)
        if entry is None:
            return 0.0
        plan, bundle, bucket = entry
        reward = reward_from_value(value, bundle, cost)
        self.bandit.update(bucket, bundle, reward)
        self.runtime.estimator.observe(plan, Trace(
            plan_id=plan.id,
            goal_text=account,
            actual_tokens=12,
            actual_cost_usd=(cost if cost is not None else bundle.cost_units()) * 0.02,
            actual_latency_seconds=0.0,
            verification_passed=value >= (cost if cost is not None else bundle.cost_units()),
        ))
        return reward

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()

    @staticmethod
    def _key(account: str) -> str:
        return hashlib.sha256(account.encode()).hexdigest()[:16]
