"""agentic-support × Context Runtime — support-context tuning tenant.

Clone of ``agentic_billing``'s structure: the tenant chooses among discrete context
bundles (bandit arms) keyed by a ticket bucket and learns which bundle resolves the
ticket at the lowest retrieval cost. ``examples/agentic_support.py`` drives a 72-round
offline benchmark proving Context Runtime learns a better policy than a fixed bundle.

Licensed under AGPL-3.0 (see LICENSE).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from ..runtime.runtime import ContextRuntime
from ..tools.base import ToolRegistry, ToolResult, function_tool
from ..types import Goal, Trace
from .bandit import EpsilonGreedyBandit


# ──────────────────────────── context bundles (bandit arms) ────────────────────────────


@dataclass(frozen=True)
class SupportContextBundle:
    """One concrete bundle of context to retrieve before answering a ticket."""

    include_kb: bool
    include_tickets: bool
    include_account: bool
    include_escalation: bool
    name: str

    @property
    def key(self) -> str:
        return self.name

    def cost_units(self) -> float:
        cost = 1.0
        if self.include_kb:
            cost += 0.6
        if self.include_tickets:
            cost += 0.8
        if self.include_account:
            cost += 0.7
        if self.include_escalation:
            cost += 0.9
        return cost


DEFAULT_SUPPORT: tuple[SupportContextBundle, ...] = (
    SupportContextBundle(True, True, True, True, "full_context"),
    SupportContextBundle(True, True, False, False, "kb_tickets"),
    SupportContextBundle(False, False, True, True, "account_escalation"),
    SupportContextBundle(True, False, True, False, "kb_account"),
    SupportContextBundle(False, True, False, True, "tickets_escalation"),
    SupportContextBundle(True, False, False, True, "kb_escalation"),
)

# The one decisive source per ticket bucket — the cheapest bundle that includes it wins.
DECISIVE_BY_BUCKET: dict[str, str] = {
    "howto": "include_kb",
    "bug": "include_tickets",
    "billing_q": "include_account",
    "outage": "include_escalation",
    "general": "include_kb",
}


# ──────────────────────────── buckets and rewards ────────────────────────────


def agentic_support_bucket(text: str) -> str:
    lowered = text.lower()
    if any(k in lowered for k in ("down", "outage", "unavailable", "500", "incident", "not loading")):
        return "outage"
    if any(k in lowered for k in ("invoice", "charge", "billing", "refund", "payment", "subscription")):
        return "billing_q"
    if any(k in lowered for k in ("error", "bug", "crash", "broken", "not working", "fails")):
        return "bug"
    if any(k in lowered for k in ("how", "setup", "configure", "guide", "where do i", "enable")):
        return "howto"
    return "general"


def reward_from_resolution(value: float, bundle: SupportContextBundle, cost: float | None = None) -> float:
    return value - (cost if cost is not None else bundle.cost_units())


# ──────────────────────────── tenant ────────────────────────────


def _support_bandit(*, epsilon: float = 0.15, arms: tuple[SupportContextBundle, ...] = DEFAULT_SUPPORT,
                    bandit: EpsilonGreedyBandit | None = None) -> EpsilonGreedyBandit:
    return bandit or EpsilonGreedyBandit(arms, epsilon=epsilon)


def _simulate_retrieve(inputs: dict) -> str:
    return (f"Context bundle {inputs.get('bundle')} retrieved: "
            f"kb={inputs.get('kb')} tickets={inputs.get('tickets')} "
            f"account={inputs.get('account')} escalation={inputs.get('escalation')}")


class AgenticSupportTenant:
    def __init__(self, runtime: ContextRuntime | None = None,
                 arms: tuple[SupportContextBundle, ...] = DEFAULT_SUPPORT,
                 bandit: EpsilonGreedyBandit | None = None, epsilon: float = 0.15,
                 retrieve_tool_factory: Callable[[dict], ToolResult] | None = None):
        self.runtime = runtime or ContextRuntime.default([])
        self.arms = arms
        self.bandit = _support_bandit(epsilon=epsilon, arms=arms, bandit=bandit)
        self.registry = ToolRegistry()
        retrieve_fn = retrieve_tool_factory or _simulate_retrieve
        self.registry.register(function_tool(
            name="retrieve_context",
            description="Retrieve the selected support context bundle (simulated).",
            fn=retrieve_fn,
        ))
        self._pending: dict[str, tuple] = {}

    def choose(self, ticket: str, bucket: str | None = None) -> SupportContextBundle:
        plan = self.runtime.plan(Goal(text=ticket))
        ctx_bucket = bucket or agentic_support_bucket(ticket)
        bundle = self.bandit.select(ctx_bucket)
        _ = self.registry.run("retrieve_context", {
            "bundle": bundle.key,
            "kb": bundle.include_kb,
            "tickets": bundle.include_tickets,
            "account": bundle.include_account,
            "escalation": bundle.include_escalation,
        })
        self._pending[self._key(ticket)] = (plan, bundle, ctx_bucket)
        return bundle

    def record_outcome(self, ticket: str, value: float, cost: float | None = None) -> float:
        key = self._key(ticket)
        entry = self._pending.pop(key, None)
        if entry is None:
            return 0.0
        plan, bundle, bucket = entry
        reward = reward_from_resolution(value, bundle, cost)
        self.bandit.update(bucket, bundle, reward)
        self.runtime.estimator.observe(plan, Trace(
            plan_id=plan.id,
            goal_text=ticket,
            actual_tokens=12,
            actual_cost_usd=(cost if cost is not None else bundle.cost_units()) * 0.02,
            actual_latency_seconds=0.0,
            verification_passed=value >= (cost if cost is not None else bundle.cost_units()),
        ))
        return reward

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()

    @staticmethod
    def _key(ticket: str) -> str:
        return hashlib.sha256(ticket.encode()).hexdigest()[:16]
