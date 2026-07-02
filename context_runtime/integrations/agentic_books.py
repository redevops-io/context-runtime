"""agentic-books × Context Runtime — ledger/report selection tuning tenant.

Clone of ``agentic_billing``'s structure: the tenant chooses among discrete report
bundles (bandit arms) keyed by a books-question bucket and learns the cheapest report
set that still answers the question correctly. ``examples/agentic_books.py`` drives a
72-round offline benchmark proving Context Runtime beats a fixed full-books bundle.

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


# ──────────────────────────── report bundles (bandit arms) ────────────────────────────


@dataclass(frozen=True)
class BooksReportBundle:
    """One concrete bundle of ledgers/reports to pull before answering a books question."""

    include_receivables: bool
    include_payables: bool
    include_tax: bool
    include_trial_balance: bool
    name: str

    @property
    def key(self) -> str:
        return self.name

    def cost_units(self) -> float:
        cost = 1.0
        if self.include_receivables:
            cost += 0.7
        if self.include_payables:
            cost += 0.7
        if self.include_tax:
            cost += 0.8
        if self.include_trial_balance:
            cost += 0.9
        return cost


DEFAULT_BOOKS: tuple[BooksReportBundle, ...] = (
    BooksReportBundle(True, True, True, True, "full_books"),
    BooksReportBundle(True, True, False, False, "ar_ap"),
    BooksReportBundle(False, False, True, True, "tax_close"),
    BooksReportBundle(True, False, True, False, "ar_tax"),
    BooksReportBundle(False, True, False, True, "ap_close"),
    BooksReportBundle(True, False, False, True, "ar_close"),
)

DECISIVE_BY_BUCKET: dict[str, str] = {
    "ar": "include_receivables",
    "ap": "include_payables",
    "tax": "include_tax",
    "close": "include_trial_balance",
    "general": "include_trial_balance",
}


# ──────────────────────────── buckets and rewards ────────────────────────────


def agentic_books_bucket(text: str) -> str:
    lowered = text.lower()
    if any(k in lowered for k in ("receivable", "invoice owed", "customer owes", "collections", "ar aging", "who owes")):
        return "ar"
    if any(k in lowered for k in ("payable", "vendor", "bill to pay", "ap aging", "we owe", "supplier")):
        return "ap"
    if any(k in lowered for k in ("tax", "vat", "gst", "sales tax", "liability")):
        return "tax"
    if any(k in lowered for k in ("close", "trial balance", "month end", "reconcile", "p&l", "balance sheet")):
        return "close"
    return "general"


def reward_from_answer(value: float, bundle: BooksReportBundle, cost: float | None = None) -> float:
    return value - (cost if cost is not None else bundle.cost_units())


# ──────────────────────────── tenant ────────────────────────────


def _books_bandit(*, epsilon: float = 0.15, arms: tuple[BooksReportBundle, ...] = DEFAULT_BOOKS,
                  bandit: EpsilonGreedyBandit | None = None) -> EpsilonGreedyBandit:
    return bandit or EpsilonGreedyBandit(arms, epsilon=epsilon)


def _simulate_pull(inputs: dict) -> str:
    return (f"Report bundle {inputs.get('bundle')} pulled: "
            f"receivables={inputs.get('receivables')} payables={inputs.get('payables')} "
            f"tax={inputs.get('tax')} trial_balance={inputs.get('trial_balance')}")


class AgenticBooksTenant:
    def __init__(self, runtime: ContextRuntime | None = None,
                 arms: tuple[BooksReportBundle, ...] = DEFAULT_BOOKS,
                 bandit: EpsilonGreedyBandit | None = None, epsilon: float = 0.15,
                 pull_tool_factory: Callable[[dict], ToolResult] | None = None):
        self.runtime = runtime or ContextRuntime.default([])
        self.arms = arms
        self.bandit = _books_bandit(epsilon=epsilon, arms=arms, bandit=bandit)
        self.registry = ToolRegistry()
        pull_fn = pull_tool_factory or _simulate_pull
        self.registry.register(function_tool(
            name="pull_reports",
            description="Pull the selected ledger/report bundle (simulated).",
            fn=pull_fn,
        ))
        self._pending: dict[str, tuple] = {}

    def choose(self, question: str, bucket: str | None = None) -> BooksReportBundle:
        plan = self.runtime.plan(Goal(text=question))
        ctx_bucket = bucket or agentic_books_bucket(question)
        bundle = self.bandit.select(ctx_bucket)
        _ = self.registry.run("pull_reports", {
            "bundle": bundle.key,
            "receivables": bundle.include_receivables,
            "payables": bundle.include_payables,
            "tax": bundle.include_tax,
            "trial_balance": bundle.include_trial_balance,
        })
        self._pending[self._key(question)] = (plan, bundle, ctx_bucket)
        return bundle

    def record_outcome(self, question: str, value: float, cost: float | None = None) -> float:
        key = self._key(question)
        entry = self._pending.pop(key, None)
        if entry is None:
            return 0.0
        plan, bundle, bucket = entry
        reward = reward_from_answer(value, bundle, cost)
        self.bandit.update(bucket, bundle, reward)
        self.runtime.estimator.observe(plan, Trace(
            plan_id=plan.id,
            goal_text=question,
            actual_tokens=12,
            actual_cost_usd=(cost if cost is not None else bundle.cost_units()) * 0.02,
            actual_latency_seconds=0.0,
            verification_passed=value >= (cost if cost is not None else bundle.cost_units()),
        ))
        return reward

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()

    @staticmethod
    def _key(question: str) -> str:
        return hashlib.sha256(question.encode()).hexdigest()[:16]
