"""Cost-aware LLM router.

Every agent task declares the *capability* it needs; the router sends it to the cheapest
tier that provides that capability and falls back up the tiers on failure — keeping the
bulk of work on local hardware and reserving premium APIs for the hard minority.

This is the cost engine of the OS: the same idea we used to build the redevops.io modules
themselves across local Qwen3.5 / DeepSeek-V4, cheap Kimi/Grok, and premium GPT-5-codex.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class Tier:
    name: str
    base_url: str
    model: str
    good_for: frozenset[str]
    api_key: str | None = None
    # rough $ per 1k output tokens, used only for budget accounting / ordering
    cost_per_1k: float = 0.0

    def can(self, capability: str) -> bool:
        return capability in self.good_for


@dataclass
class Task:
    """A unit of work handed to the router."""
    prompt: str
    capability: str = "draft"          # what this task needs; see Tier.good_for
    system: str | None = None
    max_tokens: int = 1024


@dataclass
class RouteResult:
    tier: str
    model: str
    text: str
    est_cost_usd: float


class BudgetExceeded(RuntimeError):
    pass


class Router:
    """Tries tiers cheapest-first; the first that meets the capability handles the task."""

    def __init__(self, tiers: list[Tier], monthly_budget_usd: float | None = None):
        # keep declared order, but a stable sort by cost makes "cheapest-first" explicit
        self.tiers = sorted(tiers, key=lambda t: t.cost_per_1k)
        self.monthly_budget_usd = monthly_budget_usd
        self.spent_usd = 0.0

    @classmethod
    def from_config(cls, cfg: dict) -> "Router":
        tiers = []
        for t in cfg.get("tiers", []):
            tiers.append(Tier(
                name=t["name"], base_url=t["base_url"], model=t["model"],
                good_for=frozenset(t.get("good_for", [])),
                api_key=_resolve_env(t.get("api_key")),
                cost_per_1k=float(t.get("cost_per_1k", 0.0)),
            ))
        return cls(tiers, cfg.get("monthly_budget_usd"))

    def select(self, capability: str) -> Tier:
        for tier in self.tiers:
            if tier.can(capability):
                return tier
        raise LookupError(f"no tier provides capability {capability!r} "
                          f"(have: {[t.name for t in self.tiers]})")

    def run(self, task: Task) -> RouteResult:
        """Route the task, falling back up the tiers on transport failure."""
        candidates = [t for t in self.tiers if t.can(task.capability)]
        if not candidates:
            raise LookupError(f"no tier for capability {task.capability!r}")
        last_err: Exception | None = None
        for tier in candidates:
            try:
                text, out_tokens = self._chat(tier, task)
            except (urllib.error.URLError, OSError, ValueError) as e:
                last_err = e
                continue
            cost = (out_tokens / 1000.0) * tier.cost_per_1k
            if self.monthly_budget_usd is not None and self.spent_usd + cost > self.monthly_budget_usd:
                raise BudgetExceeded(f"task would exceed monthly budget ${self.monthly_budget_usd}")
            self.spent_usd += cost
            return RouteResult(tier=tier.name, model=tier.model, text=text, est_cost_usd=cost)
        raise RuntimeError(f"all tiers failed for {task.capability!r}: {last_err}")

    def _chat(self, tier: Tier, task: Task) -> tuple[str, int]:
        messages = []
        if task.system:
            messages.append({"role": "system", "content": task.system})
        messages.append({"role": "user", "content": task.prompt})
        body = {"model": tier.model, "messages": messages, "max_tokens": task.max_tokens}
        req = urllib.request.Request(
            f"{tier.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {tier.api_key or 'EMPTY'}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        out_tokens = int((data.get("usage") or {}).get("completion_tokens", 0))
        return text, out_tokens


def _resolve_env(value: str | None) -> str | None:
    """Resolve `$VAR` references from the environment (so config holds names, not secrets)."""
    if value and value.startswith("$"):
        return os.environ.get(value[1:])
    return value
