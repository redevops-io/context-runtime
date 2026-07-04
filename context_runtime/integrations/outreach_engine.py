# SPDX-License-Identifier: AGPL-3.0-or-later
"""outreach-engine × Context Runtime — pilot-outreach tenant (Growth & Intelligence).

Lands **pilot enterprise deployments** by deciding, per account, the best *outreach play* — which
buying **signal** to lead with, which **channel**, and how deep to **personalize** — then learning
which play actually converts (reply → meeting → pilot) for each kind of account. Wraps an
open-source CRM core (**Twenty**) as the system of record; the intelligence is Context Runtime's.

The decision is a cost/quality trade the runtime is built for: an *artifact* teardown (run EXPLAIN /
a redevops-rag pass over the prospect's own public RAG artifact) converts technical buyers far
better than a template — but it's expensive, so it only pays on high-signal accounts. The bandit
learns to spend effort where it converts and to stay cheap (or skip) where it won't. Sends are
**approval-gated** (human-in-the-loop), like the other agentic modules.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from ..runtime.runtime import ContextRuntime
from ..tools.base import ToolRegistry, ToolResult, function_tool
from ..types import Goal, Plan, Trace
from .bandit import EpsilonGreedyBandit

# ──────────────────────────── outreach plays (the arms) ────────────────────────────


@dataclass(frozen=True)
class OutreachPlay:
    """One outreach play: which signal to lead with · channel · personalization depth (a bandit arm)."""

    signal: str   # "funding" | "hiring" | "tech_pain" | "leadership" | "cold"
    channel: str  # "email" | "linkedin" | "multi" (email+LinkedIn) | "video" (multi + Loom)
    depth: str    # "template" | "company" (company-level research) | "artifact" (EXPLAIN teardown)

    @property
    def key(self) -> str:
        return f"{self.signal}:{self.channel}:{self.depth}"

    def cost_units(self) -> float:
        # effort/$: a product-generated artifact teardown is the expensive, high-converting play;
        # multichannel + video cost more; a template email is nearly free.
        depth_cost = {"template": 0.4, "company": 1.2, "artifact": 2.6}.get(self.depth, 1.0)
        channel_cost = {"email": 0.5, "linkedin": 0.7, "multi": 1.1, "video": 1.8}.get(self.channel, 0.8)
        signal_cost = {"cold": 0.2}.get(self.signal, 0.6)   # signal-triggered plays need detection work
        return round(depth_cost + channel_cost + signal_cost, 3)


# A spanning set: cheap cold templates → the expensive artifact-teardown plays the research favors
# for high-signal technical accounts. The bandit learns which pays off per account bucket.
DEFAULT_PLAYS: tuple[OutreachPlay, ...] = (
    OutreachPlay("cold", "email", "template"),            # the spray baseline
    OutreachPlay("hiring", "email", "company"),           # company research on a hiring signal
    OutreachPlay("hiring", "multi", "artifact"),          # teardown + email+LinkedIn
    OutreachPlay("funding", "multi", "artifact"),         # post-round window, full effort
    OutreachPlay("tech_pain", "email", "artifact"),       # EXPLAIN teardown of their RAG artifact
    OutreachPlay("tech_pain", "video", "artifact"),       # + Loom of the EXPLAIN (6× reply)
    OutreachPlay("leadership", "linkedin", "company"),    # new Head of AI → founder LinkedIn touch
)


# ──────────────────────────── bucket + reward ────────────────────────────


def outreach_bucket(text: str) -> str:
    """The account bucket the play is chosen for — the strongest signal on the account."""
    q = text.lower()
    if any(k in q for k in ("raised", "funding", "series", "seed round")):
        return "funded"
    if any(k in q for k in ("hiring", "job", "req", "opening", "ml infra", "ml platform")):
        return "hiring"
    if any(k in q for k in ("rag", "retrieval", "vector", "reranking", "hallucinat", "latency", "cost")):
        return "tech_pain"
    if any(k in q for k in ("new head", "hired", "vp ", "appointed", "joins as")):
        return "leadership"
    return "cold"


def reward_from_pilot(value: float, play: OutreachPlay, cost: float | None = None) -> float:
    """Outcome value (reply→meeting→pilot, weighted) minus the play's effort cost."""
    return round(value - (cost if cost is not None else play.cost_units()), 4)


def _outreach_bandit(epsilon: float = 0.15, optimistic: float = 1.0,
                     arms: tuple[OutreachPlay, ...] = DEFAULT_PLAYS) -> EpsilonGreedyBandit:
    return EpsilonGreedyBandit(arms, epsilon=epsilon, optimistic=optimistic)


# ──────────────────────────── the artifact-teardown tool (the wedge) ────────────────────────────


def _draft_teardown(args: dict) -> ToolResult:
    """Draft the personalized opener. For an 'artifact' play this is a product-generated teardown —
    the EXPLAIN / redevops-rag pass over the prospect's own public RAG artifact — which is what
    converts technical buyers. Simulated here; in production it calls the runtime's own EXPLAIN."""
    account = args.get("account", "the account")
    signal = args.get("signal", "cold")
    depth = args.get("depth", "template")
    channel = args.get("channel", "email")
    if depth == "artifact":
        body = (f"[{channel}] {account}: ran EXPLAIN over your public RAG path — retrieval likely "
                f"leans on one method; a calibrated, quality-routed plan would change what's served. "
                f"3-bullet teardown + live /planner view attached. (signal: {signal})")
    elif depth == "company":
        body = (f"[{channel}] {account}: saw the {signal} signal — noted your context/RAG build; "
                f"here's how teams cut retrieval sprawl with a self-hostable planner.")
    else:
        body = f"[{channel}] {account}: quick note on production RAG context optimization."
    return ToolResult(ok=True, text=body, data={"account": account, "signal": signal,
                                                "channel": channel, "depth": depth})


# ──────────────────────────── tenant ────────────────────────────


class OutreachEngineTenant:
    """Context Runtime tenant for pilot-outreach play selection (Twenty CRM as the OSS core)."""

    def __init__(self, runtime: ContextRuntime | None = None,
                 arms: tuple[OutreachPlay, ...] = DEFAULT_PLAYS,
                 bandit: EpsilonGreedyBandit | None = None, epsilon: float = 0.15,
                 bucket_fn: Callable[[str], str] = outreach_bucket,
                 teardown_tool_factory: Callable[[dict], ToolResult] | None = None,
                 approver: Callable[[str], bool] | None = None):
        self.runtime = runtime or ContextRuntime.default([])
        self.arms = arms
        self.bandit = bandit or _outreach_bandit(epsilon=epsilon, arms=arms)
        self.bucket_fn = bucket_fn
        # human-in-the-loop: send_sequence is approval-gated (default-deny), like the other modules.
        self.approver = approver or (lambda action: False)
        self.registry = ToolRegistry()
        self.registry.register(function_tool(
            name="draft_teardown",
            description="Draft the personalized opener (EXPLAIN teardown for artifact plays).",
            fn=teardown_tool_factory or _draft_teardown,
        ))
        self._pending: dict[str, tuple[Plan, OutreachPlay, str, str]] = {}

    def choose(self, account_signal: str, bucket: str | None = None) -> OutreachPlay:
        """Pick the outreach play for an account (and draft its opener via the teardown tool)."""
        plan = self.runtime.plan(Goal(text=account_signal))
        ctx_bucket = bucket or self.bucket_fn(account_signal)
        play = self.bandit.select(ctx_bucket)
        opener = self.registry.run("draft_teardown", {
            "account": account_signal, "signal": play.signal,
            "channel": play.channel, "depth": play.depth,
        }).text or ""
        self._pending[self._key(account_signal)] = (plan, play, ctx_bucket, opener)
        return play

    def send_sequence(self, account_signal: str) -> dict:
        """Attempt to send the drafted sequence — gated behind human approval (default-deny)."""
        entry = self._pending.get(self._key(account_signal))
        if entry is None:
            return {"status": "no-draft"}
        _, play, _, opener = entry
        if not self.approver("send_sequence"):
            return {"status": "pending_approval", "action": "send_sequence",
                    "play": play.key, "preview": opener}
        return {"status": "sent", "play": play.key, "preview": opener}

    def record_outcome(self, account_signal: str, value: float, cost: float | None = None) -> float:
        """Close the loop: value = reply→meeting→pilot (weighted); the tenant learns which play wins."""
        entry = self._pending.pop(self._key(account_signal), None)
        if entry is None:
            return 0.0
        plan, play, bucket, opener = entry
        reward = reward_from_pilot(value, play, cost)
        self.bandit.update(bucket, play, reward)
        tokens = max(len(opener.split()) * 4, 32) if opener else 32
        actual_cost_units = cost if cost is not None else play.cost_units()
        self.runtime.estimator.observe(plan, Trace(
            plan_id=plan.id, goal_text=account_signal, actual_tokens=tokens,
            actual_cost_usd=actual_cost_units * 0.015, actual_latency_seconds=0.0,
            verification_passed=value >= actual_cost_units))
        return reward

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
