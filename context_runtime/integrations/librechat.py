"""LibreChat × Context Runtime — chat as a self-learning retrieval tenant.

Context Runtime handles context for every app; LibreChat is the chat app. Each user
message is a decision point: *how should we retrieve context for this request?* The
tenant lets a contextual bandit pick a **retrieval strategy** (method · depth · rerank)
per request-type, retrieves from the multimodal corpus, and learns from the ONE signal
that matters for a RAG chat — **how well the retrieved context actually answers the
request**, scored by an LLM judge (LLM-as-a-judge). Same shared bandit + cost-model as
every other tenant; only the arms (retrieval strategies) and the reward (judged
retrieval quality − retrieval cost) are app-specific.

The measurable benchmark is retrieval quality vs. the user request, so the loop is:
    request → plan → bandit picks strategy → retrieve → LLM judges the context →
    reward → policy learns the best strategy for that kind of request.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..adapters.store_inmemory import InMemoryStore
from ..runtime.runtime import ContextRuntime
from ..types import Goal, Hit, Plan, Trace
from .bandit import EpsilonGreedyBandit

# ──────────────────────────── the arms: retrieval strategies ────────────────────────────


@dataclass(frozen=True)
class RetrievalStrategy:
    """How to retrieve context for a request (the bandit arm)."""

    method: str = "hybrid"   # "bm25" | "vector" | "hybrid" | "graph"
    final_k: int = 5         # how many chunks to surface
    rerank: bool = False     # spend an extra rerank pass (more cost)

    @property
    def key(self) -> str:
        return f"{self.method}:k{self.final_k}:{'rr' if self.rerank else 'norr'}"

    def cost_units(self) -> float:
        # deeper retrieval + a rerank pass cost more; the frontier is cheapest-good-enough.
        return self.final_k / 5.0 + (0.8 if self.rerank else 0.0) + (0.4 if self.method == "graph" else 0.0)


DEFAULT_STRATEGIES: tuple[RetrievalStrategy, ...] = (
    RetrievalStrategy("bm25", 3, False),      # cheap keyword
    RetrievalStrategy("hybrid", 5, False),    # the sensible default
    RetrievalStrategy("hybrid", 8, True),     # thorough + rerank
    RetrievalStrategy("vector", 5, False),    # semantic
    RetrievalStrategy("graph", 6, False),     # multi-hop (connective questions)
)

COST_LAMBDA = 0.15   # how much retrieval cost trades against judged quality


# ──────────────────────────── the LLM judge (the reward signal) ────────────────────────────

# A judge maps (request, retrieved_context) → quality in [0,1]. In production this is an
# LLM ("rate how well this context answers the request"); offline it is a heuristic.
Judge = "callable(request: str, context: str, hits: tuple[Hit, ...]) -> float"


def heuristic_judge(request: str, context: str, hits: tuple[Hit, ...]) -> float:
    """Offline stand-in for the LLM judge: coverage of the request's salient terms in
    the retrieved context, lightly rewarding grounded (non-empty, multi-source) results."""
    terms = {w for w in _tokens(request) if len(w) > 2}
    if not terms:
        return 0.0
    ctx = context.lower()
    covered = sum(1 for t in terms if t in ctx)
    coverage = covered / len(terms)
    grounding = min(1.0, len({h.filename for h in hits}) / 3.0)
    return round(0.8 * coverage + 0.2 * grounding, 4)


def llm_judge(model) -> Judge:
    """Build an LLM-as-a-judge from any ModelPlugin. It asks the model to score, 0..1,
    how well the retrieved context answers the request, and parses the number."""
    def _judge(request: str, context: str, hits: tuple[Hit, ...]) -> float:
        prompt = (
            "You are grading a retrieval system for a chat assistant. Given a USER "
            "REQUEST and the CONTEXT the system retrieved, rate from 0.0 to 1.0 how well "
            "the context lets an assistant answer the request (1.0 = fully sufficient and "
            "on-topic, 0.0 = irrelevant/empty). Reply with ONLY the number.\n\n"
            f"USER REQUEST:\n{request}\n\nRETRIEVED CONTEXT:\n{context[:2000]}\n\nSCORE:")
        from ..types import ModelRequest
        # Reasoning models (e.g. kimi-k2.6) emit reasoning before the answer and return
        # empty content when max_tokens is tiny — reasoning grows with context size, so
        # give ample headroom for the number to land after the reasoning.
        import os
        budget = int(os.getenv("CR_JUDGE_MAX_TOKENS", "1024"))
        result = model.complete(ModelRequest(messages=[{"role": "user", "content": prompt}],
                                              capability="judge", max_tokens=budget))
        return _parse_score(result.text)
    return _judge


def _parse_score(text: str) -> float:
    import re
    m = re.search(r"(\d*\.?\d+)", text or "")
    if not m:
        return 0.0
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return 0.0


def _tokens(s: str) -> list[str]:
    import re
    return re.findall(r"\w+", s.lower())


def reward_from_judgment(score: float, strategy: RetrievalStrategy,
                         strategies: tuple[RetrievalStrategy, ...] = DEFAULT_STRATEGIES) -> float:
    """Judged retrieval quality minus a normalized retrieval-cost penalty."""
    max_cost = max(s.cost_units() for s in strategies)
    cost_norm = strategy.cost_units() / max_cost if max_cost else 0.0
    return round(max(0.0, score - COST_LAMBDA * cost_norm), 4)


# ──────────────────────────── the tenant ────────────────────────────


@dataclass
class ChatContext:
    request: str
    strategy: RetrievalStrategy
    hits: tuple[Hit, ...]
    context: str
    plan: Plan


class LibreChatTenant:
    """LibreChat as a Context Runtime tenant: learns the retrieval strategy that yields
    the best LLM-judged context per request-type, over an ingested (multimodal) corpus."""

    def __init__(self, corpus_dir: str | None = None, runtime: ContextRuntime | None = None,
                 retriever=None, bandit: EpsilonGreedyBandit | None = None,
                 strategies: tuple[RetrievalStrategy, ...] = DEFAULT_STRATEGIES,
                 judge: Judge | None = None, persist_path: str | None = None):
        self.runtime = runtime or ContextRuntime.default([])
        self.retriever = retriever or InMemoryStore([])
        if corpus_dir:
            self.retriever.index(corpus_dir)
        self.strategies = strategies
        self.bandit = bandit or EpsilonGreedyBandit(strategies, epsilon=0.12, persist_path=persist_path)
        self.judge = judge or heuristic_judge
        self._pending: dict[str, tuple[Plan, RetrievalStrategy]] = {}

    def ingest(self, corpus_dir: str) -> dict:
        """Index (more of) the corpus into the retriever."""
        return self.retriever.index(corpus_dir)

    def retrieve(self, request: str) -> ChatContext:
        """Plan the request, let the bandit pick a strategy, and retrieve context.
        No judging yet — call record_judgment (or handle) to close the loop."""
        plan = self.runtime.plan(Goal(text=request))
        strategy = self.bandit.select(plan.intent.bucket)
        hits = tuple(self.retriever.search(request, k=strategy.final_k, method=strategy.method))
        if strategy.rerank:
            hits = hits[:strategy.final_k]
        context = "\n\n".join(f"[{i+1}] {h.text}" for i, h in enumerate(hits))
        self._pending[self._key(request)] = (plan, strategy)
        return ChatContext(request, strategy, hits, context, plan)

    def record_judgment(self, request: str, score: float) -> float:
        """Feed the judged retrieval quality (0..1) back: the policy learns which strategy
        yields the best-judged context for this kind of request. Returns the reward."""
        key = self._key(request)
        if key not in self._pending:
            return 0.0
        plan, strategy = self._pending.pop(key)
        reward = reward_from_judgment(score, strategy, self.strategies)
        self.bandit.update(plan.intent.bucket, strategy, reward)
        self.runtime.estimator.observe(plan, Trace(
            plan_id=plan.id, goal_text=request, actual_tokens=strategy.final_k * 200,
            verification_passed=score >= 0.6))
        return reward

    def handle(self, request: str, judge: Judge | None = None) -> tuple[ChatContext, float, float]:
        """Retrieve → judge → learn, in one call. Returns (context, judged_score, reward)."""
        ctx = self.retrieve(request)
        score = (judge or self.judge)(request, ctx.context, ctx.hits)
        reward = self.record_judgment(request, score)
        return ctx, score, reward

    def suggest(self, request: str) -> str:
        """The retrieval strategy LibreChat has learned for this kind of request."""
        plan = self.runtime.plan(Goal(text=request))
        return self.bandit.policy().get(plan.intent.bucket, "(unlearned)")

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()

    @staticmethod
    def _key(request: str) -> str:
        return hashlib.sha256(request.encode()).hexdigest()[:16]
