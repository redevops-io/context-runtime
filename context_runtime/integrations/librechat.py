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
from .calibration import CalibrationLog, CalibrationMap
from .loadmeter import LoadMeter

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
        extra = 0.4 if self.method == "graph" else (0.3 if self.method == "community" else 0.0)
        return self.final_k / 5.0 + (0.8 if self.rerank else 0.0) + extra


DEFAULT_STRATEGIES: tuple[RetrievalStrategy, ...] = (
    RetrievalStrategy("bm25", 3, False),      # cheap keyword
    RetrievalStrategy("hybrid", 5, False),    # the sensible default
    RetrievalStrategy("hybrid", 8, True),     # thorough + rerank
    RetrievalStrategy("vector", 5, False),    # semantic
    RetrievalStrategy("graph", 6, False),     # multi-hop (connective questions)
    RetrievalStrategy("community", 4, False),  # global/broad (aggregation questions)
)

# The retrieval methods the transparency view (compare()) runs side-by-side.
COMPARE_METHODS = ("bm25", "vector", "hybrid", "community", "graph")

COST_LAMBDA = 0.15   # how much retrieval cost trades against judged quality

# Implicit user actions → a retrieval-quality score in [0,1]. This is the app's NATIVE
# success signal (the fleet-pattern reward): a kept/thumbs-up answer means the retrieved
# context was good; a regenerate/thumbs-down means it wasn't. Cheaper, truer, and
# deterministic vs. an LLM judge — which becomes only a cold-start bootstrap.
SIGNAL_REWARDS: dict[str, float] = {
    "thumbs_up": 1.0, "kept": 0.9, "copied": 0.9, "accepted": 0.9, "cited": 0.9,
    "follow_up": 0.6, "edited": 0.5, "neutral": 0.5,
    "regenerated": 0.15, "abandoned": 0.1, "thumbs_down": 0.0, "rejected": 0.0,
}


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
        # Reasoning models (e.g. kimi-k2.6) emit reasoning before the answer and can
        # exhaust max_tokens mid-reasoning, returning EMPTY content — and reasoning
        # length is non-deterministic. Give ample headroom and retry on an empty reply
        # so a transient over-reason doesn't silently score 0.
        import os
        budget = int(os.getenv("CR_JUDGE_MAX_TOKENS", "2048"))
        attempts = int(os.getenv("CR_JUDGE_RETRIES", "2"))
        for _ in range(max(1, attempts)):
            result = model.complete(ModelRequest(messages=[{"role": "user", "content": prompt}],
                                                 capability="judge", max_tokens=budget))
            if (result.text or "").strip():
                return _parse_score(result.text)
        return 0.0
    return _judge


def llm_passage_judge(model):
    """Build a PER-PASSAGE relevance judge from any ModelPlugin — the labels the
    calibration map needs. One batched call asks the model which retrieved passages are
    relevant to the request and returns a 0/1 label per passage (mapped to [0,1]).

    Returns a callable ``(request, hits) -> tuple[float, ...]`` aligned with ``hits``.
    Per-query judges (heuristic_judge/llm_judge) only score the whole context; calibrating
    score→P(relevant) needs a signal at the granularity of the thing being scored — the
    passage — so this exists alongside them, not instead of them.
    """
    import os

    def _judge(request: str, hits: tuple[Hit, ...]) -> tuple[float, ...]:
        if not hits:
            return ()
        listing = "\n".join(f"[{i+1}] {(h.text or '')[:300]}" for i, h in enumerate(hits))
        prompt = (
            "You grade retrieval for a chat assistant. For the USER REQUEST below, decide "
            "for EACH numbered passage whether it is relevant/useful for answering the "
            "request. Reply with ONLY the numbers of the relevant passages, comma-separated "
            "(e.g. '1,3,4'); reply 'none' if none are relevant.\n\n"
            f"USER REQUEST:\n{request}\n\nPASSAGES:\n{listing}\n\nRELEVANT:")
        from ..types import ModelRequest
        budget = int(os.getenv("CR_JUDGE_MAX_TOKENS", "2048"))
        result = model.complete(ModelRequest(messages=[{"role": "user", "content": prompt}],
                                             capability="judge", max_tokens=budget))
        import re as _re
        idxs = {int(x) for x in _re.findall(r"\d+", result.text or "")}
        return tuple(1.0 if (i + 1) in idxs else 0.0 for i in range(len(hits)))
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
    probs: tuple[float, ...] = ()   # calibrated P(relevant) per hit (() when uncalibrated)
    max_p_rel: float = 1.0          # confidence of the best hit (1.0 when uncalibrated)
    abstain: bool = False           # best hit below the abstain threshold → too weak to answer


class LibreChatTenant:
    """LibreChat as a Context Runtime tenant: learns the retrieval strategy that yields
    the best LLM-judged context per request-type, over an ingested (multimodal) corpus."""

    def __init__(self, corpus_dir: str | None = None, runtime: ContextRuntime | None = None,
                 retriever=None, bandit: EpsilonGreedyBandit | None = None,
                 strategies: tuple[RetrievalStrategy, ...] = DEFAULT_STRATEGIES,
                 judge: Judge | None = None, persist_path: str | None = None,
                 query_expander=None, calibration: CalibrationMap | None = None,
                 calib_log: CalibrationLog | None = None, load_meter: LoadMeter | None = None,
                 passage_judge=None, cost_profile=None, load_aware: bool = False,
                 abstain_threshold: float | None = None):
        self.runtime = runtime or ContextRuntime.default([])
        self.retriever = retriever or InMemoryStore([])
        # optional query rewrite applied to the RETRIEVAL query only (e.g. cross-language
        # expansion). Planning, learning keys, and the answer keep the original request.
        self.query_expander = query_expander
        if corpus_dir:
            self.retriever.index(corpus_dir)
        self.strategies = strategies
        self.bandit = bandit or EpsilonGreedyBandit(strategies, epsilon=0.12, persist_path=persist_path)
        self.judge = judge or heuristic_judge
        # ── opt-in machinery (all None/False ⇒ exact legacy behaviour) ──
        self.calibration = calibration        # score→P(relevant) map, applied at retrieve time
        self.calib_log = calib_log            # append (score, label) rows to fit the map from
        self.load_meter = load_meter          # in-flight load signal for load-aware depth
        self.passage_judge = passage_judge    # per-passage relevance labels for calibration
        self.cost_profile = cost_profile      # measured latency table for the sizer's budget guard
        self.load_aware = load_aware          # size the expensive stage by load (Tier A + B)
        self.abstain_threshold = abstain_threshold  # best P(relevant) below this ⇒ abstain
        # pending/last hold everything a later reward needs: plan, arm, hits (for calibration
        # logging), and the exact bandit context used at select time (so update matches).
        self._pending: dict[str, tuple[Plan, RetrievalStrategy, tuple[Hit, ...], str]] = {}
        self._last: dict[str, tuple[Plan, RetrievalStrategy, tuple[Hit, ...], str]] = {}

    def ingest(self, corpus_dir: str) -> dict:
        """Index (more of) the corpus into the retriever."""
        return self.retriever.index(corpus_dir)

    def _select_ctx(self, plan: Plan) -> str:
        """The bandit context. Load-aware mode appends a coarse load band so the policy
        learns load-conditioned depth (shallow arms when busy, deep when idle) — reusing
        the existing bandit rather than a second selector that would fight it."""
        bucket = plan.intent.bucket
        if self.load_aware and self.load_meter is not None:
            return f"{bucket}:{self.load_meter.band()}"
        return bucket

    def retrieve(self, request: str) -> ChatContext:
        """Plan the request, let the bandit pick a strategy, and retrieve context.
        No judging yet — call record_feedback (native signal) or record_judgment (bootstrap)."""
        plan = self.runtime.plan(Goal(text=request))
        ctx_key = self._select_ctx(plan)
        strategy = self.bandit.select(ctx_key)
        rq = self._expand(request)
        hits = tuple(self.retriever.search(rq, k=strategy.final_k, method=strategy.method))
        if strategy.rerank:
            hits = hits[:strategy.final_k]

        # Calibrate scores → P(relevant); optionally load-size the expensive stage; abstain.
        probs, max_p, abstain = self._calibrate(strategy, hits)
        if probs and self.load_aware and self.load_meter is not None:
            band = self.load_meter.band()
            from ..scheduler.load_aware import size_expensive_stage
            decision = size_expensive_stage(
                list(probs), load_band=band, requested_k=strategy.final_k,
                requested_rerank=strategy.rerank, cost_profile=self.cost_profile,
                max_latency_seconds=None)
            hits = hits[:decision.final_k]
            probs = probs[:decision.final_k]

        context = "\n\n".join(f"[{i+1}] {h.text}" for i, h in enumerate(hits))
        key = self._key(request)
        self._pending[key] = (plan, strategy, hits, ctx_key)
        self._last[key] = (plan, strategy, hits, ctx_key)   # persists for late implicit feedback
        return ChatContext(request, strategy, hits, context, plan,
                           probs=probs, max_p_rel=max_p, abstain=abstain)

    def _calibrate(self, strategy: RetrievalStrategy,
                            hits: tuple[Hit, ...]) -> tuple[tuple[float, ...], float, bool]:
        """Map each hit's raw score → calibrated P(relevant) (identity if no map), stash it
        on the hit for the panel, and decide abstention from the best calibrated score."""
        if self.calibration is None or not hits:
            return (), 1.0, False
        probs = tuple(round(self.calibration.apply(strategy.method, float(h.score)), 4) for h in hits)
        for h, p in zip(hits, probs):
            h.meta["p_rel"] = p
        max_p = max(probs) if probs else 0.0
        abstain = self.abstain_threshold is not None and max_p < self.abstain_threshold
        return probs, max_p, abstain

    def _expand(self, request: str) -> str:
        """Apply the optional query rewrite (cross-language expansion) for retrieval only."""
        if self.query_expander is None:
            return request
        try:
            return self.query_expander(request) or request
        except Exception:
            return request

    def compare(self, request: str, k: int = 5) -> dict:
        """Retrieval transparency: run EVERY method side-by-side and report what the learned
        policy actually chose + served. This is the 'show your work' view — it makes the
        core thesis visible: the runtime evaluates strategies and serves the best one. Read
        only — no bandit exploration, no state mutation, so it never perturbs learning."""
        rq = self._expand(request)
        per: dict[str, list[dict]] = {}
        for m in COMPARE_METHODS:
            try:
                hits = self.retriever.search(rq, k=k, method=m)
            except Exception:
                hits = ()
            per[m] = [self._compare_hit(m, h) for h in hits]
        plan = self.runtime.plan(Goal(text=request))
        bucket = plan.intent.bucket
        key = self.bandit.policy().get(self._select_ctx(plan))
        chosen = next((s for s in self.strategies if s.key == key), None) or self.strategies[1]
        hits = self.retriever.search(rq, k=chosen.final_k, method=chosen.method)
        if chosen.rerank:
            hits = hits[:chosen.final_k]
        served = "\n\n".join(f"[{i + 1}] {h.text}" for i, h in enumerate(hits))
        return {
            "request": request,
            "methods": per,
            "chosen": {"key": chosen.key, "method": chosen.method, "final_k": chosen.final_k,
                       "rerank": chosen.rerank, "bucket": bucket, "learned": key is not None},
            "served": {"context": served[:4000], "citations": [h.chunk_id for h in hits]},
        }

    def _compare_hit(self, method: str, h: Hit) -> dict:
        """One hit for the transparency panel, with calibrated P(relevant) when available —
        an honest confidence number, not a scale-free raw score."""
        d = {"chunk_id": h.chunk_id, "filename": h.filename,
             "score": round(float(h.score), 4), "snippet": (h.text or "")[:240],
             "text": (h.text or "")[:1500]}
        if self.calibration is not None and self.calibration.has(method):
            d["p_rel"] = round(self.calibration.apply(method, float(h.score)), 4)
        return d

    def annotate_relevance(self, request: str, hits: tuple[Hit, ...]) -> None:
        """Run the per-passage judge (if configured) and stash labels on hits so the next
        calibration-log write carries per-passage relevance instead of a weak query label."""
        if self.passage_judge is None or not hits:
            return
        try:
            labels = self.passage_judge(request, hits)
            for h, r in zip(hits, labels):
                h.meta["rel"] = float(r)
        except Exception:
            pass

    def record_feedback(self, request: str, signal: str) -> float:
        """Learn from an IMPLICIT user action (kept / regenerated / thumbs_up / …) — the
        app's NATIVE success signal. This is the primary online reward (cheaper, truer, and
        deterministic vs. an LLM judge, which is now only a cold-start bootstrap)."""
        entry = self._last.get(self._key(request))
        if entry is None:
            return 0.0
        plan, strategy, hits, ctx_key = entry
        return self._learn(request, plan, strategy, SIGNAL_REWARDS.get(signal, 0.5), hits, ctx_key)

    def _learn(self, request: str, plan: Plan, strategy: RetrievalStrategy, score: float,
               hits: tuple[Hit, ...] = (), ctx_key: str | None = None) -> float:
        reward = reward_from_judgment(score, strategy, self.strategies)
        self.bandit.update(ctx_key or plan.intent.bucket, strategy, reward)
        self.runtime.estimator.observe(plan, Trace(
            plan_id=plan.id, goal_text=request, actual_tokens=strategy.final_k * 200,
            verification_passed=score >= 0.6))
        self._log_calibration(plan, strategy, score, hits)
        return reward

    def _log_calibration(self, plan: Plan, strategy: RetrievalStrategy, score: float,
                         hits: tuple[Hit, ...]) -> None:
        """Append the (per-hit raw score, relevance label) row the calibration map is fit
        from. Per-passage labels are used when a passage judge annotated the hits (h.meta
        ['rel']); otherwise the per-query judge score is the weak label (fit() falls back)."""
        if self.calib_log is None or not hits:
            return
        rows = [{"chunk_id": h.chunk_id, "score": round(float(h.score), 6),
                 "rel": h.meta.get("rel")} for h in hits]
        try:
            self.calib_log.append(strategy.method, plan.intent.bucket, score, rows)
        except Exception:
            pass   # logging must never break a chat turn

    def record_judgment(self, request: str, score: float) -> float:
        """Feed a judged retrieval-quality score (0..1) back — used for the offline
        heuristic/LLM cold-start BOOTSTRAP. Prefer record_feedback (the app's native
        implicit signal) for online learning."""
        key = self._key(request)
        entry = self._pending.pop(key, None) or self._last.get(key)
        if entry is None:
            return 0.0
        plan, strategy, hits, ctx_key = entry
        return self._learn(request, plan, strategy, score, hits, ctx_key)

    def handle(self, request: str, judge: Judge | None = None) -> tuple[ChatContext, float, float]:
        """Retrieve → judge → learn, in one call. Returns (context, judged_score, reward)."""
        ctx = self.retrieve(request)
        self.annotate_relevance(request, ctx.hits)   # per-passage labels for calibration (opt-in)
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
