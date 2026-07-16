"""redevops-rag × Context Runtime — the second tenant: tune retrieval, not just plan it.

Where the sidekick tenant chose among discrete *recall strategies*, redevops-rag's
decision points are continuous-ish *numeric knobs* on ``hybrid_search``:

    pool · limit · vector_threshold · recency_half_life_days
    keyword_boost_per_term · keyword_boost_cap · reranker(on/off)

Context Runtime picks a config per query intent and learns which one maximizes **retrieval
quality per unit cost** — the efficiency thesis: the *cheapest* config that's still
good enough, not the most elaborate one. Same wrap as sidekick (plan → execute →
observe → learn), same shared bandit; only the arms and the reward differ.

Real path needs the ``[rag]`` extra (redevops-rag → torch). The optimizer/learning is
exercised offline; ``examples/rag_tuning.py`` proves the loop with a simulated quality
signal, exactly as the sidekick harness does.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..runtime.runtime import ContextRuntime
from ..types import Goal, Plan, Trace
from .bandit import EpsilonGreedyBandit

# ──────────────────────────── the arms: retrieval configs ────────────────────────────


@dataclass(frozen=True)
class RetrievalConfig:
    """A point in redevops-rag's ``hybrid_search`` parameter space (the bandit arm)."""

    pool: int = 50                      # candidate pool before fuse/boost
    limit: int = 8                      # final_k returned
    vector_threshold: float = 0.4
    recency_half_life_days: float = 90.0
    keyword_boost_per_term: float = 0.05
    keyword_boost_cap: float = 1.5
    rerank: bool = False                # the expensive cross-encoder stage
    diver: bool = False                 # the DIVER+ReasonIR temporal reasoning arm
                                        # (LLM query-expansion + listwise rerank; strong
                                        # on reasoning-intensive/temporal queries, dear)

    @property
    def key(self) -> str:
        return (f"p{self.pool}:k{self.limit}:vt{self.vector_threshold}:"
                f"hl{self.recency_half_life_days:g}:kb{self.keyword_boost_per_term:g}:"
                f"{'diver' if self.diver else ('rr' if self.rerank else 'norr')}")

    def kwargs(self) -> dict:
        """The keyword args to hand redevops-rag's ``hybrid_search``."""
        return {
            "pool": self.pool, "limit": self.limit, "vector_threshold": self.vector_threshold,
            "recency_half_life_days": self.recency_half_life_days,
            "keyword_boost_per_term": self.keyword_boost_per_term,
            "keyword_boost_cap": self.keyword_boost_cap,
        }

    # rough relative cost of running this config (latency proxy: pool size + rerank +
    # DIVER's extra LLM calls for query expansion and listwise reranking)
    def cost_units(self) -> float:
        return self.pool / 50.0 + (1.5 if self.rerank else 0.0) + (3.0 if self.diver else 0.0)


# A small, sensible arm set spanning cheap→thorough. Tune/extend per corpus.
DEFAULT_ARMS: tuple[RetrievalConfig, ...] = (
    RetrievalConfig(pool=20, limit=5, vector_threshold=0.5, rerank=False),   # cheap/precise
    RetrievalConfig(pool=50, limit=8, vector_threshold=0.4, rerank=False),   # the library default
    RetrievalConfig(pool=50, limit=8, vector_threshold=0.3, rerank=True),    # thorough + rerank
    RetrievalConfig(pool=100, limit=12, vector_threshold=0.3, rerank=True),  # max recall
    RetrievalConfig(pool=30, limit=8, recency_half_life_days=14.0, rerank=False),  # recency-biased
    RetrievalConfig(pool=25, limit=8, diver=True),  # DIVER+ReasonIR temporal reasoning arm
)

COST_LAMBDA = 0.15   # how much efficiency (cheapness) trades against quality in the reward


# ──────────────────────────── quality metrics + reward ────────────────────────────


def reciprocal_rank(hit_ids: list[str], relevant_ids: set[str]) -> float:
    """MRR contribution: 1/rank of the first relevant hit (0 if none). A clean,
    label-only quality signal when you have gold relevance."""
    for i, h in enumerate(hit_ids, 1):
        if h in relevant_ids:
            return 1.0 / i
    return 0.0


def reward_from_quality(quality: float, config: RetrievalConfig) -> float:
    """Quality minus a normalized efficiency penalty → 'cheapest config that's good
    enough'. quality ∈ [0,1]; cost normalized by the most expensive arm."""
    max_cost = max(a.cost_units() for a in DEFAULT_ARMS)
    cost_norm = config.cost_units() / max_cost if max_cost else 0.0
    return round(max(0.0, quality - COST_LAMBDA * cost_norm), 4)


# ──────────────────────────── the tenant ────────────────────────────


def _rag_bandit(epsilon: float = 0.15) -> EpsilonGreedyBandit:
    return EpsilonGreedyBandit(DEFAULT_ARMS, epsilon=epsilon)


class ContextRuntimeRetrieverTuner:
    """Wrap a redevops-rag ``RAG`` so Context Runtime tunes its knobs per query intent.

    Usage (real):
        from redevops_rag import RAG
        tuner = ContextRuntimeRetrieverTuner(rag=RAG(db_path="vault.duckdb", use_reranker=True))
        hits = tuner.search("how do we rotate API keys")
        ...                                   # measure quality from your eval/labels
        tuner.record_outcome("how do we rotate API keys", quality=0.8, latency_s=0.4)
    """

    def __init__(self, rag=None, runtime: ContextRuntime | None = None,
                 bandit: EpsilonGreedyBandit | None = None, reason_llm=None):
        self.rag = rag                                  # redevops_rag.RAG (or None offline)
        self.runtime = runtime or ContextRuntime.default([])
        self.bandit = bandit or _rag_bandit()
        self.reason_llm = reason_llm                    # callable(system,user)->str; enables the DIVER arm
        self._pending: dict[str, tuple[Plan, RetrievalConfig]] = {}

    def choose(self, query: str) -> RetrievalConfig:
        """Classify the query and let the bandit pick a config (no execution)."""
        plan = self.runtime.plan(Goal(text=query))
        cfg = self.bandit.select(plan.intent.bucket)
        self._pending[self._key(query)] = (plan, cfg)
        return cfg

    def search(self, query: str) -> list:
        """Pick a config and run redevops-rag's hybrid_search with it. Returns the raw
        redevops-rag hit dicts (so the caller can score quality however they like)."""
        cfg = self.choose(query)
        if self.rag is None:
            raise RuntimeError("no RAG bound — install 'context_runtime[rag]' and pass rag=RAG(...)")
        reranker = getattr(self.rag, "reranker", None) if cfg.rerank else None
        # DIVER arm: the merged ReasonIR+DIVER temporal reasoning retriever (needs reason_llm;
        # falls back to hybrid on cold start so the arm is always safe to select).
        if cfg.diver and self.reason_llm is not None:
            from redevops_rag.retrieve import diver_search  # type: ignore
            return diver_search(self.rag.store, query, self.reason_llm,
                                limit=cfg.limit, pool=cfg.pool, reranker=reranker)
        from redevops_rag.retrieve import hybrid_search  # type: ignore
        return hybrid_search(self.rag.store, query, reranker=reranker, **cfg.kwargs())

    def record_outcome(self, query: str, *, quality: float | None = None,
                       relevant_ids: set[str] | None = None, hit_ids: list[str] | None = None,
                       latency_s: float = 0.0, cost_usd: float = 0.0) -> float:
        """Feed retrieval quality back. Provide ``quality`` directly, or ``relevant_ids``
        + ``hit_ids`` to compute MRR. Updates the bandit AND calibrates the cost model."""
        key = self._key(query)
        if key not in self._pending:
            return 0.0
        plan, cfg = self._pending.pop(key)
        if quality is None:
            quality = reciprocal_rank(hit_ids or [], relevant_ids or set())
        reward = reward_from_quality(quality, cfg)
        self.bandit.update(plan.intent.bucket, cfg, reward)
        # calibrate the cost model on observed latency/cost (quality ≈ accuracy proxy)
        trace = Trace(plan_id=plan.id, goal_text=query, actual_cost_usd=cost_usd,
                      actual_latency_seconds=latency_s, actual_tokens=cfg.limit,
                      verification_passed=quality >= 0.5)
        self.runtime.estimator.observe(plan, trace)
        return reward

    def policy(self) -> dict[str, str]:
        """The learned config per intent bucket (for inspection / EXPLAIN)."""
        return self.bandit.policy()

    @staticmethod
    def _key(query: str) -> str:
        import hashlib
        return hashlib.sha256(query.encode()).hexdigest()[:16]
