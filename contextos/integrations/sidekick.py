"""sidekick × ContextOS — optimize sidekick's self-learning via context management.

sidekick learns *skills* (object level). This makes ContextOS learn *which skills/context
to surface, and how much budget to spend* (meta level), using sidekick's own acceptance
metric as the reward. Two pieces:

  * ``ContextOSSkillStore`` — a DROP-IN for sidekick's ``SkillStore`` (same
    save/all/recall/record_use interface), but ``recall()`` ranks via ContextOS's
    planner + retriever instead of naive token-overlap, and ``record_outcome()`` feeds
    sidekick's ``SubtaskRecord`` back so the loop closes.
  * ``EpsilonGreedyBandit`` — the "baby bandit": picks a recall Strategy (retrieval
    method · bundle size · token budget) per task, keyed by intent bucket, and learns
    which strategy yields accepted-and-efficient runs. This is the v0.1-achievable
    stand-in for the v0.3 River contextual bandit; it shares the exact reward seam.

Insertion into sidekick (the "swap recall + learn" path) is two lines — see
``examples/sidekick_learning.py`` and ``integrations/sidekick_orchestrator.patch``.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..runtime.runtime import ContextRuntime
from ..types import Goal, Plan, Trace
from .bandit import EpsilonGreedyBandit  # shared fleet-pattern learning core (re-exported)

# ──────────────────────────── sidekick-compatible Skill ────────────────────────────


@dataclass
class Skill:
    """Mirror of sidekick.skills.Skill (same JSON shape, so files interop)."""

    name: str
    trigger: str = ""
    approach: str = ""
    acceptance_checks: list[str] = field(default_factory=list)
    uses: int = 0
    created_ts: float = field(default_factory=time.time)


# ──────────────────────────── the bandit arms ────────────────────────────


@dataclass(frozen=True)
class Strategy:
    """An arm: how to recall context for a task."""

    method: str            # "bm25" | "vector" | "hybrid" | "code"
    final_k: int           # how many skills/chunks to surface
    target_tokens: int     # context budget to spend

    @property
    def key(self) -> str:
        return f"{self.method}:{self.final_k}:{self.target_tokens}"


DEFAULT_ARMS: tuple[Strategy, ...] = (
    Strategy("bm25", 3, 1500),
    Strategy("vector", 3, 2000),
    Strategy("hybrid", 5, 3000),
    Strategy("hybrid", 8, 4000),     # the "bundle more, rerank" arm
    Strategy("code", 5, 3000),
)

TOKEN_REF = 8000   # context size above which efficiency reward decays to 0


def reward_from_outcome(accepted: bool, first_attempt: bool, tokens_total: int) -> float:
    """Composite reward in [0,1]: mostly acceptance, with an efficiency + no-retry bonus.

    Cheap context that still gets accepted on the first try is the optimum sidekick wants.
    """
    acc = 1.0 if accepted else 0.0
    eff = max(0.0, 1.0 - min(1.0, tokens_total / TOKEN_REF))
    first = 1.0 if (accepted and first_attempt) else 0.0
    return round(0.6 * acc + 0.2 * (acc * eff) + 0.2 * first, 4)


def _sidekick_bandit(epsilon: float = 0.15) -> EpsilonGreedyBandit:
    return EpsilonGreedyBandit(DEFAULT_ARMS, epsilon=epsilon)


# ──────────────────────────── the outcome sidekick reports ────────────────────────────


@dataclass(frozen=True)
class SubtaskOutcome:
    """The subset of sidekick.metrics.SubtaskRecord ContextOS needs as 'actuals'."""

    accepted: bool
    first_attempt: bool = False
    tokens_total: int = 0
    cost_usd: float = 0.0
    wall_ms: int = 0


# ──────────────────────────── the drop-in store ────────────────────────────


def _safe(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower() or "skill"


class ContextOSSkillStore:
    """Drop-in for sidekick.skills.SkillStore, backed by ContextOS recall + learning.

    Same surface (save/all/recall/record_use) so a one-line constructor swap at
    ``orchestrator.py:186`` is enough. ``record_outcome()`` is the second seam that
    closes the loop (call it where sidekick records a SubtaskRecord).
    """

    def __init__(self, skills_dir, runtime: ContextRuntime | None = None,
                 bandit: EpsilonGreedyBandit | None = None, reindex: bool = True):
        self.dir = Path(skills_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.runtime = runtime or ContextRuntime.default([])
        self.bandit = bandit or _sidekick_bandit()
        self._pending: dict[str, tuple[Plan, Strategy]] = {}
        if reindex:
            self.reindex()

    # ── corpus: skills become the retrieval corpus ──
    def reindex(self) -> None:
        docs = []
        for sk in self.all():
            docs.append({
                "chunk_id": _safe(sk.name), "filename": sk.name,
                "text": f"{sk.name}\n{sk.trigger}\n{sk.approach}\n" + "\n".join(sk.acceptance_checks),
                "created_at": None,
            })
        # the in-memory store is replaceable by the redevops-rag retriever ([rag] extra);
        # both satisfy RetrieverPlugin, so this line is the only place that knows.
        if hasattr(self.runtime.retriever, "docs"):
            self.runtime.retriever.docs = docs

    # ── SkillStore-compatible surface ──
    def _path(self, name: str) -> Path:
        return self.dir / f"{_safe(name)}.json"

    def save(self, skill) -> Path:
        p = self._path(skill.name)
        data = asdict(skill) if hasattr(skill, "__dataclass_fields__") else dict(skill)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.reindex()
        return p

    def all(self) -> list[Skill]:
        out: list[Skill] = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                out.append(Skill(**json.loads(p.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, TypeError):
                pass
        return out

    def record_use(self, name: str) -> None:
        p = self._path(name)
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                d["uses"] = int(d.get("uses", 0)) + 1
                p.write_text(json.dumps(d, indent=2), encoding="utf-8")
            except (json.JSONDecodeError, OSError):
                pass

    # ── the upgraded recall ──
    def recall(self, query: str, limit: int = 3) -> list[Skill]:
        """ContextOS-planned recall: classify the task, let the bandit pick a Strategy,
        retrieve via that method/budget, return the top skills."""
        plan = self.runtime.plan(Goal(text=query))
        strategy = self.bandit.select(plan.intent.bucket)
        self._pending[self._key(query)] = (plan, strategy)
        hits = self.runtime.retriever.search(query, k=max(limit, strategy.final_k), method=strategy.method)
        by_name = {s.name: s for s in self.all()}
        out: list[Skill] = []
        for h in hits[:limit]:
            sk = by_name.get(h.filename) or by_name.get(_safe(h.filename))
            if sk:
                out.append(sk)
        return out

    def assembled_context(self, query: str, limit: int = 5) -> str:
        """The richer integration: a citation-numbered, budget-bounded context block to
        inject into the worker prompt (replaces the plain skill_hint string)."""
        plan = self.runtime.plan(Goal(text=query))
        ctx = self.runtime.build_context(plan, Goal(text=query))
        return ctx.assembled_text

    # ── the second seam: close the loop ──
    def record_outcome(self, query: str, outcome: SubtaskOutcome) -> float:
        """Feed sidekick's acceptance/efficiency back. Updates the bandit AND calibrates
        the cost-model statistics (estimate-vs-actual). Returns the reward (for logging)."""
        key = self._key(query)
        if key not in self._pending:
            return 0.0
        plan, strategy = self._pending.pop(key)
        reward = reward_from_outcome(outcome.accepted, outcome.first_attempt, outcome.tokens_total)
        self.bandit.update(plan.intent.bucket, strategy, reward)
        trace = Trace(
            plan_id=plan.id, goal_text=query,
            actual_cost_usd=outcome.cost_usd,
            actual_latency_seconds=outcome.wall_ms / 1000.0,
            actual_tokens=outcome.tokens_total,
            verification_passed=outcome.accepted,
        )
        self.runtime.estimator.observe(plan, trace)
        return reward

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()[:16]


def make_skill_store(skills_dir, use_contextos: bool = True, **kw):
    """Factory for the orchestrator.py:186 swap. ``use_contextos=False`` returns sidekick's
    own SkillStore so the change is a safe, reversible A/B toggle."""
    if use_contextos:
        return ContextOSSkillStore(skills_dir, **kw)
    from sidekick.skills import SkillStore  # type: ignore
    return SkillStore(Path(skills_dir))
