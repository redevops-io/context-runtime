"""chat-memory × Context Runtime — a 3-index agent-memory tenant.

Elastic's "Atlas" agent memory keeps user-chat history in three indices tuned for
three recall modes; the same idea fits Context Runtime *better* than a fixed
three-index fan-out, because CR's whole job is deciding **which** recall mode to
use per turn — and learning it from outcome.

The three memory indices (recall MODES):

    recency   — the last N turns, ordered by time. Cheap. Right for follow-ups
                ("and then?", "keep going") where the relevant context is simply
                what was just said.
    semantic  — bag-of-words / embedding similarity to the query. Mid cost. Right
                for factual recall ("what did we decide about pricing?").
    entity    — turns that mention the same named entities as the query. Mid cost.
                Right for entity questions ("what's Alice's role?").

Storage-agnostic on purpose: the three "indices" are an ABSTRACTION, not a Postgres
feature. This module ships a dependency-free in-memory `ChatMemoryStore` (fine for
the offline benchmark and small histories); the same three methods map onto DuckDB
(FTS ⊕ VSS ⊕ a join) or Postgres (tsvector ⊕ pgvector ⊕ jsonb) behind a StorePlugin
— see `store_backend` below. The retriever implements the RetrieverPlugin surface
(search/info) so it drops straight into the runtime.

The tenant (`ChatMemoryTenant`) puts a per-bucket EpsilonGreedyBandit over the recall
modes: it learns which mode pays off for each query bucket, measured as
recall-value − read-cost. `examples/chat_memory.py` drives a 72-round offline
benchmark proving CR beats a fixed "query all three indices" baseline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from ..types import Hit, PluginInfo
from .bandit import EpsilonGreedyBandit

# ──────────────────────────── stored turns + the 3-index store ────────────────────────────


@dataclass(frozen=True)
class Turn:
    """One chat turn in memory."""

    turn_id: str
    role: str            # "user" | "assistant"
    text: str
    ts: float            # monotonic turn index or epoch seconds — larger = more recent
    entities: frozenset[str] = frozenset()


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


# Common sentence-initial / function words that get capitalized but aren't entities.
_ENTITY_STOP = frozenset({
    "the", "we", "for", "and", "our", "you", "your", "this", "that", "they", "it",
    "a", "an", "to", "of", "in", "on", "at", "with", "okay", "great", "let", "lets",
    "launch", "refunds", "onboarding",  # domain sentence-starters in the demo corpus
})


def extract_entities(text: str) -> frozenset[str]:
    """Cheap entity signal: capitalized words (names/products) + quoted terms, minus
    capitalized function/sentence-initial words. A real deployment swaps this for an NER
    pass; the shape (a set of tags per turn) is what the entity index joins on."""
    caps = set(re.findall(r"\b([A-Z][a-zA-Z0-9]{2,})\b", text))
    quoted = set(re.findall(r"[\"']([^\"']{2,})[\"']", text))
    return frozenset(w.lower() for w in (caps | quoted) if w.lower() not in _ENTITY_STOP)


def _as_hit(t: Turn, score: float) -> Hit:
    return Hit(chunk_id=t.turn_id, filename="chat", text=t.text, score=round(score, 4),
               created_at=str(t.ts), source=t.role, meta={"entities": sorted(t.entities)})


class ChatMemoryStore:
    """Dependency-free 3-index memory over chat turns. Each method is one recall MODE
    the runtime can choose among (recency | semantic | entity). Storage-agnostic — a
    DuckDB/Postgres StorePlugin implements the same three methods against real indices."""

    def __init__(self, turns: Iterable[Turn] | None = None):
        self.turns: list[Turn] = list(turns or [])

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)

    # -- index 1: recency --------------------------------------------------
    def _recency(self, k: int) -> list[Hit]:
        ordered = sorted(self.turns, key=lambda t: t.ts, reverse=True)
        return [_as_hit(t, score=1.0 - i / max(len(ordered), 1)) for i, t in enumerate(ordered[:k])]

    # -- index 2: semantic (token-set cosine; embedding-ready) -------------
    def _semantic(self, query: str, k: int) -> list[Hit]:
        q = set(_tokens(query))
        if not q:
            return []
        scored: list[tuple[float, Turn]] = []
        for t in self.turns:
            toks = set(_tokens(t.text))
            if not toks:
                continue
            overlap = len(q & toks)
            if overlap:
                scored.append((overlap / (len(q) ** 0.5 * len(toks) ** 0.5), t))
        scored.sort(key=lambda s: (-s[0], -s[1].ts))
        return [_as_hit(t, sc) for sc, t in scored[:k]]

    # -- index 3: entity ---------------------------------------------------
    def _entity(self, query: str, k: int) -> list[Hit]:
        qe = extract_entities(query) | frozenset(_tokens(query))
        scored: list[tuple[float, Turn]] = []
        for t in self.turns:
            hits = len(t.entities & qe)
            if hits:
                scored.append((float(hits), t))
        scored.sort(key=lambda s: (-s[0], -s[1].ts))
        return [_as_hit(t, sc) for sc, t in scored[:k]]

    def search(self, query: str, k: int, method: str = "semantic") -> list[Hit]:
        """RetrieverPlugin surface. `method` is the recall mode. 'all' fans out to the
        three indices and RRF-fuses them (the fixed full-bundle baseline)."""
        if method == "recency":
            return self._recency(k)
        if method == "semantic":
            return self._semantic(query, k)
        if method == "entity":
            return self._entity(query, k)
        if method == "all":
            return _rrf(self._recency(k), self._semantic(query, k), self._entity(query, k), k=k)
        raise ValueError(f"unknown recall mode {method!r}")

    def info(self) -> PluginInfo:
        return PluginInfo(name="chat-memory", kind="retriever", version="0.1",
                          capabilities=frozenset({"recency", "semantic", "entity", "all"}))


def _rrf(*ranked: list[Hit], k: int, c: int = 60) -> list[Hit]:
    """Reciprocal-rank fusion (mirrors adapters.store_semantic._rrf_fuse)."""
    score: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for hits in ranked:
        for rank, h in enumerate(hits):
            score[h.chunk_id] = score.get(h.chunk_id, 0.0) + 1.0 / (c + rank + 1)
            best.setdefault(h.chunk_id, h)
    fused = sorted(best.values(), key=lambda h: (-score[h.chunk_id], h.chunk_id))
    return fused[:k] if k > 0 else fused


# ──────────────────────────── recall modes (bandit arms) ────────────────────────────


@dataclass(frozen=True)
class RecallMode:
    """One memory-recall strategy the tenant can choose. `methods` are the indices it
    queries; cost grows with how many indices it touches (each index is a real read)."""

    name: str
    methods: tuple[str, ...]
    k: int

    @property
    def key(self) -> str:
        return self.name

    def cost_units(self) -> float:
        # semantic (embedding read) is the priciest single index; entity (a tag join)
        # is cheaper, so when a query's entity is named the tenant learns to prefer the
        # entity index over semantic for equal recall.
        per = {"recency": 1.0, "semantic": 2.5, "entity": 1.2}
        return round(sum(per[m] for m in self.methods), 3)


# One decisive single-index mode per bucket, plus the full fan-out (baseline).
RECALL_MODES: tuple[RecallMode, ...] = (
    RecallMode("recency", ("recency",), k=5),
    RecallMode("semantic", ("semantic",), k=5),
    RecallMode("entity", ("entity",), k=5),
    RecallMode("all", ("recency", "semantic", "entity"), k=8),
)
FULL_BUNDLE = RECALL_MODES[-1]  # the "query all three indices" baseline


def memory_bucket(query: str) -> str:
    """Classify a turn into the bucket whose decisive recall mode CR should learn."""
    q = query.lower().strip()
    if re.search(r"\b(and then|then what|keep going|continue|after that|next|go on)\b", q) or len(_tokens(q)) <= 3:
        return "followup"          # decisive mode: recency
    if re.search(r"\b(who|whose|role|title|owner|contact|email|assigned)\b", q):
        return "entity"            # decisive mode: entity
    return "factual"               # decisive mode: semantic


# ──────────────────────────── the tenant (per-bucket learner) ────────────────────────────


@dataclass
class ChatMemoryTenant:
    """Per-bucket EpsilonGreedyBandit over the recall modes. Learns which memory index
    to read per query bucket, from reward = recall-value − read-cost."""

    epsilon: float = 0.15
    seed: int = 7
    _bandit: EpsilonGreedyBandit = field(init=False)

    def __post_init__(self) -> None:
        self._bandit = EpsilonGreedyBandit(RECALL_MODES, epsilon=self.epsilon, seed=self.seed)

    def choose(self, query: str) -> RecallMode:
        return self._bandit.select(memory_bucket(query))

    def record_outcome(self, query: str, mode: RecallMode, reward: float) -> None:
        self._bandit.update(memory_bucket(query), mode, reward)

    def record_feedback(self, query: str, mode: RecallMode, helpful: bool,
                        value: float = 3.0) -> None:
        """LIVE reward from a REAL signal — a thumbs-up / task-success / the user not having
        to repeat themselves — rather than the offline benchmark's simulated one. reward =
        value if the recalled memory helped, else 0, minus the mode's read cost (same shape
        as the benchmark, so a deployment learns the recall policy from actual outcomes).

        Wire it per turn:  mode = tenant.choose(q); hits = store.search(q, mode.k, mode.methods[0])
        ... answer ... then on the user's reaction: tenant.record_feedback(q, mode, helpful=thumbs_up)."""
        self.record_outcome(query, mode, (value if helpful else 0.0) - mode.cost_units())

    def policy(self) -> dict[str, str]:
        return self._bandit.policy()
