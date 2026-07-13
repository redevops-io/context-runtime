"""Temporal / bi-temporal retrieval — "what changed, and when?" (Whitepaper v3, forthcoming retrieval).

The next retrieval method: a **bi-temporal** store (Graphiti/Zep lineage) where every fact carries two
time axes —

  • **valid time**  (``valid_from`` / ``valid_to``): when the fact is true in the world;
  • **transaction time** (``recorded_at``): when the system learned it.

That lets the planner answer point-in-time questions the other methods can't: the state *as of* a past
date, and *as it was known* at a past date (so a late-arriving correction doesn't rewrite history). It is
densest in conversational corpora (chat/tickets) where facts are constantly revised.

Dependency-free (stdlib + the ``RetrieverPlugin`` seam), so it drops into the planner as one more
routable capability (``method="temporal"``). Times are compared as plain ISO strings
(``"2026-01-15"``): ``""`` = the beginning of time, ``None`` on ``valid_to`` = still valid.

Three bindings, same seam (the router can't tell them apart):
  * ``TemporalDocumentRetriever`` — the **DEFAULT**. Document retrieval (BM25) over the RAW turns +
    a bi-temporal time layer. Non-lossy → matches document recall while adding point-in-time, which
    is why it wins recall-sensitive regimes (LongMemEval-oracle) that Graphiti's lossy extraction caps.
  * ``TemporalStore`` — the stdlib bi-temporal FACT store (subject/predicate/object), for structured
    fact histories.
  * ``GraphitiTemporalRetriever`` — an **OPTIONAL** binding for its real strength: point-in-time
    reasoning over a LARGE, REVISED history (validate on the full haystack, not the oracle).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ..types import Hit, PluginInfo, Retrieval

_STOP = {"the", "a", "an", "of", "to", "in", "for", "is", "was", "did", "who", "what", "when", "how",
         "owns", "own"}
# keep hyphen/underscore inside a token so entity ids ("auth-service") stay whole and don't collide
# with a shared fragment ("service") on unrelated subjects
_WORD = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) > 1 and t not in _STOP}


@dataclass
class TemporalFact:
    subject: str
    predicate: str
    obj: str
    valid_from: str = ""          # inclusive; "" = -inf
    valid_to: str | None = None   # exclusive; None = still valid (+inf)
    recorded_at: str = ""         # transaction time; "" = -inf
    meta: dict = field(default_factory=dict)

    def text(self) -> str:
        return f"{self.subject} {self.predicate} {self.obj}"

    def valid_at(self, at: str | None) -> bool:
        if at is None:                                   # "current" = not yet superseded
            return self.valid_to is None
        return self.valid_from <= at and (self.valid_to is None or at < self.valid_to)

    def known_at(self, known: str | None) -> bool:
        return known is None or self.recorded_at <= known


class TemporalStore:
    """An in-memory bi-temporal fact store exposing the RetrieverPlugin seam."""

    def __init__(self, facts: list[TemporalFact] | None = None):
        self._facts: list[TemporalFact] = list(facts or [])

    def add(self, subject: str, predicate: str, obj: str, *, valid_from: str = "",
            valid_to: str | None = None, recorded_at: str = "", meta: dict | None = None) -> "TemporalStore":
        self._facts.append(TemporalFact(subject, predicate, obj, valid_from, valid_to,
                                        recorded_at, meta or {}))
        return self

    def _hit(self, f: TemporalFact, score: float) -> Hit:
        return Hit(
            chunk_id=f"{f.subject}:{f.predicate}:{f.valid_from or '-'}",
            filename="temporal",
            text=f.text(),
            score=score,
            created_at=f.valid_from or None,
            source="temporal",
            meta={"subject": f.subject, "predicate": f.predicate, "object": f.obj,
                  "valid_from": f.valid_from, "valid_to": f.valid_to, "recorded_at": f.recorded_at,
                  **f.meta},
        )

    def _ranked(self, query: str, candidates: list[TemporalFact], k: int) -> list[Hit]:
        q = _tokens(query)
        scored: list[tuple[float, str, TemporalFact]] = []
        for f in candidates:
            overlap = len(q & _tokens(f.text()))
            if q and overlap == 0:
                continue
            score = overlap / (len(q) or 1)
            scored.append((score, f.valid_from, f))
        # best match first; ties → most-recently-valid first
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [self._hit(f, s) for s, _, f in scored[:k]]

    # ── the RetrieverPlugin seam ──
    def search(self, query: str, k: int = 5, method: Retrieval = "temporal") -> list[Hit]:
        """Current state: facts not yet superseded (``valid_to is None``), ranked by query match."""
        return self._ranked(query, [f for f in self._facts if f.valid_at(None)], k)

    def as_of(self, query: str, k: int = 5, *, at: str, known_at: str | None = None) -> list[Hit]:
        """Point-in-time query: facts valid at ``at`` (world time) and known at ``known_at`` (transaction
        time). ``known_at`` lets you ask "what did we believe on date X", ignoring later corrections."""
        return self._ranked(query, [f for f in self._facts
                                    if f.valid_at(at) and f.known_at(known_at)], k)

    def changes(self, query: str = "", *, since: str, until: str, k: int = 20) -> list[dict]:
        """"What changed, and when?" — facts that began or ended validity in ``[since, until)``."""
        q = _tokens(query)
        out: list[dict] = []
        for f in self._facts:
            if q and not (q & _tokens(f.text())):
                continue
            if since <= f.valid_from < until:
                out.append({"at": f.valid_from, "change": "began", "fact": f.text(),
                            "subject": f.subject, "predicate": f.predicate, "object": f.obj})
            if f.valid_to is not None and since <= f.valid_to < until:
                out.append({"at": f.valid_to, "change": "ended", "fact": f.text(),
                            "subject": f.subject, "predicate": f.predicate, "object": f.obj})
        out.sort(key=lambda c: c["at"])
        return out[:k]

    def info(self) -> PluginInfo:
        return PluginInfo(name="temporal", kind="retriever",
                          capabilities=frozenset({"temporal", "retrieval"}))


# ─────────────────────────────────────────────────────────────────────────────
# Optional real backend: Graphiti (bi-temporal knowledge graph over Neo4j).
#
# Exposes the SAME RetrieverPlugin seam as ``TemporalStore`` (search / as_of /
# changes / info), so ``HopRouterRetriever`` treats the two interchangeably —
# ``method="temporal"`` routes here when a Graphiti backend is wired, and falls
# back to the dependency-free ``TemporalStore`` otherwise. All ``graphiti_core``
# imports are lazy so this module keeps importing with no heavy deps installed.
#
# Entity/edge extraction uses the configured LLM (point ``llm_base_url`` at the
# 27B served endpoint); embeddings are computed locally (sentence-transformers),
# avoiding a second served model. Async graphiti calls are bridged to the sync
# seam through a private event loop.
# ─────────────────────────────────────────────────────────────────────────────
from datetime import datetime, timezone


def _as_utc(v: "str | datetime | None") -> datetime:
    """Coerce a datetime / date string into a tz-aware UTC datetime (Neo4j wants tz-aware).
    Accepts ISO plus a few common non-ISO shapes (e.g. LongMemEval's '2023/04/10 (Mon) 17:50');
    unparseable input falls back to 'now' rather than raising."""
    if v is None or v == "":
        return datetime.now(timezone.utc)
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    s = str(v).strip()
    dt = None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M", "%Y/%m/%d", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, fmt); break
            except ValueError:
                continue
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _make_graphiti_clients(embed_model: str):
    """Build ``(embedder, cross_encoder)`` as real graphiti subclasses — ``GraphitiClients``
    validates them with ``isinstance``, so duck-typing is rejected. Defined lazily so this
    module still imports with neither graphiti_core nor sentence-transformers installed.

    Embedder is CPU-pinned (venv torch lacks Blackwell sm_120 kernels, and it keeps the GPU
    free for the served LLMs). A 1024-dim model matches graphiti's default ``EMBEDDING_DIM``
    so the Neo4j vector index dimensions line up. Reranker preserves candidate order to avoid
    a per-search LLM rerank call.
    """
    from graphiti_core.embedder.client import EmbedderClient
    from graphiti_core.cross_encoder.client import CrossEncoderClient
    from sentence_transformers import SentenceTransformer

    class _STEmbedder(EmbedderClient):
        def __init__(self, name):
            self._m = SentenceTransformer(name, device="cpu")

        async def create(self, input_data):
            # Contract (matches graphiti's OpenAIEmbedder): return ONE flat vector (list[float]),
            # even when handed a single-element list. Batches go through create_batch.
            text = input_data if isinstance(input_data, str) else (
                (list(input_data)[0] if input_data else ""))
            return self._m.encode([text], normalize_embeddings=True)[0].tolist()

        async def create_batch(self, input_data_list):
            return [v.tolist() for v in self._m.encode(list(input_data_list), normalize_embeddings=True)]

    class _OrderReranker(CrossEncoderClient):
        async def rank(self, query, passages):
            n = len(passages) or 1
            return [(p, (n - i) / n) for i, p in enumerate(passages)]

    return _STEmbedder(embed_model), _OrderReranker()


class GraphitiTemporalRetriever:
    """Real bi-temporal retriever: graphiti-core's Neo4j KG behind the temporal slot.

    Drop-in replacement for :class:`TemporalStore` when the full Graphiti engine is
    wanted (LLM-extracted entities/edges, hybrid semantic+graph search, point-in-time
    ``as_of`` via Neo4j-side temporal filters) rather than the stdlib token store.
    """

    def __init__(self, *, neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j", neo4j_password: str = "graphiti-bench-pw",
                 llm_base_url: str, llm_model: str, llm_api_key: str = "sk-noauth",
                 embed_model: str = "BAAI/bge-large-en-v1.5",
                 embedder=None, cross_encoder=None, group_id: str = "bench",
                 hydrate_sources: bool = True):
        """``embedder`` / ``cross_encoder`` accept graphiti ``EmbedderClient`` / ``CrossEncoderClient``
        instances (CR plugin ethos — swap the engines). Left as ``None`` they default to a local
        CPU sentence-transformers embedder + order-preserving reranker (see ``_make_graphiti_clients``).

        ``hydrate_sources`` (G1, redevops fork): when the installed ``graphiti_core`` supports source
        hydration, retrieved edges are backfilled with their raw source turns so hits carry the
        non-lossy text behind each LLM-extracted fact. Auto-disables against an upstream build."""
        import asyncio
        from graphiti_core import Graphiti  # lazy
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        self._loop = asyncio.new_event_loop()
        self.group_id = group_id
        self.hydrate_sources = hydrate_sources
        self._ep_name: dict[str, str] = {}   # episode uuid → episode name (source id), for provenance
        cfg = LLMConfig(api_key=llm_api_key, model=llm_model, base_url=llm_base_url,
                        small_model=llm_model)
        if embedder is None or cross_encoder is None:
            d_emb, d_ce = _make_graphiti_clients(embed_model)
            embedder = embedder or d_emb
            cross_encoder = cross_encoder or d_ce
        self._g = Graphiti(
            neo4j_uri, neo4j_user, neo4j_password,
            llm_client=OpenAIGenericClient(config=cfg),
            embedder=embedder, cross_encoder=cross_encoder,
        )
        self._run(self._g.build_indices_and_constraints())

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    # ── ingest ──
    def index(self, episodes) -> int:
        """Ingest episodes (bi-temporal). Each: dict(body, reference_time, name?, source_description?)."""
        from graphiti_core.nodes import EpisodeType  # lazy
        n = 0
        for e in episodes:
            name = e.get("name", f"ep{n}")
            res = self._run(self._g.add_episode(
                name=name,
                episode_body=e["body"],
                source_description=e.get("source_description", ""),
                reference_time=_as_utc(e.get("reference_time")),
                source=EpisodeType.text,
                group_id=self.group_id,
            ))
            # provenance: remember which source (session) this episode came from, so retrieved
            # edges can be traced back to their source and scored at source granularity.
            ep = getattr(res, "episode", None)
            if ep is not None and getattr(ep, "uuid", None):
                self._ep_name[ep.uuid] = name
            n += 1
        return n

    def _search_edges(self, query: str, k: int, *, search_filter=None):
        """Run a Graphiti hybrid search, requesting source hydration (G1) when available. Falls back
        transparently to a plain search if the installed graphiti_core predates the redevops fork."""
        kw = {"num_results": k, "group_ids": [self.group_id]}
        if search_filter is not None:
            kw["search_filter"] = search_filter
        if self.hydrate_sources:
            try:
                return self._run(self._g.search(query, hydrate_sources=True, **kw))
            except TypeError:
                self.hydrate_sources = False   # upstream graphiti-core — no hydrate_sources kwarg
        return self._run(self._g.search(query, **kw))

    def _edge_to_hit(self, edge, rank: int, total: int) -> Hit:
        va = getattr(edge, "valid_at", None)
        iva = getattr(edge, "invalid_at", None)
        # provenance: map the edge's source episode uuids back to their source names (session ids),
        # so temporal retrieval can be scored at source granularity (comparable to doc-level recall).
        ep_uuids = getattr(edge, "episodes", None) or []
        sessions = [self._ep_name[u] for u in ep_uuids if u in self._ep_name]
        # G1 source hydration: prefer the raw source turns behind the fact (non-lossy). hydrated_text()
        # falls back to the extracted fact when the edge wasn't/couldn't be hydrated, so this is safe
        # on both fork and upstream builds. The extracted fact stays available in meta.
        fact = getattr(edge, "fact", "") or ""
        hydrate = getattr(edge, "hydrated_text", None)
        text = hydrate() if callable(hydrate) else fact
        return Hit(
            chunk_id=str(getattr(edge, "uuid", f"e{rank}")),
            filename="graphiti",
            text=text,
            score=(total - rank) / (total or 1),
            created_at=(va.isoformat() if va else None),
            source="temporal",
            meta={"name": getattr(edge, "name", ""),
                  "fact": fact,
                  "hydrated": callable(hydrate) and bool(getattr(edge, "source_episodes", None)),
                  "valid_at": (va.isoformat() if va else None),
                  "invalid_at": (iva.isoformat() if iva else None),
                  "source_sessions": sessions},
        )

    # ── the RetrieverPlugin seam ──
    def search(self, query: str, k: int = 5, method: Retrieval = "temporal") -> list[Hit]:
        edges = self._search_edges(query, k)
        return [self._edge_to_hit(e, i, len(edges)) for i, e in enumerate(edges)]

    def as_of(self, query: str, k: int = 5, *, at: "str | datetime",
              known_at: "str | datetime | None" = None) -> list[Hit]:
        """Point-in-time: edges valid at ``at`` (valid_from ≤ at < valid_to), Neo4j-side filtered."""
        from graphiti_core.search.search_filters import (  # lazy
            SearchFilters, DateFilter, ComparisonOperator)
        atdt = _as_utc(at)
        # inner list = OR: a fact is live at `at` when it hadn't been invalidated yet
        # (invalid_at > at) OR is still valid (invalid_at IS NULL). Without the null branch,
        # still-current facts (the common case) would be wrongly excluded.
        # SearchFilters nesting: outer list = OR-groups, inner list = AND within a group.
        # "not yet invalidated" = (invalid_at > at) OR (invalid_at IS NULL) → two OR-groups.
        sf = SearchFilters(
            valid_at=[[DateFilter(date=atdt, comparison_operator=ComparisonOperator.less_than_equal)]],
            invalid_at=[[DateFilter(date=atdt, comparison_operator=ComparisonOperator.greater_than)],
                        [DateFilter(comparison_operator=ComparisonOperator.is_null)]],
        )
        edges = self._search_edges(query, k, search_filter=sf)
        return [self._edge_to_hit(e, i, len(edges)) for i, e in enumerate(edges)]

    def changes(self, query: str = "", *, since: str, until: str, k: int = 20) -> list[dict]:
        """"What changed, and when?" — edges that began/ended validity in ``[since, until)``."""
        lo, hi = _as_utc(since), _as_utc(until)
        edges = self._run(self._g.search(query or "*", num_results=max(k * 3, 30),
                                         group_ids=[self.group_id]))
        out: list[dict] = []
        for e in edges:
            va, iva = getattr(e, "valid_at", None), getattr(e, "invalid_at", None)
            if va and lo <= va < hi:
                out.append({"at": va.isoformat(), "change": "began", "fact": getattr(e, "fact", "")})
            if iva and lo <= iva < hi:
                out.append({"at": iva.isoformat(), "change": "ended", "fact": getattr(e, "fact", "")})
        out.sort(key=lambda c: c["at"])
        return out[:k]

    def info(self) -> PluginInfo:
        return PluginInfo(name="graphiti", kind="retriever",
                          capabilities=frozenset({"temporal", "retrieval", "graph"}))

    def close(self):
        try:
            self._run(self._g.close())
        finally:
            self._loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT temporal binding: a bi-temporal DOCUMENT retriever (non-lossy).
#
# Retrieves the RAW turns/sessions (not LLM-extracted facts), so it matches
# document-retrieval recall while adding the two time axes that make it a
# *temporal* method — point-in-time (`valid_at`) and as-known (`recorded_at`).
# This is why it wins the recall-sensitive regime (e.g. LongMemEval-oracle),
# where Graphiti's lossy LLM extraction caps out at what it managed to extract:
# hydration can recover the turns behind FOUND edges, never the sessions the
# extractor missed. Graphiti (below) stays an OPTIONAL binding for its real
# strength — point-in-time reasoning over a LARGE, REVISED history.
#
# Engine: compact Okapi BM25 (dependency-free). A deployment may inject a hybrid
# (BM25 ⊕ dense) substrate for higher recall at scale by overriding `_rank`.
# Timestamps are compared as ISO strings ("2026-01-15"); "" = the beginning of
# time. Callers pass ISO `valid_at` / `reference_time`.
# ─────────────────────────────────────────────────────────────────────────────

_DOC_WORD = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_DOC_STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "is", "was", "are", "were", "did",
             "do", "does", "who", "what", "when", "where", "how", "why", "and", "or", "but", "with",
             "at", "by", "as", "it", "this", "that", "i", "you", "he", "she", "they", "we", "my",
             "your", "me", "am", "be", "been"}


def _doc_tokens(text: str) -> list[str]:
    return [t for t in _DOC_WORD.findall((text or "").lower()) if len(t) > 1 and t not in _DOC_STOP]


def _iso(v) -> str:
    """Coerce to a comparable ISO-ish string; None/'' = the beginning of time."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    fn = getattr(v, "isoformat", None)
    return fn() if callable(fn) else str(v)


@dataclass
class TemporalDoc:
    id: str
    text: str
    valid_at: str = ""       # when the content is true (session/event time); "" = -inf
    recorded_at: str = ""    # when the system learned it (transaction time); "" = -inf
    meta: dict = field(default_factory=dict)


class TemporalDocumentRetriever:
    """The default ``temporal`` binding: document retrieval + a bi-temporal time layer.

    Non-lossy — it returns raw turns, not LLM-extracted facts — so it matches document recall while
    answering point-in-time questions plain document retrieval can't (``as_of`` / ``known_at``).
    Same ``RetrieverPlugin`` seam as ``TemporalStore`` / ``GraphitiTemporalRetriever`` (search /
    as_of / changes / info), so the router can't tell them apart.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75):
        self._docs: list[TemporalDoc] = []
        self._k1, self._b = k1, b

    # ── ingest ──
    def index(self, items) -> int:
        """Index timestamped items. Each: dict(text|body, valid_at|reference_time, id?/name?,
        recorded_at?, meta?). Returns the total doc count."""
        for it in items:
            vt = it.get("valid_at")
            if vt is None:
                vt = it.get("reference_time")
            self._docs.append(TemporalDoc(
                id=str(it.get("id") or it.get("name") or f"doc{len(self._docs)}"),
                text=it.get("text") or it.get("body") or "",
                valid_at=_iso(vt),
                recorded_at=_iso(it.get("recorded_at")),
                meta=dict(it.get("meta") or {}),
            ))
        return len(self._docs)

    def add(self, text: str, *, valid_at="", recorded_at="", id=None, meta=None) -> "TemporalDocumentRetriever":
        self._docs.append(TemporalDoc(id=str(id or f"doc{len(self._docs)}"), text=text,
                                      valid_at=_iso(valid_at), recorded_at=_iso(recorded_at), meta=meta or {}))
        return self

    # ── Okapi BM25 over a candidate view (IDF computed over the point-in-time corpus) ──
    def _rank(self, query: str, candidates: list[TemporalDoc], k: int) -> list[tuple[TemporalDoc, float]]:
        q = set(_doc_tokens(query))
        if not q or not candidates:
            return []
        doc_toks = [_doc_tokens(d.text) for d in candidates]
        n = len(candidates)
        dls = [len(dt) for dt in doc_toks]
        avgdl = (sum(dls) / n) or 1.0
        df: dict[str, int] = {}
        for dt in doc_toks:
            for t in set(dt):
                df[t] = df.get(t, 0) + 1
        scored: list[tuple[TemporalDoc, float]] = []
        for i, dt in enumerate(doc_toks):
            tf: dict[str, int] = {}
            for t in dt:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * (f * (self._k1 + 1)) / (f + self._k1 * (1 - self._b + self._b * dls[i] / avgdl))
            if s > 0:
                scored.append((candidates[i], s))
        scored.sort(key=lambda ds: ds[1], reverse=True)
        return scored[:k]

    def _hit(self, d: TemporalDoc, score: float) -> Hit:
        return Hit(chunk_id=d.id, filename="temporal", text=d.text, score=round(score, 4),
                   created_at=d.valid_at or None, source="temporal",
                   meta={"valid_at": d.valid_at, "recorded_at": d.recorded_at, **d.meta})

    # ── the RetrieverPlugin seam ──
    def search(self, query: str, k: int = 5, method: Retrieval = "temporal") -> list[Hit]:
        """Relevance over all turns (the current view); each hit carries its time axes for the answer."""
        return [self._hit(d, s) for d, s in self._rank(query, self._docs, k)]

    def as_of(self, query: str, k: int = 5, *, at, known_at=None) -> list[Hit]:
        """Point-in-time: only turns valid at ``at`` (valid_at ≤ at) and, if given, known by
        ``known_at`` (recorded_at ≤ known_at) — the bi-temporal view document retrieval can't give."""
        at_s = _iso(at)
        known_s = _iso(known_at) if known_at is not None else None
        cands = [d for d in self._docs
                 if d.valid_at <= at_s and (known_s is None or d.recorded_at <= known_s)]
        return [self._hit(d, s) for d, s in self._rank(query, cands, k)]

    def changes(self, query: str = "", *, since: str, until: str, k: int = 20) -> list[dict]:
        """Turns that entered the record in ``[since, until)`` — "what came in, and when?"."""
        qt = set(_doc_tokens(query))
        out = []
        for d in self._docs:
            if since <= d.valid_at < until and (not qt or qt & set(_doc_tokens(d.text))):
                out.append({"at": d.valid_at, "change": "recorded", "id": d.id, "text": d.text[:200]})
        out.sort(key=lambda c: c["at"])
        return out[:k]

    def info(self) -> PluginInfo:
        return PluginInfo(name="temporal-doc", kind="retriever",
                          capabilities=frozenset({"temporal", "retrieval", "bm25", "point-in-time"}))
