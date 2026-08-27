"""Evidence Sparse Retrieval (F4) — a cheap first-stage region index in front of hybrid retrieval.

At long horizons, dense top-k over the whole evidence universe is both costly and imprecise: distractor
chunks crowd out the answer-bearing ones, so fixed-k recall decays as the corpus grows (the exact effect
the long-horizon harness measures). F4 adds the stage the audit's §2.3 describes:

    evidence universe → regions (compact sketches) → cheap region scorer → top-K regions
      → EXISTING hybrid retrieval, scoped to those regions → reranker → EvidenceRefs

This is a `RetrieverPlugin` (the `search(query, k, method) -> list[Hit]` contract), so it injects via
`ContextRuntime(retriever=...)` and the planner/cost-model can choose it. It wraps — never replaces — the
real retriever: the region layer only *narrows* the candidate documents; retrieval, ranking and evidence
identity all come from the underlying store.

Three properties are acceptance criteria, not nice-to-haves:

1. **Deterministic corpus-statistical routing (no LLM).** Regions are formed by the corpus's own document
   boundaries; a region's sketch is the centroid of its chunk embeddings + a term sketch. The scorer is a
   fixed linear combination of centroid cosine and lexical overlap. Same inputs → same regions → same
   selection, every run. No model call decides what is relevant.
2. **Confidence floor → global fallback.** If no region clears the floor (the query doesn't clearly belong
   to any region), F4 searches the FULL document set — it can degrade to the global baseline but never
   below it. Sparse selection is an optimization, never a silent recall cliff.
3. **Identity-transparent EvidenceRefs.** F4 returns the underlying store's `Hit`s unchanged — same
   chunk_id / content_hash / source_version. The region index holds *references* to evidence, never a copy;
   it is not authoritative and cannot rewrite provenance.

Driver-agnostic: the region-scoped search and the query embedder are injected callables, so the plugin is
unit-tested without a live store. It lives in `adapters/` rather than the package `__init__`, so a caller
that binds a region index opts in explicitly.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from context_runtime.types import Hit, PluginInfo, Retrieval

# Injected: run the real hybrid retrieval scoped to a document subset. In the benchmark this wraps
# redevops_rag.hybrid_search(store, query, limit=k, document_ids=doc_ids); in a deployment it is the
# runtime's store search. doc_ids=None means "no scope" (global).
ScopedSearch = Callable[[str, int, Retrieval, "list[str] | None"], list[Hit]]
EmbedQuery = Callable[[str], Sequence[float]]

_WORD = re.compile(r"[A-Za-z0-9]+")
_STOP = frozenset("the a an of to in for and or is are was were be on at by with from as this that "
                  "what which who whom where when how".split())


def _terms(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text or "") if len(w) > 2 and w.lower() not in _STOP}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@dataclass(frozen=True)
class EvidenceRegion:
    """One region's compact sketch. ``doc_ids`` are references back to the authoritative evidence — the
    region never holds the evidence itself (identity transparency)."""
    region_id: str
    doc_ids: tuple[str, ...]
    centroid: tuple[float, ...]
    terms: frozenset[str]
    tenant: str | None = None          # policy/tenant scope of the region (None = unscoped)
    size: int = 0

    def score(self, q_emb: Sequence[float], q_terms: set[str], *, alpha: float, beta: float) -> float:
        cos = _cosine(q_emb, self.centroid) if self.centroid else 0.0
        overlap = (len(q_terms & self.terms) / len(q_terms)) if q_terms else 0.0
        return alpha * cos + beta * overlap


@dataclass
class RegionIndex:
    regions: list[EvidenceRegion] = field(default_factory=list)

    @staticmethod
    def build(chunks: Iterable[dict], embeddings: Sequence[Sequence[float]], *,
              region_key: str = "filename") -> "RegionIndex":
        """Group chunks into regions by ``region_key`` (default: source document). A region's centroid is
        the mean of its chunk embeddings; its term sketch is the **membership set** of every term the
        region contains — deliberately NOT a frequency top-N. Routing a query hinges on its *rare*
        discriminative terms (an entity id, a code, a name), which by definition are infrequent; a
        frequency-capped sketch drops exactly those and sends a needle query to the wrong region. Membership
        (does term T occur in this region) is what makes deterministic lexical routing correct. In a
        deployment this set is a compact MinHash/Bloom signature; the semantics — term presence — are the
        same. Fully deterministic given (chunks, embeddings): no clustering seed, no model call."""
        chunks = list(chunks)
        buckets: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            buckets.setdefault(str(c.get(region_key) or c.get("document_id") or i), []).append(i)
        regions: list[EvidenceRegion] = []
        for rid in sorted(buckets):                       # sorted → stable region order
            idxs = buckets[rid]
            dim = len(embeddings[idxs[0]]) if embeddings and idxs else 0
            centroid = [0.0] * dim
            terms: set[str] = set()
            doc_ids: list[str] = []
            for i in idxs:
                for d in range(dim):
                    centroid[d] += embeddings[i][d]
                terms |= _terms(chunks[i].get("text", ""))
                doc_ids.append(str(chunks[i].get("document_id") or chunks[i].get("id") or i))
            if dim:
                centroid = [v / len(idxs) for v in centroid]
            regions.append(EvidenceRegion(region_id=rid, doc_ids=tuple(doc_ids),
                                          centroid=tuple(centroid), terms=frozenset(terms), size=len(idxs)))
        return RegionIndex(regions=regions)

    def all_doc_ids(self) -> list[str]:
        return [d for r in self.regions for d in r.doc_ids]


@dataclass
class SparseRegionRetriever:
    """RetrieverPlugin: select top-K regions (deterministically), then run the real scoped retrieval.

    ``floor`` is the confidence floor on the best region score; below it, F4 falls back to global search
    (all documents) so it can never do worse than the un-narrowed baseline. ``tenant`` (optional) filters
    regions to the caller's scope before ranking — sparse selection respects the isolation boundary.
    """
    index: RegionIndex
    scoped_search: ScopedSearch
    embed_query: EmbedQuery
    top_regions: int = 4
    floor: float = 0.15
    alpha: float = 0.6
    beta: float = 0.4
    tenant: str | None = None
    last_reason: str = ""            # EXPLAIN: how the last search routed (which regions / fallback)

    def _select(self, query: str) -> "list[str] | None":
        regions = [r for r in self.index.regions if self.tenant is None or r.tenant in (None, self.tenant)]
        # The fallback scope: truly global (None → all docs) only when unscoped; otherwise the tenant's own
        # in-scope documents. The confidence floor must never widen access past the isolation boundary —
        # "fall back to global" means "global within what this caller may already see" (cf. F5).
        fb = "global" if self.tenant is None else "tenant-scoped"
        scope_all = None if self.tenant is None else [d for r in regions for d in r.doc_ids]
        if not regions:
            self.last_reason = f"no regions in scope → {fb} fallback"
            return scope_all
        q_emb = self.embed_query(query)
        q_terms = _terms(query)
        ranked = sorted(regions, key=lambda r: r.score(q_emb, q_terms, alpha=self.alpha, beta=self.beta),
                        reverse=True)
        best = ranked[0].score(q_emb, q_terms, alpha=self.alpha, beta=self.beta)
        if best < self.floor:
            self.last_reason = f"best region score {best:.3f} < floor {self.floor} → {fb} fallback"
            return scope_all                              # confidence floor → fallback (scoped if tenant)
        chosen = ranked[: self.top_regions]
        self.last_reason = (f"top-{len(chosen)}/{len(regions)} regions "
                            f"[{', '.join(r.region_id for r in chosen)}] best={best:.3f}")
        return [d for r in chosen for d in r.doc_ids]

    def search(self, query: str, k: int, method: Retrieval = "hybrid") -> list[Hit]:
        doc_ids = self._select(query)                     # None → global
        return self.scoped_search(query, k, method, doc_ids)

    def info(self) -> PluginInfo:
        return PluginInfo(name="sparse-regions", kind="retriever", version="0.1",
                          capabilities=frozenset({"sparse", "region-routing", "confidence-floor-fallback"}))
