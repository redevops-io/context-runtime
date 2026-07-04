"""Two-stage cost-gated fusion retrieval.

The expensive part of modern RAG is *not* the first-pass recall — it is the second pass: a
cross-encoder rerank, or (for multimodal) a VLM that actually looks at each candidate page.
Running it on every candidate is what blows the latency/$ budget. DSpark's insight, already
shipped here as ``scheduler/load_aware.size_expensive_stage``, is to size that expensive stage
by *expected accepted relevance* and current load — run it deep when the prefix is confident and
the engine is idle, prune it hard when it isn't. This module wires that gate into a concrete
retrieve → fuse → (gated) rerank pipeline so the reuse is a real component, not just a helper.

    Stage 1 (cheap, broad):  run N base methods, reciprocal-rank-fuse the union.
    ── gate ──               calibrate the fused scores → P(relevant); size_expensive_stage
                              picks how deep the rerank runs (or drops it entirely under load).
    Stage 2 (expensive):     rerank ONLY the admitted prefix with the injected reranker.

The reranker is injected (a cross-encoder, an LLM, or a VLM for page images), so this is fully
testable without a model and the same gate serves text and visual reranking. Everything degrades
to "fuse and return" when no reranker/calibration is supplied ⇒ safe to adopt incrementally.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..types import Hit, PluginInfo, Retrieval
from .load_aware import SizingDecision, size_expensive_stage


def rrf_fuse(ranked_lists: list[list[Hit]], k: int, c: int = 60) -> list[Hit]:
    """Reciprocal-rank fusion over the union of several ranked hit lists.

    Order-only fusion (RRF) is the right stage-1 combiner precisely because raw scores are not
    comparable across methods (the whole reason calibration exists downstream). Keeps the best
    representative Hit per chunk and rewrites its score to the fused RRF weight.
    """
    agg: dict[str, float] = {}
    rep: dict[str, Hit] = {}
    for ranked in ranked_lists:
        for rank, h in enumerate(ranked):
            agg[h.chunk_id] = agg.get(h.chunk_id, 0.0) + 1.0 / (c + rank + 1)
            if h.chunk_id not in rep:
                rep[h.chunk_id] = h
    fused = sorted(agg.items(), key=lambda kv: -kv[1])[: max(k, 0) or len(agg)]
    out: list[Hit] = []
    for cid, w in fused:
        h = rep[cid]
        out.append(Hit(chunk_id=h.chunk_id, filename=h.filename, text=h.text,
                       score=round(w, 6), created_at=h.created_at, source=h.source,
                       meta=dict(h.meta)))
    return out


@dataclass(frozen=True)
class TwoStageResult:
    hits: tuple[Hit, ...]
    decision: SizingDecision
    reranked: bool
    stage1_n: int


class TwoStageRetriever:
    """Retrieve → RRF-fuse → cost-gated rerank, exposed as a normal retriever.

    ``base`` is any retriever with ``.search(query, k, method)``. ``stage1_methods`` are the
    cheap recall methods fused in stage 1. ``reranker(query, hits) -> list[Hit]`` is the
    expensive stage (cross-encoder / LLM / VLM); it is called on the ADMITTED prefix only.
    ``calibration`` (a CalibrationMap) maps fused scores → P(relevant) for the gate; without
    one the gate uses the fused scores directly as a monotone proxy.
    """

    def __init__(self, base, *, stage1_methods: tuple[str, ...] = ("bm25", "vector"),
                 reranker=None, calibration=None, cost_profile=None,
                 fanout_k: int = 20, source: str = "two_stage"):
        self.base = base
        self.stage1_methods = tuple(stage1_methods)
        self.reranker = reranker
        self.calibration = calibration
        self.cost_profile = cost_profile
        self.fanout_k = int(fanout_k)
        self.source = source

    def index(self, path: str) -> dict:
        return self.base.index(path) if hasattr(self.base, "index") else {}

    # ── the gate: how deep does the expensive stage run for THIS request? ──
    def _probs(self, hits: list[Hit]) -> list[float]:
        if self.calibration is not None:
            # fused hits are method-agnostic; calibrate on the fusion channel, fall back to
            # any per-method map the fused score happens to key under.
            return [round(self.calibration.apply("hybrid", float(h.score)), 6) for h in hits]
        # no calibration → min-max the fused scores into a monotone [0,1] confidence proxy so
        # the survival-product gate still has something ordered to admit against.
        if not hits:
            return []
        xs = [float(h.score) for h in hits]
        lo, hi = min(xs), max(xs)
        span = (hi - lo) or 1.0
        return [round((x - lo) / span, 6) for x in xs]

    def retrieve(self, query: str, k: int = 5, *, load_band: str = "lo",
                 rerank: bool = True, max_latency_seconds: float | None = None) -> TwoStageResult:
        # ── stage 1: broad recall, RRF-fused ──
        runs: list[list[Hit]] = []
        for m in self.stage1_methods:
            try:
                runs.append(list(self.base.search(query, self.fanout_k, m)))
            except Exception:
                continue
        fused = rrf_fuse(runs, k=self.fanout_k)
        stage1_n = len(fused)
        if not fused:
            return TwoStageResult((), SizingDecision(0, False, 0.0, "no-candidates"), False, 0)

        # ── the cost gate: size the expensive stage by survival product + load ──
        probs = self._probs(fused)
        decision = size_expensive_stage(
            probs, load_band=load_band, requested_k=min(k, len(fused)),
            requested_rerank=rerank and self.reranker is not None,
            cost_profile=self.cost_profile, max_latency_seconds=max_latency_seconds)
        admitted = fused[: decision.final_k]

        # ── stage 2: rerank ONLY the admitted prefix (the expensive pass) ──
        reranked = False
        if decision.rerank and self.reranker is not None and admitted:
            try:
                admitted = list(self.reranker(query, admitted))[: decision.final_k]
                reranked = True
            except Exception:
                pass
        return TwoStageResult(tuple(admitted), decision, reranked, stage1_n)

    def search(self, query: str, k: int, method: Retrieval = "hybrid",
               *, load_band: str = "lo") -> list[Hit]:
        """Retriever-shaped entrypoint (drops the sizing metadata). ``method`` is accepted for
        interface parity but the pipeline always fuses its own ``stage1_methods``."""
        return list(self.retrieve(query, k, load_band=load_band).hits)

    def info(self) -> PluginInfo:
        return PluginInfo(name="two_stage_retriever", kind="retriever",
                          capabilities=frozenset({"search", "fusion", "rerank", "cost-gated"}))
