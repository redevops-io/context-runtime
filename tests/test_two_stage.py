"""Phase 5 — two-stage cost-gated fusion: RRF stage-1, sizer-gated stage-2 rerank."""
from __future__ import annotations

from context_runtime.scheduler.two_stage import TwoStageRetriever, rrf_fuse
from context_runtime.types import Hit


def _h(cid, score=1.0, text="t"):
    return Hit(chunk_id=cid, filename=f"{cid}.txt", text=text, score=score)


def test_rrf_fuse_rewards_agreement():
    a = [_h("x"), _h("y"), _h("z")]        # x best in run A
    b = [_h("y"), _h("x"), _h("w")]        # y best in run B, x second
    fused = rrf_fuse([a, b], k=4)
    ids = [h.chunk_id for h in fused]
    # x and y appear high in both → they top the fused list ahead of singletons z/w
    assert set(ids[:2]) == {"x", "y"}
    assert set(ids) == {"x", "y", "z", "w"}


class _Base:
    """Per-method ranked lists; the two relevant chunks (r0,r1) rank first under 'vector'."""
    def search(self, query, k, method):
        if method == "vector":
            return [_h("r0", 0.9), _h("r1", 0.8), _h("n0", 0.2), _h("n1", 0.1)][:k]
        return [_h("n0", 5.0), _h("r0", 4.0), _h("n1", 3.0), _h("r1", 2.0)][:k]  # bm25 big scale


def test_gate_trims_under_load_and_reranks_only_admitted():
    seen = {}

    def reranker(query, hits):
        seen["n"] = len(hits)                       # how many the expensive stage actually saw
        return list(reversed(hits))                 # deterministic reorder to prove it ran

    ret = TwoStageRetriever(_Base(), stage1_methods=("bm25", "vector"), reranker=reranker)
    # idle: rerank runs on the admitted prefix
    lo = ret.retrieve("q", k=4, load_band="lo")
    assert lo.reranked and lo.decision.final_k >= 1
    assert seen["n"] == lo.decision.final_k         # stage-2 saw ONLY the admitted prefix
    # busy: the sizer drops rerank entirely (never enable a pass the load can't afford)
    seen.clear()
    hi = ret.retrieve("q", k=4, load_band="hi")
    assert hi.reranked is False and "n" not in seen
    assert hi.decision.final_k <= lo.decision.final_k   # busy admits no more than idle


def test_degrades_to_fuse_without_reranker():
    ret = TwoStageRetriever(_Base(), stage1_methods=("bm25", "vector"), reranker=None)
    res = ret.retrieve("q", k=3, load_band="lo")
    assert res.reranked is False and len(res.hits) >= 1
    assert res.stage1_n >= len(res.hits)            # stage-1 recall ⊇ what was served
    # retriever-shaped entrypoint returns plain hits
    assert all(isinstance(h, Hit) for h in ret.search("q", 3, "hybrid"))
