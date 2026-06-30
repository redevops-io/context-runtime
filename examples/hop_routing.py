"""ContextOS routes single-hop (redevops-rag) vs multi-hop (HippoRAG) per query.

The control plane's job: recognize when an answer lives in the CONNECTIONS between
documents (multi-hop → graph) versus inside one chunk (single-hop → hybrid), and only
pay the graph cost when it's warranted. Run offline with the in-memory single-hop
store + the SimGraph multi-hop retriever (HippoRAG's dependency-free stand-in).

    python examples/hop_routing.py
"""
from __future__ import annotations

from contextos import ContextRuntime, Goal
from contextos.adapters.model_stub import StubModel
from contextos.adapters.store_inmemory import InMemoryStore
from contextos.adapters.store_hipporag import SimGraphRetriever
from contextos.adapters.store_router import HopRouterRetriever
from contextos.runtime.config import Config

_TIERS = ("local", "cheap", "premium")

# The classic multi-hop corpus (the Use-cases.odt Parkinson's example): the link from
# the query's first term to its last lives in a BRIDGE doc that mentions neither.
CORPUS = [
    {"chunk_id": "d1", "filename": "d1", "text": "Mitochondrial dysfunction impairs ATP production in dopaminergic neurons.", "created_at": None},
    {"chunk_id": "d2", "filename": "d2", "text": "Reduced ATP production triggers alpha-synuclein aggregation in cells.", "created_at": None},
    {"chunk_id": "d3", "filename": "d3", "text": "Alpha-synuclein aggregation is the pathological hallmark of Parkinson disease.", "created_at": None},
    {"chunk_id": "d4", "filename": "d4", "text": "Coffee consumption has been weakly associated with lower Parkinson risk.", "created_at": None},
]


def _runtime() -> ContextRuntime:
    single = InMemoryStore(list(CORPUS))            # bm25/vector/hybrid (redevops-rag in prod)
    graph = SimGraphRetriever(list(CORPUS))         # graph/multi-hop (HippoRAG in prod)
    router = HopRouterRetriever(single_hop=single, graph=graph)
    return ContextRuntime(models={t: StubModel(tier=t) for t in _TIERS},
                          retriever=router, config=Config(default_tier="cheap"))


def show(rt, q):
    plan = rt.plan(q)
    method = next((s.params.get("method") for s in plan.chosen.steps if s.type == "retrieve"), "?")
    ctx = rt.build_context(plan, Goal(text=q))
    print(f"\nQ: {q}")
    print(f"   intent={plan.intent.bucket}  → method={method}  (est cost ${plan.score.cost_usd}, acc {plan.score.expected_accuracy})")
    for h in ctx.hits:
        print(f"     {h.meta.get('hop','?'):<18} {h.filename}: {h.text[:60]}")


def run() -> None:
    rt = _runtime()
    print("=" * 78)
    print("Single-hop question — the answer is in one chunk; planner picks hybrid:")
    show(rt, "What is the pathological hallmark of Parkinson disease?")

    print("\n" + "=" * 78)
    print("Multi-hop question — the link runs d1→d2→d3; planner picks graph (multi-hop).")
    print("Single-hop would miss d2 (the bridge: it shares no query term).")
    show(rt, "How is mitochondrial dysfunction linked to Parkinson disease?")

    # prove the contrast: what single-hop alone returns for the multi-hop question
    print("\n" + "-" * 78)
    q = "How is mitochondrial dysfunction linked to Parkinson disease?"
    single_only = InMemoryStore(list(CORPUS)).search(q, k=4, method="hybrid")
    graph_only = SimGraphRetriever(list(CORPUS)).search(q, k=4, method="graph")
    print(f"single-hop returns: {[h.filename for h in single_only]}  (no d2 — the bridge is invisible)")
    print(f"multi-hop returns:  {[(h.filename, h.meta['hop']) for h in graph_only]}  (d2 surfaced via the ATP→α-synuclein hop)")


if __name__ == "__main__":
    run()
