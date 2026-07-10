"""Knowledge representations — the planner's first-class decision axis (Whitepaper v4).

Context Runtime does not assume every problem is document retrieval. A request is first
mapped to the *knowledge representation* that can answer it — a document, a graph of
relationships, a bi-temporal fact history, an analytical/OLAP cube, code, or media — and
only then to a concrete ``Retrieval`` method within that representation. Retrieval is one
specialization; the planner routes across representations without the application changing.

This module holds the (deliberately small, inspectable) mapping from retrieval method →
representation, so plans, traces and the cost model can reason about *which representation*
was chosen — not just which method. The heavy engines behind each representation (a hosted
Graphiti temporal store, an OLAP connector, a real HippoRAG graph) bind at construction time
via the ``RetrieverPlugin`` seam; this table is only the taxonomy.
"""
from __future__ import annotations

from ..types import KnowledgeRepresentation, Retrieval

# method → the knowledge representation it operates over
REPRESENTATION_OF: dict[Retrieval, KnowledgeRepresentation] = {
    "vector": "document", "bm25": "document", "hybrid": "document", "file": "document",
    "graph": "graph",
    "community": "community",
    "temporal": "temporal",
    "code": "code",
    "sql": "analytical", "logs": "analytical", "api": "analytical",
    "image": "multimodal", "colpali": "multimodal", "video": "multimodal",
}


def representation_for(method: Retrieval | str) -> KnowledgeRepresentation:
    """The knowledge representation a retrieval method operates over (default: document)."""
    return REPRESENTATION_OF.get(method, "document")  # type: ignore[arg-type]


def methods_for(representation: KnowledgeRepresentation | str) -> tuple[Retrieval, ...]:
    """The retrieval methods that specialize a given representation."""
    return tuple(m for m, r in REPRESENTATION_OF.items() if r == representation)
