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

import re

from ..types import IntentBucket, KnowledgeRepresentation, Retrieval

# method → the knowledge representation it operates over
REPRESENTATION_OF: dict[Retrieval, KnowledgeRepresentation] = {
    "vector": "document", "bm25": "document", "hybrid": "document", "file": "document",
    "graph": "graph",
    "community": "community",
    "temporal": "temporal",
    "code": "code",
    # the analytical representation = a STRUCTURED data store. `sql` for relational; `mongo` / `elastic`
    # where SQL is not applicable; `api`/`logs` for other structured feeds. A deployment binds a
    # RetrieverPlugin under whichever method matches its DB; the planner routes here, the bound plugin
    # executes the query. Unbound methods are pruned by the cost model (the `hybrid` fallback catches
    # a store that isn't wired).
    "sql": "analytical", "mongo": "analytical", "elastic": "analytical",
    "logs": "analytical", "api": "analytical",
    "image": "multimodal", "colpali": "multimodal", "video": "multimodal",
}


def representation_for(method: Retrieval | str) -> KnowledgeRepresentation:
    """The knowledge representation a retrieval method operates over (default: document)."""
    return REPRESENTATION_OF.get(method, "document")  # type: ignore[arg-type]


def methods_for(representation: KnowledgeRepresentation | str) -> tuple[Retrieval, ...]:
    """The retrieval methods that specialize a given representation."""
    return tuple(m for m, r in REPRESENTATION_OF.items() if r == representation)


# ─── the classify head: intent → representation (v4's first-class decision) ───
# A bucket carries a *default* representation (the buckets are already representation-leaning:
# multi_hop→graph, temporal→temporal, code→code). Content HINTS override it — most importantly
# they reach representations no bucket produces on its own (analytical OLAP, multimodal).
BUCKET_REPRESENTATION: dict[IntentBucket, KnowledgeRepresentation] = {
    "multi_hop": "graph",
    "temporal": "temporal",
    "code_reasoning": "code",
    # everything else defaults to document unless a hint says otherwise
}

# Ordered: first hit wins. These are the signals a bucket alone misses.
HINT_RULES: list[tuple[re.Pattern, KnowledgeRepresentation]] = [
    (re.compile(r"\b(how\s+many|number\s+of|count\s+of|\btotal\b|\bsum\b|average|avg|median|"
                r"per\s+(day|week|month|quarter|user|account)|over\s+the\s+(last|past)\s+\w+|"
                r"top\s+\d+|rank(ed|ing)?|group\s+by|distribution\s+of|trend|breakdown|"
                r"month[- ]over[- ]month|year[- ]over[- ]year|\bmrr\b|\barr\b|conversion\s+rate)\b",
                re.I), "analytical"),
    # structured-store lookups/filters against a KNOWN schema (relational or NoSQL/search) — not just
    # aggregates. A deployment with a plugged-in DB routes these to its SQL/Mongo/Elastic retriever.
    (re.compile(r"\b(select\s+.+\s+from\b|\bjoin\b|\bsql\b|\bnosql\b|"
                r"(records?|rows?|documents?|entries?|orders?|customers?|users?|invoices?|"
                r"transactions?|accounts?)\s+(where|with|whose|that\s+have)\b|"
                r"(from|in|query|querying)\s+(the\s+)?[\w-]+\s+(table|collection|index|database|db|schema)\b|"
                r"(table|collection|index|schema)\s+(named|called|for|containing)\b|"
                r"(columns?|fields?|attributes?)\s+(of|in|from|for)\b|"
                r"filter(ed)?\s+by\b|look\s*up\s+\w+\s+by\b)", re.I), "analytical"),
    (re.compile(r"\b(screenshot|screen\s?shot|image|photo|picture|diagram|figure|chart\s+image|"
                r"scanned|scan\s+of|receipt|invoice\s+image|whiteboard|slide)\b", re.I), "multimodal"),
    (re.compile(r"\b(as\s+of|point[- ]in[- ]time|what\s+changed|history\s+of|superseded|"
                r"no\s+longer|used\s+to\s+be|previously|valid\s+(from|until))\b", re.I), "temporal"),
    (re.compile(r"\b(related\s+to|connected\s+to|depend(s|ency|encies)?\s+on|graph\s+of|"
                r"network\s+of|linked\s+to|relationship\s+between|traverse|multi[- ]hop)\b",
                re.I), "graph"),
]


def classify(bucket: IntentBucket | str, text: str, entities: tuple[str, ...] = ()) -> KnowledgeRepresentation:
    """Map an analysed intent to the knowledge representation that can answer it.

    Content hints win over the bucket default, so 'how many incidents last week' routes to the
    analytical representation even though its bucket is 'incident'. Default: document."""
    for pat, rep in HINT_RULES:
        if pat.search(text or ""):
            return rep
    return BUCKET_REPRESENTATION.get(bucket, "document")  # type: ignore[arg-type]
