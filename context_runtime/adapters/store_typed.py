"""A provider of typed, exact operations — the application's computations behind the runtime's retriever seam.

The application registers ``TypedOperation``s (name, how to recognise a request for it, how to compute it, how to render and
what facts the rendering is accountable for); this module holds no domain. The provider declares ``exact`` and ``typed``,
returns provenance-bearing ``Hit``s (``meta``: op, args, result, facts), and falls back to a wrapped natural-language retriever (for
example the guarded text-to-SQL ``AnalyticalRetriever``) when no operation applies.

``resolve`` is what the runtime's deterministic fast path asks: does exactly one unambiguous set of typed calls fully answer
this question? The measures a question asks for come from the application's ``vocabulary`` (measure -> pattern); the operations declare
what they ``provide``. A question that asks for a measure no matched operation provides is NOT covered, so the fast path declines it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..types import Hit, PluginInfo
from ..verification.narration import Fact


@dataclass(frozen=True)
class TypedOperation:
    name: str
    match: Callable[[str], "Sequence[Any] | None"]       # question -> args (a tuple/list) when this operation applies, else None
    run: Callable[..., Any]                              # *args -> structured result
    render: Callable[[Any], str]                         # structured result -> text (deterministic)
    facts: Callable[[Any], Sequence[Fact]] = lambda result: ()
    provides: frozenset = frozenset()                    # measures this operation can answer


@dataclass(frozen=True)
class TypedCall:
    op: TypedOperation
    args: tuple


@dataclass(frozen=True)
class TypedResolution:
    calls: tuple
    covered: bool
    uncovered: tuple = ()


class TypedOperationRetriever:
    natural_query_methods = frozenset({"sql", "mongo", "elastic", "logs", "api"})

    def __init__(self, operations: Sequence[TypedOperation], *, fallback=None, vocabulary: Mapping[str, str] | None = None):
        self.operations = tuple(operations)
        self.fallback = fallback
        self.vocabulary = {m: re.compile(p, re.I) for m, p in (vocabulary or {}).items()}

    # ---- resolution (no execution, no model) ----
    def resolve(self, question: str) -> TypedResolution | None:
        calls = []
        for op in self.operations:
            args = op.match(question)
            if args is not None:
                calls.append(TypedCall(op, tuple(args)))
        if not calls:
            return None
        provided = set().union(*(c.op.provides for c in calls))
        asked = {m for m, pat in self.vocabulary.items() if pat.search(question)}
        uncovered = tuple(sorted(asked - provided))
        return TypedResolution(tuple(calls), covered=not uncovered, uncovered=uncovered)

    def execute(self, resolution: TypedResolution) -> list[Hit]:
        hits = []
        for i, c in enumerate(resolution.calls):
            result = c.op.run(*c.args)
            hits.append(Hit(chunk_id=f"typed:{c.op.name}:{i}", filename="typed-operation", text=c.op.render(result), score=1.0,
                            source="typed", meta={"op": c.op.name, "args": list(c.args), "result": result, "at_row_cap": False,
                                                  "facts": [f.__dict__ if isinstance(f, Fact) else f for f in c.op.facts(result)]}))
        return hits

    # ---- RetrieverPlugin ----
    def search(self, query: str, k: int, method: str = "sql") -> list[Hit]:
        res = self.resolve(query)
        if res is not None:
            return self.execute(res)
        return self.fallback.search(query, k, method) if self.fallback is not None else []

    def info(self) -> PluginInfo:
        return PluginInfo(name="typed-operations", kind="retriever", capabilities=frozenset({"sql", "api", "exact", "typed"}))
