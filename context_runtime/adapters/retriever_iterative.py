"""IterativeRetriever — retrieve → reason → refine → retrieve (SPEC §4.5).

The fourth RAG paradigm we didn't have: instead of one shot, gather evidence over
several rounds, letting what came back reshape the next query. This is what lifts hard
multi-hop questions where the bridge fact only becomes searchable AFTER the first hop is
retrieved ("who succeeded the person who founded X?" → round 1 finds the founder, round
2 searches the founder's successor).

It wraps ANY base retriever/method (bm25/vector/hybrid/graph/community), so it composes
with everything else. Query refinement is model-driven when a ModelPlugin is supplied
(the same seam sidekick/an LLM plugs into) and falls back to deterministic query
EXPANSION otherwise — salient terms from the first-round hits are appended, giving a
real, testable second hop with no LLM. Rounds are bounded so the cost model can price it.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from ..types import Hit, Retrieval
from .store_inmemory import _token_list, _tokens


class IterativeRetriever:
    def __init__(self, base: Any, *, model: Any = None, model_name: str = "",
                 max_rounds: int = 2, expand_terms: int = 4):
        self.base = base                # any RetrieverPlugin
        self.model = model              # optional ModelPlugin → query refinement
        self.model_name = model_name
        self.max_rounds = max(1, max_rounds)
        self.expand_terms = expand_terms

    def search(self, query: str, k: int, method: Retrieval = "hybrid") -> list[Hit]:
        seen: set[str] = set()
        acc: list[Hit] = []
        q = query
        for r in range(self.max_rounds):
            for h in self.base.search(q, k, method):
                if h.chunk_id not in seen:
                    seen.add(h.chunk_id)
                    acc.append(h)
            if r == self.max_rounds - 1:
                break
            nxt = self._refine(query, acc)
            if not nxt or nxt == q:
                break
            q = nxt
        acc.sort(key=lambda h: h.score, reverse=True)
        return acc[:k] if k else acc

    def _refine(self, original: str, hits: list[Hit]) -> str | None:
        if self.model is not None:
            refined = self._llm_refine(original, hits)
            if refined is not None:
                return refined
        # deterministic fallback: expand the query with the most salient NEW terms seen
        base_terms = _tokens(original)
        counts: Counter = Counter()
        for h in hits[:3]:
            for t in _token_list(h.text):
                if t not in base_terms:
                    counts[t] += 1
        extra = [t for t, _ in counts.most_common(self.expand_terms)]
        return (original + " " + " ".join(extra)) if extra else None

    def _llm_refine(self, original: str, hits: list[Hit]) -> str | None:
        from ..types import ModelRequest
        context = "\n\n".join(f"[{i+1}] {h.text[:500]}" for i, h in enumerate(hits[:4]))
        try:
            res = self.model.complete(ModelRequest(
                model=self.model_name,
                prompt=("You are refining a search. Given the QUESTION and the SNIPPETS "
                        "retrieved so far, output ONE follow-up search query that would "
                        "retrieve still-missing information needed to answer it fully. "
                        "If the snippets already suffice, output exactly DONE.\n\n"
                        f"QUESTION: {original}\n\nSNIPPETS:\n{context}\n\nFOLLOW-UP QUERY:"),
                max_tokens=64))
            out = (getattr(res, "text", "") or "").strip()
            if not out or out.upper().startswith("DONE"):
                return None
            return out.splitlines()[0][:200]
        except Exception:
            return None  # fall through to deterministic expansion

    def index(self, path: str) -> dict:
        return self.base.index(path) if hasattr(self.base, "index") else {}

    def info(self):
        from ..types import PluginInfo
        base_caps = set(self.base.info().capabilities) if hasattr(self.base, "info") else set()
        return PluginInfo(name="iterative_retriever", kind="retriever", version="0.1",
                          capabilities=frozenset(base_caps | {"iterative", "multi_round"}))
