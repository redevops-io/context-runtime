"""CommunityRetriever — community-detection + summaries for GLOBAL/broad queries (SPEC §4.5).

The honest gap our own eval found: broad, aggregation-style questions ("what's abnormal
across all the panels?", "результаты анализа крови") return diffuse context with no
single focused passage — because the answer spans MANY passages, and single-hit
retrieval (BM25/vector) can only return one. This is exactly what Microsoft GraphRAG's
"global search" and FastMemory's topology clustering solve: cluster the corpus into
communities of related passages, summarize each, and answer broad queries from the
best-matching community summary rather than a lone chunk.

Pipeline (deterministic core, optional LLM summaries):
  1. passage graph  — passages linked by count of shared significant terms
  2. communities    — deterministic label propagation (no deps, no randomness)
  3. summaries      — extractive by default (top central passages); LLM if a model is given
  4. search         — score the query against community term profiles, return summaries

It implements the RetrieverPlugin contract, so the runtime routes to it via
``method="community"`` exactly like bm25/vector/hybrid/graph.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..types import Hit, PluginInfo, Retrieval
from .store_inmemory import _token_list, _tokens


class CommunityRetriever:
    def __init__(self, docs: list[dict] | None = None, *, model: Any = None,
                 model_name: str = "", min_shared: int = 2, top_members: int = 3,
                 summary_chars: int = 700, source: str = "community"):
        self.docs = docs or []
        self.model = model            # optional ModelPlugin → LLM community summaries
        self.model_name = model_name
        self.min_shared = min_shared  # edge threshold: shared significant terms
        self.top_members = top_members
        self.summary_chars = summary_chars
        self.source = source
        self._cache_key = None
        self._communities: list[dict] = []

    # ──────────────────────────── graph + communities ────────────────────────────

    def _build(self) -> list[dict]:
        key = (id(self.docs), len(self.docs))
        if self._cache_key == key:
            return self._communities
        n = len(self.docs)
        tok_sets = [_tokens(d["text"]) for d in self.docs]
        # global document frequency → down-weight ubiquitous terms (they link everything)
        df: dict[str, int] = defaultdict(int)
        for s in tok_sets:
            for t in s:
                df[t] += 1
        maxdf = max(3, int(0.30 * n)) if n else 1

        # shared-term counts via an inverted index over discriminative terms only
        inv: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(tok_sets):
            for t in s:
                if df[t] <= maxdf:
                    inv[t].append(i)
        shared: dict[tuple[int, int], int] = defaultdict(int)
        for members in inv.values():
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    shared[(members[a], members[b])] += 1
        adj: dict[int, dict[int, float]] = defaultdict(dict)
        for (i, j), c in shared.items():
            if c >= self.min_shared:
                adj[i][j] = float(c)
                adj[j][i] = float(c)

        labels = self._greedy_modularity(adj, n)
        groups: dict[int, list[int]] = defaultdict(list)
        for i, lab in enumerate(labels):
            groups[lab].append(i)

        communities = []
        for cid, (_, members) in enumerate(sorted(groups.items())):
            # order members by intra-community degree (centrality), then id — deterministic
            members.sort(key=lambda i: (-sum(adj[i].get(j, 0.0) for j in members),
                                        self.docs[i]["chunk_id"]))
            profile: Counter = Counter()
            for i in members:
                profile.update(_token_list(self.docs[i]["text"]))
            communities.append({
                "id": cid, "members": members, "profile": profile,
                "summary": self._summarize(members), "df": df, "n": n,
            })
        self._communities = communities
        self._cache_key = key
        return communities

    @staticmethod
    def _greedy_modularity(adj: dict[int, dict[int, float]], n: int) -> list[int]:
        """Deterministic weighted greedy modularity (Clauset-Newman-Moore): agglomerate the
        community pair with the largest modularity gain until no positive gain remains. Ties
        break on smallest (c,d), so it is fully reproducible. Unlike label propagation this
        resists the "monster community" collapse on densely-connected corpora — merging stops
        at the modularity peak rather than absorbing everything."""
        deg = {i: sum(adj[i].values()) for i in range(n)}
        m2 = sum(deg.values())  # 2m
        if m2 == 0:
            return list(range(n))
        members = {i: {i} for i in range(n)}
        a = {i: deg[i] for i in range(n)}          # community weighted degree
        active = {i for i in range(n) if deg[i] > 0}  # isolated nodes stay singletons
        e: dict[tuple[int, int], float] = defaultdict(float)  # inter-community weight
        for i in range(n):
            for j, w in adj[i].items():
                if i < j:
                    e[(i, j)] += w
        while True:
            best, best_dq = None, 1e-9
            for (c, d), w in e.items():
                if c not in active or d not in active:
                    continue
                dq = 2 * (w / m2 - (a[c] * a[d]) / (m2 * m2))
                if dq > best_dq or (dq == best_dq and (best is None or (c, d) < best)):
                    best, best_dq = (c, d), dq
            if best is None:
                break
            c, d = best  # merge d into c
            members[c] |= members[d]
            a[c] += a[d]
            active.discard(d)
            merged: dict[tuple[int, int], float] = defaultdict(float)
            for (x, y), w in e.items():
                x = c if x == d else x
                y = c if y == d else y
                if x != y:
                    merged[(min(x, y), max(x, y))] += w
            e = merged
        labels = list(range(n))
        for c in active:
            for node in members[c]:
                labels[node] = c
        return labels

    # ──────────────────────────── summaries ────────────────────────────

    def _summarize(self, members: list[int]) -> str:
        texts = [self.docs[i]["text"] for i in members[: self.top_members]]
        if self.model is not None and texts:
            llm = self._llm_summary(texts)
            if llm:
                return llm
        # extractive default: stitch the most central passages, clipped to a budget
        budget = self.summary_chars
        out, used = [], 0
        for t in texts:
            t = t.strip()
            take = t[: max(0, budget - used)]
            if take:
                out.append(take)
                used += len(take)
            if used >= budget:
                break
        return "\n\n".join(out)

    def _llm_summary(self, texts: list[str]) -> str:
        from ..types import ModelRequest
        joined = "\n\n---\n\n".join(t[:1500] for t in texts)
        try:
            res = self.model.complete(ModelRequest(
                model=self.model_name,
                prompt=("Summarize the shared theme and the key facts/values across these "
                        "related passages in 3-4 sentences for a search index. Preserve "
                        "numbers and named entities.\n\n" + joined),
                max_tokens=400))
            return (getattr(res, "text", "") or "").strip()
        except Exception:
            return ""

    # ──────────────────────────── retriever contract ────────────────────────────

    def search(self, query: str, k: int, method: Retrieval = "community") -> list[Hit]:
        communities = self._build()
        if not communities:
            return []
        q_terms = _tokens(query)
        if not q_terms:
            return []
        n = communities[0]["n"]
        df = communities[0]["df"]
        scored = []
        for c in communities:
            profile, size = c["profile"], len(c["members"])
            s = 0.0
            for t in q_terms:
                tf = profile.get(t, 0)
                if tf:
                    idf = math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
                    s += idf * tf / (tf + 1.0)  # saturating TF, IDF-weighted
            if s > 0:
                scored.append((s / math.sqrt(size), c))  # favour focused communities
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        hits = []
        for s, c in scored[:k]:
            members = [self.docs[i]["chunk_id"] for i in c["members"]]
            hits.append(Hit(
                chunk_id=f"community::{c['id']}",
                filename=f"community-{c['id']} (n={len(members)})",
                text=c["summary"], score=float(s), source=self.source,
                meta={"members": members, "size": len(members)}))
        return hits

    def index(self, path: str) -> dict:
        p = Path(path).expanduser()
        n = 0
        for fp in sorted(p.rglob("*")):
            if fp.suffix.lower() in (".md", ".txt", ".rst") and fp.is_file():
                self.docs.append({"chunk_id": f"{fp.name}::0", "filename": fp.name,
                                  "text": fp.read_text(errors="ignore"), "created_at": None})
                n += 1
        self._cache_key = None
        return {"files": n, "chunks": n, "communities": len(self._build())}

    def info(self) -> PluginInfo:
        return PluginInfo(name="community_retriever", kind="retriever", version="0.1",
                          capabilities=frozenset({"community", "global", "summaries", "multi_hop"}))
