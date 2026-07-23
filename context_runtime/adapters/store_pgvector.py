"""PgVectorRetriever — dense retrieval over Postgres + pgvector (SPEC §4.5).

Closes the audit gap: the article listed ``pgvector`` as a native engine but no adapter existed (the
Postgres store was ``tsvector`` full-text only). This is a real vector adapter — embed the query, then
``ORDER BY embedding <=> query`` (cosine) — and it doubles as the **Amazon Aurora/RDS pgvector** arm,
so a managed-Postgres deployment is a document RetrieverPlugin the router and bandit can select.

Both the DB connection and the embedder are injectable: the real path lazily uses ``psycopg``
(``[postgres]`` extra) + a fastembed embedder (``[embeddings]`` extra); tests pass fakes and touch
neither. The query is parameterized and the retriever only ever SELECTs.
"""
from __future__ import annotations

from ..types import Hit, PluginInfo, Retrieval


def _vector_literal(vec) -> str:
    return "[" + ",".join(f"{float(x):.6g}" for x in vec) + "]"


class PgVectorRetriever:
    def __init__(self, conn=None, *, dsn: str | None = None, table: str = "documents",
                 embedder=None, text_col: str = "text", id_col: str = "chunk_id",
                 filename_col: str = "filename", embedding_col: str = "embedding"):
        self._conn = conn
        self.dsn = dsn
        self.table = table
        self._embedder = embedder
        self.text_col = text_col
        self.id_col = id_col
        self.filename_col = filename_col
        self.embedding_col = embedding_col

    def _connection(self):
        if self._conn is None:
            import psycopg  # optional: context-runtime[postgres]
            self._conn = psycopg.connect(self.dsn)
        return self._conn

    def _embed(self, query: str):
        if self._embedder is not None:
            return self._embedder(query)     # injected: text -> sequence[float]
        # default: reuse the semantic adapter's fastembed embedder ([embeddings] extra)
        from .store_semantic import _embed  # lazy; raises without the extra
        return _embed([query])[0]

    def search(self, query: str, k: int, method: Retrieval = "vector") -> list[Hit]:
        vec = _vector_literal(self._embed(query))
        sql = (
            f"SELECT {self.id_col}, {self.filename_col}, {self.text_col}, "
            f"1 - ({self.embedding_col} <=> %s::vector) AS score "
            f"FROM {self.table} ORDER BY {self.embedding_col} <=> %s::vector LIMIT %s"
        )
        cur = self._connection().execute(sql, (vec, vec, k))
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else \
            [self.id_col, self.filename_col, self.text_col, "score"]
        out: list[Hit] = []
        for row in rows:
            r = dict(zip(cols, row))
            out.append(Hit(
                chunk_id=str(r.get(self.id_col, "")),
                filename=str(r.get(self.filename_col, self.table)),
                text=str(r.get(self.text_col, "")),
                score=float(r.get("score") or 0.0),
                source="pgvector",
                meta={"table": self.table},
            ))
        return out

    def index(self, path: str) -> dict:  # rows/embeddings are populated by the ingest pipeline
        return {"pgvector": "index/embeddings managed by the ingest pipeline", "table": self.table}

    def info(self) -> PluginInfo:
        return PluginInfo(name="pgvector", kind="retriever",
                          capabilities=frozenset({"vector", "hybrid"}))
