"""PgVectorRetriever is a dense document RetrieverPlugin over Postgres+pgvector.

Exercised with an injected connection + embedder (no psycopg / no pgvector / no fastembed): proves the
SQL shape (cosine `<=>`, parameterized), the row→Hit mapping, and that it slots into the router as a
document arm — the same code path a managed Aurora/RDS pgvector deployment uses.
"""
from context_runtime.adapters.store_pgvector import PgVectorRetriever, _vector_literal
from context_runtime.plugins import base


class FakeCursor:
    def __init__(self, rows, cols):
        self._rows = rows
        self.description = [(c,) for c in cols]

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, rows, cols):
        self._rows = rows
        self._cols = cols
        self.executed = []

    def execute(self, sql, params):
        self.executed.append((sql, params))
        return FakeCursor(self._rows, self._cols)


def _r():
    rows = [("c1", "q3.md", "revenue grew", 0.92), ("c2", "q3.md", "costs flat", 0.41)]
    cols = ["chunk_id", "filename", "text", "score"]
    conn = FakeConn(rows, cols)
    return PgVectorRetriever(conn=conn, table="docs", embedder=lambda q: [0.1, 0.2, 0.3]), conn


def test_satisfies_retriever_protocol():
    r, _ = _r()
    assert isinstance(r, base.RetrieverPlugin)


def test_vector_literal_format():
    assert _vector_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"


def test_search_maps_rows_and_builds_cosine_sql():
    r, conn = _r()
    hits = r.search("revenue", k=5, method="vector")
    assert [h.chunk_id for h in hits] == ["c1", "c2"]
    assert hits[0].text == "revenue grew" and hits[0].score == 0.92 and hits[0].source == "pgvector"
    sql, params = conn.executed[0]
    assert "<=> %s::vector" in sql and "LIMIT %s" in sql
    assert params == ("[0.1,0.2,0.3]", "[0.1,0.2,0.3]", 5)   # embedded query bound twice + k


def test_routes_as_document_arm():
    from context_runtime.adapters.store_router import HopRouterRetriever
    from context_runtime.adapters.store_hipporag import SimGraphRetriever
    r, _ = _r()
    router = HopRouterRetriever(single_hop=r, graph=SimGraphRetriever([]))
    hits = router.search("revenue", k=3, method="vector")
    assert hits and hits[0].source == "pgvector"
