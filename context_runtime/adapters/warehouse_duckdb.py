"""DuckDBWarehouse — a local ``WarehouseBackend`` for the AnalyticalRetriever.

The offline/dev backend: a DuckDB file (or in-memory) the text-to-SQL engine queries, so the whole
analytical path — generate → guard → execute → rows-as-Hits — runs with no cloud, mirroring how the
soc_triage/rag examples simulate their core. In production the same engine points at Athena
(``providers.aws.athena_backend``) instead; the retriever doesn't change.

Read-only by contract: ``run_sql`` executes the already-guarded query and caps rows. DuckDB is an
optional dep (``context-runtime[duckdb]``); the connection is injectable for tests.
"""
from __future__ import annotations


class DuckDBWarehouse:
    def __init__(self, conn=None, *, database: str = ":memory:"):
        self._conn = conn
        self.database = database

    def _connection(self):
        if self._conn is None:
            import duckdb  # optional: context-runtime[duckdb]
            self._conn = duckdb.connect(self.database)
        return self._conn

    def dialect(self) -> str:
        return "duckdb"

    def schema(self) -> str:
        """A compact column summary the SQL generator can read: ``table(col type, …)`` per table."""
        rows = self._connection().execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema NOT IN ('information_schema','pg_catalog') "
            "ORDER BY table_name, ordinal_position"
        ).fetchall()
        tables: dict[str, list[str]] = {}
        for tname, col, dtype in rows:
            tables.setdefault(tname, []).append(f"{col} {dtype}")
        return "\n".join(f"{t}({', '.join(cols)})" for t, cols in tables.items()) or "(no tables)"

    def run_sql(self, sql: str, max_rows: int = 100) -> list[dict]:
        cur = self._connection().execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, r)) for r in cur.fetchmany(max_rows)]
