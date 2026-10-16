"""DuckDBWarehouse — a local ``WarehouseBackend`` for the AnalyticalRetriever.

The offline/dev backend: a DuckDB file (or in-memory) the text-to-SQL engine queries, so the whole
analytical path — generate → guard → execute → rows-as-Hits — runs with no cloud, mirroring how the
soc_triage/rag examples simulate their core. In production the same engine points at Athena
(``providers.aws.athena_backend``) instead; the retriever doesn't change.

Read-only by contract: ``run_sql`` executes the already-guarded query and caps rows. DuckDB is an
optional dep (``context-runtime[duckdb]``); the connection is injectable for tests. External file and
network access is disabled on the connection before any query runs.
"""
from __future__ import annotations

import threading


class DuckDBWarehouse:
    def __init__(self, conn=None, *, database: str = ":memory:", timeout_s: float = 30.0,
                 memory_limit: str = "1GB", threads: int = 2, max_expression_depth: int = 200):
        self._conn = conn
        self.timeout_s = timeout_s
        self.memory_limit = memory_limit
        self.threads = threads
        self.max_expression_depth = max_expression_depth
        self.database = database
        self._locked = False

    def _connection(self):
        if self._conn is None:
            import duckdb  # optional: context-runtime[duckdb]
            self._conn = duckdb.connect(self.database)
        if not self._locked:
            # No file / network access from queries, and the setting cannot be changed back.
            self._conn.execute("SET enable_external_access=false")
            self._conn.execute(f"SET memory_limit='{self.memory_limit}'")
            self._conn.execute(f"SET threads={int(self.threads)}")
            self._conn.execute(f"SET max_expression_depth={int(self.max_expression_depth)}")
            self._conn.execute("SET lock_configuration=true")
            self._locked = True
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
        conn = self._connection()
        # Statement timeout: interrupt the running query if it exceeds timeout_s.
        timer = threading.Timer(self.timeout_s, conn.interrupt)
        timer.daemon = True
        timer.start()
        try:
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(cols, r)) for r in cur.fetchmany(max_rows)]
        finally:
            timer.cancel()
