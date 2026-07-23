"""BigQueryBackend — a ``WarehouseBackend`` for the analytical (text-to-SQL) representation on GCP.

The same provider-neutral engine (``adapters.store_analytical.AnalyticalRetriever``) points at BigQuery
instead of Athena/DuckDB - text-to-SQL over a BigQuery dataset is exactly the analytical route the
planner already produces. ``schema()`` reads ``INFORMATION_SCHEMA.COLUMNS``; ``run_sql`` runs the
already-guarded query. Client injectable; defence in depth: run under a read-only BigQuery role.
"""
from __future__ import annotations


class BigQueryBackend:
    def __init__(self, session=None, *, dataset: str, project: str | None = None, client=None):
        self._session = session
        self.dataset = dataset
        self.project = project or (session.project if session else None)
        self._client = client

    def _bq(self):
        if self._client is None:
            self._client = self._session.bigquery_client()
        return self._client

    def dialect(self) -> str:
        return "bigquery"  # GoogleSQL

    def _rows(self, sql: str, max_rows: int) -> list[dict]:
        job = self._bq().query(sql)
        result = job.result(max_results=max_rows) if hasattr(job, "result") else job
        out = []
        for row in result:
            out.append(dict(row))
            if len(out) >= max_rows:
                break
        return out

    def schema(self) -> str:
        ds = f"`{self.project}`.{self.dataset}" if self.project else self.dataset
        sql = (f"SELECT table_name, column_name, data_type FROM {ds}.INFORMATION_SCHEMA.COLUMNS "
               f"ORDER BY table_name, ordinal_position")
        tables: dict[str, list[str]] = {}
        for r in self._rows(sql, 1000):
            tables.setdefault(r["table_name"], []).append(f"{r['column_name']} {r['data_type']}")
        return "\n".join(f"{t}({', '.join(cols)})" for t, cols in tables.items()) or "(no tables)"

    def run_sql(self, sql: str, max_rows: int = 100) -> list[dict]:
        return self._rows(sql, max_rows)
