"""AthenaBackend — an AWS Athena ``WarehouseBackend`` for the AnalyticalRetriever.

The production analytical arm: the same provider-neutral text-to-SQL engine
(``adapters.store_analytical``) points at Athena instead of local DuckDB. Text-to-SQL over an S3 data
lake / Glue catalog is exactly the "analytical representation" the planner already routes to.

``schema()`` reads the Glue catalog via ``information_schema``; ``run_sql`` runs the already-guarded
query synchronously (start → poll → results). Client is injectable for tests. Defence in depth: the
Athena workgroup should use a read-only IAM role — the SQL guard is the first line, not the only one.
"""
from __future__ import annotations

import time


class AthenaBackend:
    def __init__(self, session=None, *, database: str, workgroup: str = "primary",
                 output_location: str, client=None, poll_interval: float = 0.5,
                 max_wait_s: float = 30.0, sleep=time.sleep):
        self._session = session
        self.database = database
        self.workgroup = workgroup
        self.output_location = output_location
        self._client = client
        self.poll_interval = poll_interval
        self.max_wait_s = max_wait_s
        self._sleep = sleep

    def _athena(self):
        if self._client is None:
            self._client = self._session.client("athena")
        return self._client

    def dialect(self) -> str:
        return "athena"  # Trino/Presto SQL

    def schema(self) -> str:
        sql = ("SELECT table_name, column_name, data_type FROM information_schema.columns "
               f"WHERE table_schema = '{self.database}' ORDER BY table_name, ordinal_position")
        tables: dict[str, list[str]] = {}
        for row in self.run_sql(sql, max_rows=500):
            tables.setdefault(row["table_name"], []).append(f"{row['column_name']} {row['data_type']}")
        return "\n".join(f"{t}({', '.join(cols)})" for t, cols in tables.items()) or "(no tables)"

    def run_sql(self, sql: str, max_rows: int = 100) -> list[dict]:
        client = self._athena()
        qid = client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": self.database},
            WorkGroup=self.workgroup,
            ResultConfiguration={"OutputLocation": self.output_location},
        )["QueryExecutionId"]

        waited = 0.0
        while True:
            state = client.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]["State"]
            if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
                break
            if waited >= self.max_wait_s:
                raise TimeoutError(f"athena query {qid} still {state} after {self.max_wait_s}s")
            self._sleep(self.poll_interval)
            waited += self.poll_interval
        if state != "SUCCEEDED":
            raise RuntimeError(f"athena query {qid} {state}")

        res = client.get_query_results(QueryExecutionId=qid, MaxResults=min(max_rows + 1, 1000))
        rows = res.get("ResultSet", {}).get("Rows", []) or []
        if not rows:
            return []
        header = [c.get("VarCharValue", "") for c in rows[0].get("Data", [])]
        out: list[dict] = []
        for r in rows[1: max_rows + 1]:
            vals = [c.get("VarCharValue") for c in r.get("Data", [])]
            out.append(dict(zip(header, vals)))
        return out
