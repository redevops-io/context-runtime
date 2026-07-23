"""AnalyticalRetriever — the analytical representation made real (text-to-SQL over a warehouse).

This is the engine the planner's ``analytical`` route has always wanted and never had: a
``RetrieverPlugin`` that turns a natural-language question into a **guarded, read-only** SQL query,
runs it against a pluggable ``WarehouseBackend``, and marshals rows into ``Hit``s. Provider-neutral by
construction — the text-to-SQL generator is any ``ModelPlugin`` and the execution backend is any
``WarehouseBackend`` (local DuckDB, AWS Athena, later BigQuery/Synapse/Postgres). Swapping clouds
swaps the backend, not the engine.

Security is the whole game here: a generated statement is validated by ``guard_sql`` before it ever
reaches the backend — SELECT/WITH only, single statement, no DDL/DML, row-capped. A backend that
also enforces read-only credentials is defence in depth, not a substitute.
"""
from __future__ import annotations

import re

from ..types import Hit, ModelRequest, PluginInfo, Retrieval

_ALLOWED_START = re.compile(r"^\s*(with|select)\b", re.I)
# whole-word mutation/DDL/admin verbs that must never appear in a read path
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|merge|replace|"
    r"attach|copy|pragma|call|exec|execute|vacuum|analyze|set|reset|load|install)\b", re.I)


class SqlGuardError(ValueError):
    """A generated statement failed the read-only safety check; it is never executed."""


def strip_sql(text: str) -> str:
    """Pull a bare SQL statement out of a model reply (strip ``` fences / prose / trailing ';')."""
    t = text.strip()
    m = re.search(r"```(?:sql)?\s*(.+?)```", t, re.S | re.I)
    if m:
        t = m.group(1).strip()
    # if the model prefixed prose, keep from the first WITH/SELECT
    m2 = re.search(r"(?is)\b(with|select)\b.*", t)
    if m2:
        t = m2.group(0).strip()
    return t.rstrip(";").strip()


def guard_sql(sql: str, max_rows: int) -> str:
    """Validate + normalize a read-only query. Raises SqlGuardError on anything unsafe."""
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise SqlGuardError("empty query")
    if ";" in s:
        raise SqlGuardError("multiple statements are not allowed")
    if not _ALLOWED_START.match(s):
        raise SqlGuardError("only SELECT / WITH queries are allowed")
    if _FORBIDDEN.search(s):
        raise SqlGuardError(f"forbidden keyword in query: {_FORBIDDEN.search(s).group(0)!r}")
    # enforce a row cap: append LIMIT when the outer query has none
    if not re.search(r"\blimit\s+\d+\s*$", s, re.I):
        s = f"{s}\nLIMIT {max_rows}"
    return s


def _row_text(row: dict) -> str:
    return " | ".join(f"{k}={v}" for k, v in row.items())


_SQL_SYSTEM = (
    "You translate a question into ONE read-only SQL SELECT query for the given schema. "
    "Output only SQL, no prose, no explanation. Never write INSERT/UPDATE/DELETE/DDL. "
    "Prefer explicit column lists and add aggregations when the question asks 'how many', "
    "'total', 'average', 'per', 'top N', or 'group by'."
)


class AnalyticalRetriever:
    """RetrieverPlugin: NL question → guarded SQL → rows-as-Hits over a WarehouseBackend."""

    def __init__(self, backend, model, *, max_rows: int = 100, sql_capability: str = "draft"):
        self.backend = backend      # WarehouseBackend: schema() + run_sql() + dialect()
        self.model = model          # ModelPlugin: generates SQL (any provider — Bedrock or local)
        self.max_rows = max_rows
        self.sql_capability = sql_capability

    def _generate_sql(self, query: str) -> str:
        schema = self.backend.schema()
        prompt = (f"Schema ({self.backend.dialect()}):\n{schema}\n\n"
                  f"Question: {query}\n\nSQL:")
        res = self.model.complete(ModelRequest(
            messages=({"role": "user", "content": prompt},),
            system=_SQL_SYSTEM, capability=self.sql_capability, max_tokens=400))
        return strip_sql(res.text)

    def search(self, query: str, k: int, method: Retrieval = "sql") -> list[Hit]:
        raw = self._generate_sql(query)
        sql = guard_sql(raw, min(k, self.max_rows) if k else self.max_rows)
        rows = self.backend.run_sql(sql, max_rows=self.max_rows)
        dialect = self.backend.dialect()
        hits: list[Hit] = []
        for i, row in enumerate(rows[: (k or self.max_rows)]):
            hits.append(Hit(
                chunk_id=f"sql:{i}",
                filename=f"{dialect}:analytical",
                text=_row_text(row),
                score=1.0,               # exact query results — rank is query order, not relevance
                source=f"analytical:{dialect}",
                meta={"row": row, "sql": sql, "method": method},
            ))
        return hits

    def index(self, path: str) -> dict:
        return {"analytical": "queries a live warehouse; nothing to index", "dialect": self.backend.dialect()}

    def info(self) -> PluginInfo:
        return PluginInfo(name="analytical", kind="retriever",
                          capabilities=frozenset({"sql", "mongo", "elastic", "logs", "api"}))
