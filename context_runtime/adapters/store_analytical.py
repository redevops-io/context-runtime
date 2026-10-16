"""AnalyticalRetriever — the analytical representation made real (text-to-SQL over a warehouse).

This is the engine the planner's ``analytical`` route has always wanted and never had: a
``RetrieverPlugin`` that turns a natural-language question into a **guarded, read-only** SQL query,
runs it against a pluggable ``WarehouseBackend``, and marshals rows into ``Hit``s. Provider-neutral by
construction — the text-to-SQL generator is any ``ModelPlugin`` and the execution backend is any
``WarehouseBackend`` (local DuckDB, AWS Athena, later BigQuery/Synapse/Postgres). Swapping clouds
swaps the backend, not the engine.

Security is the whole game here: a generated statement is validated by ``guard_sql`` before it ever
reaches the backend — parsed, single statement, SELECT/WITH only, allowlisted functions, row-capped. A backend that
also enforces read-only credentials is defence in depth, not a substitute.
"""
from __future__ import annotations

import re

from ..types import Hit, ModelRequest, PluginInfo, Retrieval

class SqlGuardError(ValueError):
    """A generated statement failed the read-only safety check; it is never executed."""


# Functions a generated read-only query may call (upper-case SQL names). Anything else - file and
# network readers, system/admin functions, sleep, generators - is rejected.
ALLOWED_FUNCTIONS = frozenset("""
COUNT SUM AVG MIN MAX STDDEV STDDEV_POP STDDEV_SAMP VARIANCE VAR_POP VAR_SAMP MEDIAN MODE ANY_VALUE
APPROX_DISTINCT APPROX_COUNT_DISTINCT COUNT_IF BOOL_AND BOOL_OR STRING_AGG GROUP_CONCAT LISTAGG
ARRAY_AGG FIRST LAST CORR COVAR_POP COVAR_SAMP PERCENTILE_CONT PERCENTILE_DISC QUANTILE
ABS CEIL CEILING FLOOR ROUND TRUNC TRUNCATE SQRT POWER POW EXP LN LOG LOG2 LOG10 SIGN MOD
GREATEST LEAST COALESCE NULLIF IFNULL NVL IF CASE CAST TRY_CAST SAFE_CAST
LOWER UPPER LENGTH CHAR_LENGTH TRIM LTRIM RTRIM SUBSTRING SUBSTR REPLACE CONCAT CONCAT_WS LEFT RIGHT
POSITION STRPOS STARTS_WITH ENDS_WITH REVERSE INITCAP
DATE_TRUNC DATE_PART EXTRACT DATE_DIFF DATEDIFF DATE_ADD DATEADD DATE_SUB TIMESTAMP_DIFF TIMESTAMP_ADD
CURRENT_DATE CURRENT_TIMESTAMP CURRENT_TIME NOW LOCALTIMESTAMP YEAR MONTH DAY HOUR MINUTE SECOND
QUARTER WEEK DAYOFWEEK DAYOFMONTH DAYOFYEAR STRFTIME TO_CHAR TO_DATE TO_TIMESTAMP DATE TIMESTAMP
AND OR XOR
TIMESTAMP_TRUNC TIME_TO_STR TIME_TO_UNIX UNIX_TO_TIME STR_TO_TIME STR_POSITION LOGICAL_OR LOGICAL_AND
QUANTILE_CONT QUANTILE_DISC MD5_DIGEST LIST
LPAD RPAD PAD SPLIT_PART REGEXP_LIKE REGEXP_MATCHES REGEXP_REPLACE REGEXP_FULL_MATCH REGEXP_EXTRACT
CONTAINS EPOCH LAST_DAY LEN MD5 TRY_ELEMENT_AT DAY_OF_WEEK DAY_OF_MONTH DAY_OF_YEAR WEEK_OF_YEAR
ROW_NUMBER RANK DENSE_RANK NTILE LAG LEAD FIRST_VALUE LAST_VALUE NTH_VALUE PERCENT_RANK CUME_DIST
""".split())

# Dialects name the same function differently (DATEDIFF / DATE_DIFF, STRPOS / STR_POSITION), so
# names are compared with underscores removed.
_ALLOWED_NORMALIZED = frozenset(n.replace("_", "") for n in ALLOWED_FUNCTIONS)

# Statement-level nodes that may never appear anywhere in the tree (including inside CTEs).
_FORBIDDEN_NODE_NAMES = (
    "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "AlterTable", "TruncateTable",
    "Command", "Copy", "Set", "SetItem", "Pragma", "Attach", "Detach", "Install", "Load", "Use", "Grant",
    "Into", "Lock", "Fetch", "Transaction", "Commit", "Rollback", "Describe", "Show", "Analyze",
    "LoadData", "Export", "Kill", "Declare",
)
_ROOT_NODE_NAMES = ("Select", "Union", "Except", "Intersect")
# A plain table name part: no path separators, dots or quotes (blocks "file.csv"-style scans).
_TABLE_PART = re.compile(r"^[A-Za-z_][A-Za-z0-9_$ -]*$")
_FILE_EXTENSIONS = frozenset(
    "csv parquet json jsonl ndjson tsv txt xlsx arrow db duckdb gz zst".split())
_URL_SCHEMES = frozenset("http https s3 s3a gs gcs file ftp ftps az abfs abfss hf r2".split())
_SYSTEM_SCHEMAS = ("information_schema", "pg_catalog", "sys", "mysql", "performance_schema")
_SYSTEM_PREFIXES = ("pg_", "duckdb_", "sqlite_")


def _sqlglot():
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError as e:  # pragma: no cover - fail closed when the parser is missing
        raise SqlGuardError("sqlglot is required to validate SQL (pip install 'context-runtime[duckdb]')") from e
    return sqlglot, exp


def _node_types(exp, names):
    return tuple(t for t in (getattr(exp, n, None) for n in names) if isinstance(t, type))


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


def guard_sql(sql: str, max_rows: int, dialect: str | None = None, allowed_schemas=None) -> str:
    """Validate and normalize a read-only query. Raises SqlGuardError on anything unsafe.

    The statement is parsed (sqlglot); it must be exactly one SELECT / WITH...SELECT / set operation,
    contain only allowlisted functions and plain table references, and its row limit is capped at
    ``max_rows``. If ``allowed_schemas`` is given, qualified table names must use one of those catalog/schema names. The returned SQL is re-generated from the validated tree, so what runs is what was checked.
    """
    if max_rows < 1:
        raise SqlGuardError("max_rows must be positive")
    sqlglot, exp = _sqlglot()
    text = (sql or "").strip()
    if "\x00" in text:
        raise SqlGuardError("invalid character in query")
    if not text:
        raise SqlGuardError("empty query")
    read = dialect if dialect in _known_dialects(sqlglot) else None
    try:
        parsed = [e for e in sqlglot.parse(text, read=read) if e is not None]
    except Exception as e:  # sqlglot raises ParseError / TokenError
        raise SqlGuardError("query could not be parsed") from e
    if len(parsed) != 1:
        raise SqlGuardError("exactly one statement is allowed" if parsed else "empty query")
    root = parsed[0]
    allowed = None if allowed_schemas is None else {str(a).lower() for a in allowed_schemas}
    if not isinstance(root, _node_types(exp, _ROOT_NODE_NAMES)):
        raise SqlGuardError("only SELECT / WITH queries are allowed")
    forbidden = _node_types(exp, _FORBIDDEN_NODE_NAMES)
    for node in root.walk():
        if isinstance(node, forbidden):
            raise SqlGuardError(f"forbidden construct in query: {type(node).__name__}")
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).upper()
            if name.replace("_", "") not in _ALLOWED_NORMALIZED:
                raise SqlGuardError(f"function not allowed: {name.lower()}")
        elif isinstance(node, exp.Table):
            _check_table(node, exp, allowed)
    _cap_limit(root, max_rows, exp)
    return root.sql(dialect=read, comments=False)


def _known_dialects(sqlglot) -> set[str]:
    try:
        from sqlglot.dialects.dialect import Dialect
        return set(Dialect.classes)
    except Exception:
        return set()


def _check_table(node, exp, allowed_schemas=None) -> None:
    """Reject file-like or URL-like references, system tables, and (when configured) unlisted schemas."""
    for part in ("this", "db", "catalog"):
        v = node.args.get(part)
        if v is None:
            continue
        if not isinstance(v, exp.Identifier):
            raise SqlGuardError("table functions and file references are not allowed")
        name = v.name
        low = name.lower()
        if not _TABLE_PART.match(name):
            raise SqlGuardError("invalid table reference")
        if low in _URL_SCHEMES and part != "this":
            raise SqlGuardError("file and URL references are not allowed")
        if part == "this" and node.args.get("db") is not None and low in _FILE_EXTENSIONS:
            raise SqlGuardError("file references are not allowed")
        if part in ("db", "catalog") and low in _SYSTEM_SCHEMAS:
            raise SqlGuardError("system schemas are not allowed")
        if allowed_schemas is not None and part in ("db", "catalog") and low not in allowed_schemas:
            raise SqlGuardError("schema is not in the allowed list")
        if part == "this" and (low.startswith(_SYSTEM_PREFIXES) or low in _SYSTEM_SCHEMAS):
            raise SqlGuardError("system tables are not allowed")


def _cap_limit(root, max_rows: int, exp) -> None:
    lim = root.args.get("limit")
    n = None
    if lim is not None:
        e = lim.args.get("expression")
        if isinstance(e, exp.Literal) and not e.is_string and e.name.isdigit():
            n = int(e.name)
    if n is None or n > max_rows:
        root.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))


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

    def __init__(self, backend, model, *, max_rows: int = 100, sql_capability: str = "draft",
                 allowed_schemas=None):
        self.backend = backend      # WarehouseBackend: schema() + run_sql() + dialect()
        self.model = model          # ModelPlugin: generates SQL (any provider — Bedrock or local)
        self.max_rows = max_rows
        self.sql_capability = sql_capability
        # optional config: catalog/schema names generated SQL may reference (None = any non-file name)
        self.allowed_schemas = allowed_schemas if allowed_schemas is not None else getattr(backend, "allowed_schemas", None)

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
        sql = guard_sql(raw, min(k, self.max_rows) if k else self.max_rows, self.backend.dialect(),
                        allowed_schemas=self.allowed_schemas)
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
