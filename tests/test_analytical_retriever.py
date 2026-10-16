"""The analytical representation, made real: guarded text-to-SQL over a WarehouseBackend.

Runs the whole path offline against a real in-memory DuckDB (the provider-neutral engine + local
backend), with a fake model standing in for the text-to-SQL generator. Also covers the security
guard and the Athena backend's request/response handling with a fake boto3 client.
"""
import pytest

from context_runtime.adapters.store_analytical import (
    AnalyticalRetriever,
    SqlGuardError,
    guard_sql,
    strip_sql,
)
from context_runtime.plugins import base


# ── the security guard: the whole point of the feature ──────────────────────────────────────────
def test_guard_allows_select_and_appends_limit():
    assert guard_sql("SELECT count(*) FROM invoices", 50) == "SELECT COUNT(*) FROM invoices LIMIT 50"
    assert guard_sql("WITH x AS (SELECT 1) SELECT * FROM x LIMIT 5", 50).endswith("LIMIT 5")


@pytest.mark.parametrize("bad", [
    "DELETE FROM invoices", "UPDATE t SET x=1", "INSERT INTO t VALUES (1)", "DROP TABLE t",
    "SELECT 1; DROP TABLE t", "CREATE TABLE t (a int)", "TRUNCATE t", "GRANT ALL ON t TO x",
    "ALTER TABLE t ADD c int", "PRAGMA table_info(t)", "SET x=1",
])
def test_guard_rejects_mutations_and_multi_statement(bad):
    with pytest.raises(SqlGuardError):
        guard_sql(bad, 50)


def test_strip_sql_unwraps_fences_and_prose():
    assert strip_sql("Here you go:\n```sql\nSELECT 1\n```") == "SELECT 1"
    assert strip_sql("SELECT a FROM t;") == "SELECT a FROM t"


# ── end-to-end over a real DuckDB, fake SQL generator ───────────────────────────────────────────
class FakeSqlModel:
    """A ModelPlugin stub that returns a fixed SQL string as its 'generation'."""
    def __init__(self, sql):
        self.sql = sql
        self.seen = []

    def complete(self, req):
        from context_runtime.types import ModelResult
        self.seen.append(req)
        return ModelResult(text=self.sql, model="fake", tier="cheap")

    def capabilities(self, model):
        from context_runtime.types import ModelCapabilities
        return ModelCapabilities()

    def count_tokens(self, text, model):
        return len(text) // 4

    def info(self):
        from context_runtime.types import PluginInfo
        return PluginInfo(name="fake", kind="model")


def _duckdb_warehouse():
    import duckdb
    from context_runtime.adapters.warehouse_duckdb import DuckDBWarehouse
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE invoices (id INTEGER, amount DECIMAL, status VARCHAR)")
    conn.execute("INSERT INTO invoices VALUES (1, 100, 'paid'), (2, 200, 'open'), (3, 50, 'open')")
    return DuckDBWarehouse(conn=conn)


def test_analytical_retriever_runs_aggregate_over_duckdb():
    wh = _duckdb_warehouse()
    # the schema is fed to the generator; here we hardcode the SQL it "produces"
    model = FakeSqlModel("SELECT status, count(*) AS n FROM invoices GROUP BY status ORDER BY status")
    r = AnalyticalRetriever(wh, model)
    assert isinstance(r, base.RetrieverPlugin)
    hits = r.search("how many invoices per status?", k=10, method="sql")
    # two groups: open=2, paid=1
    by = {h.meta["row"]["status"]: h.meta["row"]["n"] for h in hits}
    assert by == {"open": 2, "paid": 1}
    assert all(h.source == "analytical:duckdb" for h in hits)
    assert hits[0].meta["sql"].startswith("SELECT status")
    # the generator saw the real schema
    assert "invoices(" in model.seen[0].messages[0]["content"]


def test_analytical_retriever_blocks_a_malicious_generation():
    wh = _duckdb_warehouse()
    r = AnalyticalRetriever(wh, FakeSqlModel("DELETE FROM invoices"))
    with pytest.raises(SqlGuardError):
        r.search("delete everything", k=5, method="sql")


def test_duckdb_schema_summary():
    wh = _duckdb_warehouse()
    schema = wh.schema()
    assert "invoices(" in schema and "amount" in schema


# ── Athena backend with a fake boto3 client ─────────────────────────────────────────────────────
class FakeAthena:
    """Query-aware fake: returns catalog rows for information_schema, aggregate rows otherwise."""
    def __init__(self):
        self.started = []
        self._sql = {}
        self._n = 0

    def start_query_execution(self, **kw):
        self.started.append(kw)
        self._n += 1
        qid = f"q-{self._n}"
        self._sql[qid] = kw["QueryString"]
        return {"QueryExecutionId": qid}

    def get_query_execution(self, QueryExecutionId):
        return {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}

    def get_query_results(self, QueryExecutionId, MaxResults):
        sql = self._sql.get(QueryExecutionId, "")
        if "information_schema" in sql:
            rows = [
                {"Data": [{"VarCharValue": "table_name"}, {"VarCharValue": "column_name"}, {"VarCharValue": "data_type"}]},
                {"Data": [{"VarCharValue": "invoices"}, {"VarCharValue": "status"}, {"VarCharValue": "varchar"}]},
                {"Data": [{"VarCharValue": "invoices"}, {"VarCharValue": "amount"}, {"VarCharValue": "decimal"}]},
            ]
        else:
            rows = [
                {"Data": [{"VarCharValue": "status"}, {"VarCharValue": "n"}]},
                {"Data": [{"VarCharValue": "open"}, {"VarCharValue": "2"}]},
                {"Data": [{"VarCharValue": "paid"}, {"VarCharValue": "1"}]},
            ]
        return {"ResultSet": {"Rows": rows}}


def test_athena_backend_executes_and_parses():
    from context_runtime.providers.aws.athena_backend import AthenaBackend
    fake = FakeAthena()
    be = AthenaBackend(client=fake, database="lake", output_location="s3://out/",
                       sleep=lambda s: None)
    rows = be.run_sql("SELECT status, count(*) n FROM invoices GROUP BY status", max_rows=10)
    assert rows == [{"status": "open", "n": "2"}, {"status": "paid", "n": "1"}]
    assert fake.started[0]["QueryExecutionContext"] == {"Database": "lake"}
    assert be.dialect() == "athena"


def test_athena_analytical_end_to_end():
    from context_runtime.providers.aws.athena_backend import AthenaBackend
    be = AthenaBackend(client=FakeAthena(), database="lake", output_location="s3://out/",
                       sleep=lambda s: None)
    r = AnalyticalRetriever(be, FakeSqlModel("SELECT status, count(*) n FROM invoices GROUP BY status"))
    hits = r.search("per status", k=10, method="sql")
    assert {h.meta["row"]["status"] for h in hits} == {"open", "paid"}
    assert all(h.source == "analytical:athena" for h in hits)


# ── shared allow/deny vectors (also consumed by other ports) ────────────────────────────────────
def _vectors():
    import json
    from pathlib import Path
    doc = json.loads((Path(__file__).parent / "vectors" / "sql_guard.json").read_text())
    return doc["max_rows"], doc["vectors"]


def test_guard_matches_shared_vectors():
    pytest.importorskip("sqlglot")
    max_rows, vectors = _vectors()
    import re
    for v in vectors:
        if v["allow"]:
            out = guard_sql(v["sql"], max_rows)
            if "limit" in v:
                m = re.search(r"LIMIT\s+(\d+)", out, re.I)
                assert m and int(m.group(1)) == v["limit"], (v["sql"], out)
        else:
            with pytest.raises(SqlGuardError):
                guard_sql(v["sql"], max_rows)


def test_guard_caps_caller_supplied_limit():
    assert guard_sql("SELECT * FROM t LIMIT 100000", 50).endswith("LIMIT 50")
    assert guard_sql("SELECT * FROM t LIMIT 5", 50).endswith("LIMIT 5")


def test_guard_accepts_backend_dialects():
    assert guard_sql("SELECT status, count(*) n FROM invoices GROUP BY status", 10, "athena").endswith("LIMIT 10")
    assert guard_sql("SELECT 1", 10, "not-a-dialect").endswith("LIMIT 10")


def test_duckdb_warehouse_disables_external_access():
    duckdb = pytest.importorskip("duckdb")
    from context_runtime.adapters.warehouse_duckdb import DuckDBWarehouse
    wh = DuckDBWarehouse()
    assert wh.run_sql("SELECT 1 AS a") == [{"a": 1}]
    with pytest.raises(duckdb.Error):
        wh.run_sql("SELECT * FROM read_csv('/etc/hostname')")
    with pytest.raises(duckdb.Error):
        wh.run_sql("SET enable_external_access=true")
