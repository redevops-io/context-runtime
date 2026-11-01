"""The DuckDB store never downloads native code by default and falls back to in-memory BM25 visibly."""

from __future__ import annotations

import logging
import os

import pytest

pytest.importorskip("duckdb")

from context_runtime.adapters import store_duckdb  # noqa: E402
from context_runtime.adapters.store_duckdb import DuckDBStore  # noqa: E402

DOCS = [
    {"chunk_id": "a::0", "filename": "a.md", "text": "quarterly revenue and operating margin rose", "ts": 1.0},
    {"chunk_id": "b::0", "filename": "b.md", "text": "the incident review covered the outage", "ts": 2.0},
    {"chunk_id": "c::0", "filename": "c.md", "text": "revenue recognition policy for subscriptions", "ts": 3.0},
]


class _Result:
    def fetchone(self):
        return (0,)

    def fetchall(self):
        return []


class _RecordingConn:
    """Stands in for a DuckDB connection with no `fts` extension available and records every statement."""

    def __init__(self):
        self.statements: list[str] = []

    def execute(self, sql, *a):
        self.statements.append(sql)
        if sql.startswith(("LOAD", "INSTALL")):
            raise RuntimeError("Extension 'fts' not found (no network in this test)")
        return _Result()

    def executemany(self, sql, rows):
        self.statements.append(sql)

    def cursor(self):
        return self


class _FakeDuckDB:
    def __init__(self, conn):
        self._conn = conn

    def connect(self, path):
        return self._conn


def test_default_path_never_issues_install(monkeypatch, caplog):
    monkeypatch.delenv(store_duckdb.INSTALL_FTS_ENV, raising=False)
    monkeypatch.delenv(store_duckdb.FTS_EXTENSION_ENV, raising=False)
    conn = _RecordingConn()
    monkeypatch.setattr(store_duckdb, "_duckdb", lambda: _FakeDuckDB(conn))
    with caplog.at_level(logging.WARNING, logger="context_runtime.duckdb"):
        store = DuckDBStore()
    assert not any("INSTALL" in s for s in conn.statements), conn.statements
    assert any(s == "LOAD fts" for s in conn.statements)
    assert store.fts_available is False
    assert any("in-memory BM25" in r.getMessage() for r in caplog.records)


def test_install_is_only_attempted_when_explicitly_enabled(monkeypatch):
    monkeypatch.delenv(store_duckdb.FTS_EXTENSION_ENV, raising=False)
    conn = _RecordingConn()
    monkeypatch.setattr(store_duckdb, "_duckdb", lambda: _FakeDuckDB(conn))
    DuckDBStore(install_fts=True)
    assert any(s == "INSTALL fts" for s in conn.statements)
    conn2 = _RecordingConn()
    monkeypatch.setattr(store_duckdb, "_duckdb", lambda: _FakeDuckDB(conn2))
    monkeypatch.setenv(store_duckdb.INSTALL_FTS_ENV, "1")
    DuckDBStore()
    assert any(s == "INSTALL fts" for s in conn2.statements)


def test_a_configured_extension_file_is_the_only_thing_tried(tmp_path, monkeypatch):
    conn = _RecordingConn()
    monkeypatch.setattr(store_duckdb, "_duckdb", lambda: _FakeDuckDB(conn))
    ext = tmp_path / "it's" / "fts.duckdb_extension"
    DuckDBStore(fts_extension=str(ext))
    loads = [s for s in conn.statements if s.startswith("LOAD")]
    assert loads == [f"LOAD '{str(ext).replace(chr(39), chr(39) * 2)}'"]      # quote escaped, no fallback LOAD
    assert not any("INSTALL" in s for s in conn.statements)


def test_extension_directory_is_set_before_loading(tmp_path, monkeypatch):
    conn = _RecordingConn()
    monkeypatch.setattr(store_duckdb, "_duckdb", lambda: _FakeDuckDB(conn))
    monkeypatch.setenv(store_duckdb.EXTENSION_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(store_duckdb.FTS_EXTENSION_ENV, raising=False)
    DuckDBStore()
    i_set = next(i for i, s in enumerate(conn.statements) if s.startswith("SET extension_directory"))
    i_load = next(i for i, s in enumerate(conn.statements) if s == "LOAD fts")
    assert i_set < i_load


def test_fallback_serves_bm25_and_recency_from_the_same_rows(tmp_path, caplog):
    missing = tmp_path / "no-such" / "fts.duckdb_extension"
    with caplog.at_level(logging.WARNING, logger="context_runtime.duckdb"):
        store = DuckDBStore(DOCS, fts_extension=str(missing))
    assert store.fts_available is False
    assert sum("in-memory BM25" in r.getMessage() for r in caplog.records) == 1     # one visible line
    hits = store.search("revenue", 5)
    assert {h.chunk_id for h in hits} == {"a::0", "c::0"}
    assert [h.chunk_id for h in store.search("anything", 2, "recency")] == ["c::0", "b::0"]
    # rows added later are visible to the fallback
    store._insert([{"chunk_id": "d::0", "filename": "d.md", "text": "revenue forecast", "ts": 4.0}])
    assert "d::0" in {h.chunk_id for h in store.search("revenue", 5)}


def test_fallback_indexes_a_corpus_folder(tmp_path):
    (tmp_path / "x.md").write_text("zanzibar harbour tariffs", encoding="utf-8")
    (tmp_path / "y.md").write_text("unrelated notes", encoding="utf-8")
    store = DuckDBStore(fts_extension=str(tmp_path / "missing.duckdb_extension"))
    assert store.index(str(tmp_path))["chunks"] == 2
    assert [h.filename for h in store.search("zanzibar", 3)] == ["x.md"]


@pytest.mark.skipif(os.environ.get(store_duckdb.INSTALL_FTS_ENV, "").strip() not in ("1", "true", "yes"),
                    reason=f"{store_duckdb.INSTALL_FTS_ENV}=1 not set: the engine fts path needs a one-time download")
def test_engine_fts_path_when_the_extension_may_be_installed():
    store = DuckDBStore(DOCS)
    assert store.fts_available is True
    assert {h.chunk_id for h in store.search("revenue", 5)} == {"a::0", "c::0"}
