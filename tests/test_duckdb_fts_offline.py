"""The DuckDB store never downloads native code by default and falls back to in-memory BM25 visibly."""

from __future__ import annotations

import logging
import os
from pathlib import Path

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


# ───────────────────────── unavailable-extension paths and engine/fallback parity ─────────────────────────

PARITY_DOCS = [
    {"chunk_id": "p0", "filename": "p0.md", "ts": 1.0,
     "text": "quarterly revenue rose on strong subscription renewals and revenue recognition changes"},
    {"chunk_id": "p1", "filename": "p1.md", "ts": 2.0,
     "text": "the outage postmortem lists the incident timeline and the failed database failover"},
    {"chunk_id": "p2", "filename": "p2.md", "ts": 3.0,
     "text": "hiring plan for backend engineers and a new site reliability lead"},
    {"chunk_id": "p3", "filename": "p3.md", "ts": 4.0,
     "text": "marketing budget for the spring campaign covers events and webinars"},
    {"chunk_id": "p4", "filename": "p4.md", "ts": 5.0,
     "text": "database migration runbook with rollback steps for the failover cluster"},
]
PARITY_QUERIES = ["revenue", "outage incident", "hiring engineers", "webinars", "failover database"]


def _warnings(caplog):
    return [r for r in caplog.records if r.name == "context_runtime.duckdb" and r.levelno == logging.WARNING]


def _use_everything(store):
    store._insert([{"chunk_id": "z", "filename": "z.md", "text": "revenue forecast", "ts": 9.0}])
    store._reindex()
    store.search("revenue", 3)
    store.search("anything", 3, "recency")


def test_network_blocked_install_falls_back_with_a_single_warning(monkeypatch, caplog):
    """install_fts=True but the download fails (no network): one warning line, then in-memory BM25."""
    monkeypatch.delenv(store_duckdb.FTS_EXTENSION_ENV, raising=False)
    conn = _RecordingConn()
    monkeypatch.setattr(store_duckdb, "_duckdb", lambda: _FakeDuckDB(conn))
    with caplog.at_level(logging.INFO, logger="context_runtime.duckdb"):
        store = DuckDBStore(install_fts=True)
    assert any(s == "INSTALL fts" for s in conn.statements)         # it did try, and it was refused
    assert store.fts_available is False
    assert len(_warnings(caplog)) == 1
    assert "in-memory BM25" in _warnings(caplog)[0].getMessage()


def test_a_corrupt_extension_file_falls_back_with_a_single_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv(store_duckdb.INSTALL_FTS_ENV, raising=False)
    junk = tmp_path / "fts.duckdb_extension"
    junk.write_bytes(b"this is not a duckdb extension")
    with caplog.at_level(logging.INFO, logger="context_runtime.duckdb"):
        store = DuckDBStore(PARITY_DOCS, fts_extension=str(junk))
        _use_everything(store)
    assert store.fts_available is False
    assert len(_warnings(caplog)) == 1                              # not repeated by later inserts or searches
    assert store.search("webinars", 3)[0].chunk_id == "p3"


def test_environment_variable_with_a_wrong_path_falls_back_with_a_single_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(store_duckdb.FTS_EXTENSION_ENV, str(tmp_path / "nowhere" / "fts.duckdb_extension"))
    monkeypatch.setenv(store_duckdb.INSTALL_FTS_ENV, "1")           # a configured file wins; no install attempt
    with caplog.at_level(logging.INFO, logger="context_runtime.duckdb"):
        store = DuckDBStore(PARITY_DOCS)
        _use_everything(store)
    assert store.fts_available is False and len(_warnings(caplog)) == 1


def test_install_disabled_and_nothing_installed_uses_a_private_extension_directory(tmp_path, monkeypatch, caplog):
    """Real DuckDB, an empty extension directory, install disabled: no download is possible."""
    monkeypatch.delenv(store_duckdb.INSTALL_FTS_ENV, raising=False)
    monkeypatch.delenv(store_duckdb.FTS_EXTENSION_ENV, raising=False)
    monkeypatch.setenv(store_duckdb.EXTENSION_DIR_ENV, str(tmp_path / "empty-extensions"))
    with caplog.at_level(logging.INFO, logger="context_runtime.duckdb"):
        store = DuckDBStore(PARITY_DOCS)
        _use_everything(store)
    assert store.fts_available is False and len(_warnings(caplog)) == 1
    assert not list((tmp_path / "empty-extensions").rglob("*.duckdb_extension"))     # nothing was fetched


def test_fallback_ranking_matches_the_engine_ranking_when_the_engine_is_available(tmp_path, caplog):
    """BM25 parity: for unambiguous queries the in-memory fallback returns the same top-ranked chunks as the
    DuckDB `fts` engine. (Scores are not compared: the two tokenisers and stemmers differ.)"""
    engine = DuckDBStore(PARITY_DOCS, install_fts=True)
    if not engine.fts_available:
        pytest.skip("UNVERIFIED on this platform: the DuckDB fts extension could not be loaded or installed here")
    fallback = DuckDBStore(PARITY_DOCS, fts_extension=str(tmp_path / "absent.duckdb_extension"))
    assert fallback.fts_available is False
    for q in PARITY_QUERIES:
        e = [h.chunk_id for h in engine.search(q, 3)]
        f = [h.chunk_id for h in fallback.search(q, 3)]
        assert e and f and e[0] == f[0], (q, e, f)                 # same best chunk
        assert set(e[:2]) == set(f[:2]), (q, e, f)                  # same top-2 membership


def _installed_fts_file():
    """Path of the fts extension DuckDB can load on this machine, or None (loads it once to find out)."""
    import duckdb
    con = duckdb.connect()
    try:
        con.execute("SET autoinstall_known_extensions=false")
        con.execute("LOAD fts")
        row = con.execute("SELECT install_path FROM duckdb_extensions() WHERE extension_name='fts'").fetchone()
        return Path(row[0]) if row and row[0] else None
    except Exception:  # noqa: BLE001
        return None
    finally:
        con.close()


def test_pre_provisioned_extension_directory_and_file_work_offline(tmp_path, monkeypatch, caplog):
    """The documented offline recipe: copy the extension file out of a machine that has it, then point
    CONTEXT_RUNTIME_DUCKDB_FTS_EXTENSION at the file, or CONTEXT_RUNTIME_DUCKDB_EXTENSION_DIR at a copy of the
    <version>/<platform>/ tree. Install stays disabled: the engine path must come up without a download."""
    if not os.environ.get(store_duckdb.INSTALL_FTS_ENV):
        DuckDBStore(install_fts=True)                         # populate the local copy first (needs network once)
    src = _installed_fts_file()
    if src is None or not src.is_file():
        pytest.skip("UNVERIFIED on this platform: no installed fts extension to copy from")
    monkeypatch.delenv(store_duckdb.INSTALL_FTS_ENV, raising=False)

    # 1. a single extension file, from anywhere
    copy = tmp_path / "bundle" / "fts.duckdb_extension"
    copy.parent.mkdir()
    copy.write_bytes(src.read_bytes())
    monkeypatch.delenv(store_duckdb.EXTENSION_DIR_ENV, raising=False)
    with caplog.at_level(logging.WARNING, logger="context_runtime.duckdb"):
        by_file = DuckDBStore(PARITY_DOCS, fts_extension=str(copy), install_fts=False)
    assert by_file.fts_available is True and not _warnings(caplog)
    assert by_file.search("webinars", 1)[0].chunk_id == "p3"

    # 2. an extension directory with DuckDB's own layout: <dir>/<version>/<platform>/fts.duckdb_extension
    tree = tmp_path / "ext" / src.parent.parent.name / src.parent.name
    tree.mkdir(parents=True)
    (tree / src.name).write_bytes(src.read_bytes())
    monkeypatch.setenv(store_duckdb.EXTENSION_DIR_ENV, str(tmp_path / "ext"))
    monkeypatch.delenv(store_duckdb.FTS_EXTENSION_ENV, raising=False)
    by_dir = DuckDBStore(PARITY_DOCS, install_fts=False)
    assert by_dir.fts_available is True
    assert by_dir.search("webinars", 1)[0].chunk_id == "p3"
