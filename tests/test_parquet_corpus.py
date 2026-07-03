"""Parquet corpus format: columnar interchange + fast bulk-load into the stores. Skips
cleanly when no parquet backend (polars/pyarrow/duckdb) is installed."""
from __future__ import annotations

import pytest

from context_runtime.adapters.store_inmemory import InMemoryStore
from context_runtime.ingest.parquet_corpus import (
    PARQUET_NAME,
    parquet_available,
    read_corpus_parquet,
    resolve_parquet,
    write_corpus_parquet,
)

pytestmark = pytest.mark.skipif(not parquet_available(),
                                reason="no parquet backend (polars/pyarrow/duckdb) installed")

ROWS = [{"chunk_id": f"c{i}", "filename": f"c{i}.txt",
         "text": f"passage {i} about revenue growth and operating margins", "ts": 0.0}
        for i in range(6)]


def test_write_read_roundtrip(tmp_path):
    p = tmp_path / PARQUET_NAME
    n = write_corpus_parquet(ROWS, p)
    assert n == 6 and p.is_file()
    back = read_corpus_parquet(p)
    assert [r["chunk_id"] for r in back] == [r["chunk_id"] for r in ROWS]
    assert back[0]["text"] == ROWS[0]["text"]


def test_resolve_parquet_dir_and_file(tmp_path):
    p = tmp_path / PARQUET_NAME
    write_corpus_parquet(ROWS, p)
    assert resolve_parquet(tmp_path) == p     # a dir containing corpus.parquet
    assert resolve_parquet(p) == p            # a .parquet file directly
    assert resolve_parquet(tmp_path / "nope") is None


def test_inmemory_store_bulk_loads_parquet(tmp_path):
    write_corpus_parquet(ROWS, tmp_path / PARQUET_NAME)
    store = InMemoryStore([])
    rep = store.index(str(tmp_path))          # detects corpus.parquet, bulk-loads columnar
    assert rep["chunks"] == 6 and "parquet" in rep
    hits = store.search("revenue operating margins", 3, "bm25")
    assert hits and all(h.chunk_id.startswith("c") for h in hits)


def test_duckdb_store_reads_parquet(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    from context_runtime.adapters.store_duckdb import DuckDBStore
    write_corpus_parquet(ROWS, tmp_path / PARQUET_NAME)
    store = DuckDBStore(path=":memory:")
    rep = store.index(str(tmp_path))          # DuckDB native read_parquet bulk load
    assert rep["chunks"] == 6 and "parquet" in rep
    assert store.search("revenue", 2)
