"""Parquet corpus format — the columnar interchange + fast bulk-load path.

A normalized corpus of N chunks is otherwise N individual `.txt` files (one filesystem
entry each), which is slow to ship and slow to ingest at scale — every chunk is a separate
open/read. A single `corpus.parquet` (columns: chunk_id, filename, text, ts) is columnar,
compressed, one file, and DuckDB loads it in a single `read_parquet` call rather than
walking thousands of files. This module is the writer/reader for that format.

Note: the PERSISTENT BM25 index stays in DuckDB's native `.duckdb` file — Parquet has no
indexes, so it can't hold the FTS index. Parquet is purely the corpus/chunk interchange
and bulk-load format; DuckDB reads it, then builds its index over it.

Backend-agnostic: uses whichever of polars / pyarrow / duckdb is installed (any one of the
optional extras suffices), so it works in the ingest environment without pinning a lib.
"""
from __future__ import annotations

from pathlib import Path

# the columns of a corpus row (matches the store insert order)
COLUMNS = ("chunk_id", "filename", "text", "ts")
PARQUET_NAME = "corpus.parquet"   # canonical name inside a corpus dir


def parquet_available() -> bool:
    return _backend() is not None


def _backend() -> str | None:
    for mod in ("polars", "pyarrow", "duckdb"):
        try:
            __import__(mod)
            return mod
        except Exception:
            continue
    return None


def _norm(rows: list[dict]) -> list[dict]:
    return [{"chunk_id": r["chunk_id"], "filename": r.get("filename", ""),
             "text": r.get("text", ""), "ts": float(r.get("ts") or 0.0)} for r in rows]


def write_corpus_parquet(rows: list[dict], path: str | Path) -> int:
    """Write corpus rows to a Parquet file. Returns the row count. Raises if no backend."""
    path = str(path)
    rows = _norm(rows)
    backend = _backend()
    if backend == "polars":
        import polars as pl
        pl.DataFrame(rows, schema=list(COLUMNS)).write_parquet(path)
    elif backend == "pyarrow":
        import pyarrow as pa
        import pyarrow.parquet as pq
        cols = {c: [r[c] for r in rows] for c in COLUMNS}
        pq.write_table(pa.table(cols), path)
    elif backend == "duckdb":
        import duckdb
        con = duckdb.connect()
        con.execute("CREATE TABLE t (chunk_id VARCHAR, filename VARCHAR, text VARCHAR, ts DOUBLE)")
        con.executemany("INSERT INTO t VALUES (?,?,?,?)",
                        [(r["chunk_id"], r["filename"], r["text"], r["ts"]) for r in rows])
        con.execute("COPY t TO ? (FORMAT PARQUET)", [path])
        con.close()
    else:
        raise RuntimeError("Parquet needs one of polars / pyarrow / duckdb — "
                           "`pip install context-runtime[duckdb]` (or polars/pyarrow).")
    return len(rows)


def read_corpus_parquet(path: str | Path) -> list[dict]:
    """Read corpus rows from a Parquet file into dicts. Raises if no backend."""
    path = str(path)
    backend = _backend()
    if backend == "polars":
        import polars as pl
        return pl.read_parquet(path).select(list(COLUMNS)).to_dicts()
    if backend == "pyarrow":
        import pyarrow.parquet as pq
        t = pq.read_table(path, columns=list(COLUMNS)).to_pydict()
        n = len(t["chunk_id"])
        return [{c: t[c][i] for c in COLUMNS} for i in range(n)]
    if backend == "duckdb":
        import duckdb
        con = duckdb.connect()
        rows = con.execute(
            "SELECT chunk_id, filename, text, ts FROM read_parquet(?)", [path]).fetchall()
        con.close()
        return [dict(zip(COLUMNS, r)) for r in rows]
    raise RuntimeError("Parquet needs one of polars / pyarrow / duckdb to read a .parquet corpus.")


def resolve_parquet(path: str | Path) -> Path | None:
    """If `path` is a .parquet file, or a directory containing corpus.parquet, return that
    Parquet path; else None (caller falls back to the per-file .txt walk)."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".parquet":
        return p
    if p.is_dir() and (p / PARQUET_NAME).is_file():
        return p / PARQUET_NAME
    return None
