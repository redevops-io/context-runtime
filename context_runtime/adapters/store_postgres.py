"""Postgres-backed store — a flat full-text index over `tsvector` (StorePlugin surface).

The counterpart to `store_duckdb` for users whose data already lives in Postgres: the same
lexical retrieval, backed by a `tsvector` GIN index and ranked with `ts_rank_cd`. Drops
into `ShardedRetriever` as a shard exactly like the in-memory / DuckDB stores, so the
heterogeneous-corpus results (flat-mixed vs coverage-routed) reproduce on a Postgres
implementation too.

    store = PostgresStore(dsn="postgresql://user:pw@host/db", table="docs")
    store.index("/path/to/corpus")
    hits = store.search("capital expenditure", k=5)

`psycopg` (v3) is an OPTIONAL dependency — deferred import; `PostgresStore(...)` raises a
clear error if it's absent. Recency (ORDER BY ts) lives on the same table, so the three
chat-memory recall modes sit on one Postgres table (semantic = tsvector, entity = a tag join).
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path

from ..types import Hit, PluginInfo


def _psycopg():
    try:
        import psycopg
        return psycopg
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "PostgresStore needs the optional `psycopg` dependency — "
            "`pip install context-runtime[postgres]` (or `pip install psycopg[binary]`).") from e


class PostgresStore:
    """BM25-style full-text retrieval over a Postgres `tsvector` GIN index."""

    def __init__(self, docs: list[dict] | None = None, *, dsn: str,
                 table: str = "cr_docs", ts_config: str = "english", source: str = "postgres"):
        if not table.isidentifier():
            raise ValueError(f"unsafe table name {table!r}")
        self.source = source
        self.table = table
        self.ts_config = ts_config
        # Concurrency: a single psycopg connection is not safe for concurrent use. Use a
        # ConnectionPool when psycopg_pool is installed (real parallel queries under the
        # concurrent control plane); otherwise fall back to one connection + a lock (correct,
        # but serialized). Pool size via CR_PG_POOL (default 8).
        self._pool = None
        self._conn = None
        self._lock = threading.Lock()
        try:
            from psycopg_pool import ConnectionPool
            size = int(os.getenv("CR_PG_POOL", "8"))
            self._pool = ConnectionPool(dsn, min_size=1, max_size=max(1, size),
                                        kwargs={"autocommit": True}, open=True)
        except Exception:
            self._conn = _psycopg().connect(dsn, autocommit=True)
        with self._connection() as conn:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "chunk_id text PRIMARY KEY, filename text, body text, ts double precision, "
                "tsv tsvector)")
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {table}_tsv_idx ON {table} USING GIN(tsv)")
        if docs:
            self._insert(docs)

    @contextmanager
    def _connection(self):
        """Yield a usable connection — from the pool (parallel) or the locked single one."""
        if self._pool is not None:
            with self._pool.connection() as conn:
                yield conn
        else:
            with self._lock:
                yield self._conn

    def _insert(self, docs: list[dict]) -> int:
        rows = [(d["chunk_id"], d.get("filename", ""), d.get("text", ""), float(d.get("ts") or 0.0))
                for d in docs]
        with self._connection() as conn, conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (chunk_id, filename, body, ts, tsv) "
                f"VALUES (%s, %s, %s, %s, to_tsvector(%s, %s)) "
                "ON CONFLICT (chunk_id) DO UPDATE SET "
                "filename=EXCLUDED.filename, body=EXCLUDED.body, ts=EXCLUDED.ts, tsv=EXCLUDED.tsv",
                [(cid, fn, body, ts, self.ts_config, body) for cid, fn, body, ts in rows])
        return len(rows)

    def index(self, path: str) -> dict:
        """Index a folder of text/markdown files (one chunk per file, matching InMemoryStore)."""
        p = Path(path).expanduser()
        docs = []
        for fp in sorted(p.rglob("*")):
            if fp.suffix.lower() in (".md", ".txt", ".rst") and fp.is_file():
                docs.append({"chunk_id": f"{fp.name}::0", "filename": fp.name,
                             "text": fp.read_text(errors="ignore"), "ts": fp.stat().st_mtime})
        n = self._insert(docs)
        return {"files": n, "chunks": n}

    def search(self, query: str, k: int, method: str = "bm25") -> list[Hit]:
        """Full-text rank (method='bm25'/'hybrid') or most-recent (method='recency')."""
        if not query.strip():
            return []
        if method == "recency":
            with self._connection() as conn:
                rows = conn.execute(
                    f"SELECT chunk_id, filename, body, ts FROM {self.table} ORDER BY ts DESC LIMIT %s",
                    (k,)).fetchall()
            return [Hit(chunk_id=r[0], filename=r[1], text=r[2], score=0.0, created_at=str(r[3]))
                    for r in rows]
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT chunk_id, filename, body, ts_rank_cd(tsv, q) AS score "
                f"FROM {self.table}, plainto_tsquery(%s, %s) q "
                "WHERE tsv @@ q ORDER BY score DESC LIMIT %s",
                (self.ts_config, query, k)).fetchall()
        return [Hit(chunk_id=r[0], filename=r[1], text=r[2], score=round(float(r[3]), 4)) for r in rows]

    def info(self) -> PluginInfo:
        return PluginInfo(name="postgres", kind="store", version="0.1",
                          capabilities=frozenset({"bm25", "hybrid", "recency", "persistent"}))
