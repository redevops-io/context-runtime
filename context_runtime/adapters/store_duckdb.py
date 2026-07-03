"""DuckDB-backed store — a persistent BM25 index (StorePlugin/RetrieverPlugin surface).

The in-memory store rebuilds its BM25 stats in the process; DuckDB gives the same lexical
retrieval backed by a real, persistent index (its `fts` extension computes Okapi BM25 in
the engine). This is the concrete backend the 3-index chat memory and the sharded
retriever map onto for real corpora: one DuckDB file per shard / per user's local data.

    store = DuckDBStore(path="finance.duckdb")   # or ":memory:"
    store.index("/path/to/corpus")               # folder of .md/.txt/.rst
    hits = store.search("capital expenditure", k=5)

`duckdb` is an OPTIONAL dependency (the `polars`-style extras pattern) — the import is
deferred so the module loads without it; `DuckDBStore(...)` raises a clear error if it's
absent. Recency (ORDER BY ts) and an entity-tag join live on the same table, so the three
chat-memory recall modes all sit on one DuckDB file.
"""
from __future__ import annotations

from pathlib import Path

from ..types import Hit, PluginInfo


def _duckdb():
    try:
        import duckdb
        return duckdb
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "DuckDBStore needs the optional `duckdb` dependency — "
            "`pip install context-runtime[duckdb]` (or `pip install duckdb`).") from e


class DuckDBStore:
    """BM25 retrieval over a DuckDB `fts` index. Persistent when given a file path."""

    def __init__(self, docs: list[dict] | None = None, *, path: str = ":memory:",
                 source: str = "duckdb"):
        import threading
        self.source = source
        # A DuckDB connection object is not safe for concurrent use; reads go through
        # per-call .cursor() (independent connections to the same DB), and writes (insert/
        # reindex) are serialized by this lock so the control plane can serve concurrently.
        self._write_lock = threading.Lock()
        self._db = _duckdb().connect(path)
        self._db.execute("INSTALL fts; LOAD fts;")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS docs "
            "(chunk_id VARCHAR PRIMARY KEY, filename VARCHAR, text VARCHAR, ts DOUBLE)")
        self._n = self._db.execute("SELECT count(*) FROM docs").fetchone()[0]
        if docs:
            self._insert(docs)
            self._reindex()

    def _insert(self, docs: list[dict]) -> int:
        rows = [(d["chunk_id"], d.get("filename", ""), d.get("text", ""),
                 float(d.get("ts") or 0.0)) for d in docs]
        with self._write_lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO docs (chunk_id, filename, text, ts) VALUES (?,?,?,?)", rows)
            self._n = self._db.execute("SELECT count(*) FROM docs").fetchone()[0]
        return len(rows)

    def _reindex(self) -> None:
        # (re)build the BM25 index over the text column; overwrite so re-index is idempotent.
        with self._write_lock:
            self._db.execute("PRAGMA create_fts_index('docs','chunk_id','text', overwrite=1)")

    def index(self, path: str) -> dict:
        """Index a corpus. Fast path: a `corpus.parquet` (or a .parquet file) is bulk-loaded
        with DuckDB's native `read_parquet` in one shot (columnar, no per-file open); else a
        folder of text/markdown files (one chunk per file, matching InMemoryStore)."""
        from ..ingest.parquet_corpus import resolve_parquet
        pq = resolve_parquet(Path(path).expanduser())
        if pq is not None:
            with self._write_lock:
                self._db.execute(
                    "INSERT OR REPLACE INTO docs (chunk_id, filename, text, ts) "
                    "SELECT chunk_id, filename, text, COALESCE(ts, 0.0) FROM read_parquet(?)",
                    [str(pq)])
                self._n = self._db.execute("SELECT count(*) FROM docs").fetchone()[0]
            self._reindex()
            return {"files": 1, "chunks": self._n, "parquet": str(pq)}
        p = Path(path).expanduser()
        docs = []
        for fp in sorted(p.rglob("*")):
            if fp.suffix.lower() in (".md", ".txt", ".rst") and fp.is_file():
                docs.append({"chunk_id": f"{fp.name}::0", "filename": fp.name,
                             "text": fp.read_text(errors="ignore"), "ts": fp.stat().st_mtime})
        n = self._insert(docs)
        self._reindex()
        return {"files": n, "chunks": n}

    def search(self, query: str, k: int, method: str = "bm25") -> list[Hit]:
        """BM25 (method='bm25'/'hybrid') or most-recent (method='recency')."""
        if not query.strip() or self._n == 0:
            return []
        cur = self._db.cursor()   # independent connection to the same DB → concurrent reads
        try:
            if method == "recency":
                rows = cur.execute(
                    "SELECT chunk_id, filename, text, ts FROM docs ORDER BY ts DESC LIMIT ?",
                    [k]).fetchall()
                return [Hit(chunk_id=r[0], filename=r[1], text=r[2], score=0.0,
                            created_at=str(r[3])) for r in rows]
            rows = cur.execute(
                "SELECT chunk_id, filename, text, score FROM "
                "(SELECT *, fts_main_docs.match_bm25(chunk_id, ?) AS score FROM docs) sq "
                "WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?",
                [query, k]).fetchall()
            return [Hit(chunk_id=r[0], filename=r[1], text=r[2], score=round(float(r[3]), 4))
                    for r in rows]
        finally:
            cur.close()

    def info(self) -> PluginInfo:
        return PluginInfo(name="duckdb", kind="store", version="0.1",
                          capabilities=frozenset({"bm25", "hybrid", "recency", "persistent"}))
