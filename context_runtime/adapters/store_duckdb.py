"""DuckDB-backed store — a persistent BM25 index (StorePlugin/RetrieverPlugin surface).

The in-memory store rebuilds its BM25 stats in the process; DuckDB gives lexical
retrieval backed by a real, persistent index (its `fts` extension computes Okapi BM25 in
the engine). When the extension cannot be loaded the store falls back to an APPROXIMATE lexical search in
process (the in-memory store's tokeniser and BM25): same kind of ranking, but NOT score-compatible with the
engine and not identical in which queries match (see `_fallback_search`). This is the concrete backend the 3-index chat memory and the sharded
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

import logging
import os
from pathlib import Path

from ..types import Hit, PluginInfo

_log = logging.getLogger("context_runtime.duckdb")

#: Absolute path of a pre-provisioned ``fts.duckdb_extension`` (for example one shipped with an installer).
FTS_EXTENSION_ENV = "CONTEXT_RUNTIME_DUCKDB_FTS_EXTENSION"
#: A directory holding pre-provisioned extensions (DuckDB's ``extension_directory`` layout).
EXTENSION_DIR_ENV = "CONTEXT_RUNTIME_DUCKDB_EXTENSION_DIR"
#: Set to 1 to let the store ``INSTALL fts`` from the network when it is not already present. Off by default:
#: the default path never downloads native code.
INSTALL_FTS_ENV = "CONTEXT_RUNTIME_DUCKDB_INSTALL_FTS"


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
                 source: str = "duckdb", fts_extension: str | None = None,
                 install_fts: bool | None = None):
        import threading
        self.source = source
        # A DuckDB connection object is not safe for concurrent use; reads go through
        # per-call .cursor() (independent connections to the same DB), and writes (insert/
        # reindex) are serialized by this lock so the control plane can serve concurrently.
        self._write_lock = threading.Lock()
        self._db = _duckdb().connect(path)
        # BM25 runs in the engine when the `fts` extension loads; otherwise an approximate lexical fallback runs
        # in process (see _fallback_search). Nothing is downloaded unless install_fts / INSTALL_FTS_ENV says so.
        self._fts = self._load_fts(fts_extension, install_fts)
        self._fallback = None
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS docs "
            "(chunk_id VARCHAR PRIMARY KEY, filename VARCHAR, text VARCHAR, ts DOUBLE)")
        self._n = self._db.execute("SELECT count(*) FROM docs").fetchone()[0]
        if docs:
            self._insert(docs)
            self._reindex()

    def _load_fts(self, extension: str | None, install: bool | None) -> bool:
        """Load the `fts` extension without touching the network unless explicitly allowed.

        Order: an explicit extension file (argument or ``FTS_EXTENSION_ENV``) > a pre-provisioned extension
        directory (``EXTENSION_DIR_ENV``) > an extension DuckDB already has installed > an opt-in
        ``INSTALL fts`` (``install_fts=True`` or ``INSTALL_FTS_ENV=1``). When none works the store logs one
        line and uses the approximate in-memory lexical fallback.
        """
        db = self._db
        ext_file = extension or os.environ.get(FTS_EXTENSION_ENV, "").strip()
        ext_dir = os.environ.get(EXTENSION_DIR_ENV, "").strip()
        if install is None:
            install = os.environ.get(INSTALL_FTS_ENV, "").strip().lower() in ("1", "true", "yes")
        try:
            db.execute("SET autoinstall_known_extensions=false")   # never fetch behind our back
            if ext_file:
                db.execute(f"LOAD '{ext_file.replace(chr(39), chr(39) * 2)}'")
                _log.info("duckdb: fts extension loaded from %s", ext_file)
                return True
            if ext_dir:
                db.execute(f"SET extension_directory='{ext_dir.replace(chr(39), chr(39) * 2)}'")
            try:
                db.execute("LOAD fts")
            except Exception:  # noqa: BLE001 - not installed yet
                if not install:
                    raise
                db.execute("INSTALL fts")
                db.execute("LOAD fts")
            return True
        except Exception as e:  # noqa: BLE001
            _log.warning("duckdb: the fts extension is not available (%s: %s); using the APPROXIMATE in-memory lexical fallback (not score-compatible with the engine) "
                         "(set %s to a bundled fts.duckdb_extension, or %s=1 to allow a one-time download)",
                         type(e).__name__, str(e).splitlines()[0][:160] if str(e) else "", FTS_EXTENSION_ENV,
                         INSTALL_FTS_ENV)
            return False

    @property
    def fts_available(self) -> bool:
        """True when BM25 runs in DuckDB's `fts` index, False on the approximate in-memory fallback."""
        return self._fts

    def _insert(self, docs: list[dict]) -> int:
        rows = [(d["chunk_id"], d.get("filename", ""), d.get("text", ""),
                 float(d.get("ts") or 0.0)) for d in docs]
        with self._write_lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO docs (chunk_id, filename, text, ts) VALUES (?,?,?,?)", rows)
            self._n = self._db.execute("SELECT count(*) FROM docs").fetchone()[0]
            self._fallback = None
        return len(rows)

    def _reindex(self) -> None:
        # (re)build the BM25 index over the text column; overwrite so re-index is idempotent.
        with self._write_lock:
            self._fallback = None
            if self._fts:
                self._db.execute("PRAGMA create_fts_index('docs','chunk_id','text', overwrite=1)")

    def _fallback_search(self, query: str, k: int) -> list[Hit]:
        """Approximate lexical search in process, used when the `fts` extension is unavailable.

        It runs the in-memory store's BM25 over the same rows, which is NOT the engine's BM25: the engine drops
        English stop words and digit tokens and stems with Porter, the fallback does none of that, and the score
        scales differ. A query can therefore match more (``the``, ``404``) or fewer (``errors`` vs ``error``)
        chunks than on the engine path, and scores must not be compared across the two. The snapshot is built
        and installed under the write lock, so a concurrent insert (which resets it under the same lock) can
        never be overwritten by an older snapshot.
        """
        with self._write_lock:
            fb = self._fallback
            if fb is None:
                from .store_inmemory import InMemoryStore
                rows = self._db.cursor().execute(
                    "SELECT chunk_id, filename, text FROM docs ORDER BY chunk_id").fetchall()
                fb = InMemoryStore([{"chunk_id": r[0], "filename": r[1], "text": r[2], "created_at": None}
                                    for r in rows], source=self.source)
                self._fallback = fb
        return fb.search(query, k, method="bm25")

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
                             "text": fp.read_text(encoding="utf-8", errors="ignore"), "ts": fp.stat().st_mtime})
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
            if not self._fts:
                return self._fallback_search(query, k)
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
