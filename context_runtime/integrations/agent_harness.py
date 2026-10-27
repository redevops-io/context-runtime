"""agent-harness × Context Runtime: the context manager for ``agent_harness.loop.run_loop``.

    from context_runtime.integrations.agent_harness import ContextRuntimeManager
    mgr = ContextRuntimeManager(archive_dir="runs/x/archive")      # BM25 offline
    mgr = ContextRuntimeManager(archive_dir=..., use_redevops_rag=True)   # hybrid vector+BM25, needs the [rag] extra
    mgr.index_files(Path("task_folder"))                           # make input files searchable too
    run_loop(client, registry, system, task, context_manager=mgr)

Dependency direction: this package imports ``agent_harness`` (tool contract); ``agent_harness`` never imports this.
"""
from __future__ import annotations

from pathlib import Path

from ..compression.transcript import Archive, TranscriptCompactor


class ContextRuntimeManager(TranscriptCompactor):
    def __init__(self, archive_dir: str | Path | None = None, *, use_redevops_rag: bool = False,
                 rag_db: str | None = None, **kw):
        retriever = None
        if use_redevops_rag:
            from ..adapters.store_redevops import RedevopsRagRetriever
            directory = Path(archive_dir) if archive_dir else None
            retriever = RedevopsRagRetriever(db_path=rag_db or str((directory or Path.cwd()) / "archive.duckdb"),
                                             source="archive")
        super().__init__(Archive(archive_dir, retriever), **kw)

    def index_files(self, root: str | Path, suffixes: tuple[str, ...] = (".txt", ".md", ".csv", ".json", ".rst")) -> int:
        """Register input files with the same retriever so ``search_context`` also finds their content."""
        n = 0
        for fp in sorted(Path(root).rglob("*")):
            if fp.is_file() and fp.suffix.lower() in suffixes and fp.stat().st_size < 5_000_000:
                self.archive.add(fp.read_text(errors="ignore"), kind="input_file", tool=fp.name, turn=0)
                n += 1
        return n
