"""LocalFolderSource — the reference SourcePlugin: walk files/folders on disk.

This is the plugin form of what build_corpus() used to do inline. It yields one
RawAsset per file (uri = absolute path, label = path relative to the source root), so
downstream extraction/quality/indexing is identical whether the bytes came from a
folder or a dlt connector.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..types import PluginInfo, RawAsset


class LocalFolderSource:
    def __init__(self, paths: list[str] | str, *, follow_symlinks: bool = True,
                 limit: int | None = None):
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self.follow_symlinks = follow_symlinks
        self.limit = limit

    def read(self) -> Iterator[RawAsset]:
        n = 0
        for src in self.paths:
            base = Path(src)
            entries: list[tuple[Path, str]] = []
            if base.is_file():
                entries.append((base, base.name))
            else:
                for p in sorted(base.rglob("*")):
                    try:
                        if p.is_file() or (self.follow_symlinks and p.is_symlink()
                                           and p.resolve().is_file()):
                            entries.append((p, str(p.relative_to(base))))
                    except OSError:
                        continue
            for path, label in entries:
                if self.limit is not None and n >= self.limit:
                    return
                n += 1
                yield RawAsset(id=label, uri=str(path), label=label,
                               mime=None, meta={"source": "local_folder"})

    def info(self) -> PluginInfo:
        return PluginInfo(name="local_folder_source", kind="source", version="0.1",
                          capabilities=frozenset({"files", "recursive"}))
