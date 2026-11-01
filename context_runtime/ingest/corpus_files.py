"""Deterministic corpus enumeration: the same folder yields the same files, labels and order on every OS.

``sorted(Path.rglob("*"))`` is not portable: ``WindowsPath`` orders case-insensitively and ``PosixPath``
case-sensitively, labels built with ``str(path.relative_to(root))`` carry ``\\`` on Windows, and the same
file name can arrive as NFC (Windows, most Linux tools) or NFD (older macOS volumes, some archives).
Chunk ids, labels and ranking ties then differ from one operating system to the next.

:func:`iter_corpus_files` fixes the contract once:

* **label** (:attr:`CorpusFile.rel`): the path relative to the root, ``/``-separated, Unicode NFC;
* **order**: component by component, by Unicode code point of the NFC components (so ``B`` < ``a``,
  and ``a/x`` sorts before ``a.txt`` because ``a`` < ``a.txt``), ties broken by the raw on-disk name;
* **what is listed**: regular files only. Directories that are symlinks, Windows junctions or other
  directory reparse points (mount points) are not entered, because a junction is not a symlink and
  escapes every ``is_symlink()`` confinement check; each one skipped is logged.

The file itself is always opened through :attr:`CorpusFile.path`, the name as stored on disk.
"""

from __future__ import annotations

import logging
import os
import stat
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

_log = logging.getLogger("context_runtime.corpus")


class CorpusFile(NamedTuple):
    rel: str       # NFC, POSIX-style path relative to the root (the label)
    path: Path     # the file on disk (use this to open it)

    @property
    def name(self) -> str:
        """The NFC file name (the last component of :attr:`rel`)."""
        return self.rel.rsplit("/", 1)[-1]


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def is_junction(path: str | os.PathLike[str]) -> bool:
    """True for a Windows junction (always False elsewhere). Kept as a function so tests can substitute it."""
    return os.path.isjunction(path)


def is_reparse_point(path: str | os.PathLike[str]) -> bool:
    """True when the entry carries the Windows reparse-point attribute (junction, symlink, mount point,
    cloud placeholder ...). Always False where the platform has no such attribute."""
    try:
        attrs = getattr(os.lstat(path), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def is_redirecting_directory(path: str | os.PathLike[str]) -> bool:
    """A directory that points somewhere else: a symlink, a junction, or a directory reparse point."""
    return os.path.islink(path) or is_junction(path) or (os.path.isdir(path) and is_reparse_point(path))


def _walk(root: Path) -> Iterator[tuple[tuple[str, ...], str, Path]]:
    stack = [(root, ())]
    while stack:
        directory, parts = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as e:
            _log.warning("corpus: cannot list %s (%s)", directory, e)
            continue
        for entry in entries:
            path = Path(entry.path)
            names = (*parts, nfc(entry.name))
            try:
                if entry.is_dir(follow_symlinks=False):
                    if is_junction(entry.path) or is_reparse_point(entry.path):
                        _log.warning("corpus: not entering %s (junction or reparse point)", path)
                        continue
                    stack.append((path, names))
                elif entry.is_symlink() and entry.is_dir():
                    continue                     # a symlink to a directory is never entered
                elif entry.is_file():
                    yield names, entry.name, path
            except OSError:
                continue


def _warn_duplicate_names(base: Path, chosen: list[tuple[tuple[str, ...], str, Path]]) -> None:
    """Log one warning when several listed files share a file name (after NFC). The stores build chunk ids from
    the file name alone (``<name>::0``), so those files get the same id and replace or shadow one another. The id
    format is kept (changing it would invalidate existing indexes); this makes the collision visible."""
    seen: dict[str, str] = {}
    dups: list[str] = []
    for names, _raw, _path in chosen:
        if names[-1] in seen:
            dups.append(f"{seen[names[-1]]} and {'/'.join(names)}")
        else:
            seen[names[-1]] = "/".join(names)
    if dups:
        _log.warning("corpus: %d file(s) under %s share a file name with another file and will get the same chunk id "
                     "(for example %s); rename them to keep both", len(dups), base, "; ".join(dups[:3]))


def iter_corpus_files(root: str | os.PathLike[str], suffixes: Iterable[str] | None = None, *,
                      warn_duplicates: bool = True) -> Iterator[CorpusFile]:
    """The regular files under ``root`` in the portable order described in the module docstring.

    ``suffixes`` (for example ``(".md", ".txt")``) keeps only files whose lower-cased suffix is listed; the
    comparison uses :attr:`pathlib.Path.suffix`, as the callers always did. ``warn_duplicates=False`` skips the
    duplicate-name warning for callers that label by the full relative path. ``root`` may also be a single
    file, which is yielded alone. A root that does not exist yields nothing.
    """
    base = Path(root).expanduser()
    wanted = None if suffixes is None else {s.lower() for s in suffixes}
    found: list[tuple[tuple[str, ...], str, Path]] = []
    if base.is_file():
        found.append(((nfc(base.name),), base.name, base))
    elif base.is_dir():
        found.extend(_walk(base))
    found.sort(key=lambda t: (t[0], t[1]))
    chosen = [t for t in found if wanted is None or Path(t[0][-1]).suffix.lower() in wanted]
    if warn_duplicates:
        _warn_duplicate_names(base, chosen)
    for names, _raw, path in chosen:
        yield CorpusFile("/".join(names), path)
