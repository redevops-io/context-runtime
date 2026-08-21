"""Stream-parse the frozen MediaWiki XML dumps into typed records.

The dumps (MediaWiki export schema 0.11) are read incrementally with
``xml.etree.ElementTree.iterparse`` over a transparently-decompressed stream (``.bz2``
or ``.gz``), so the full revision history never has to sit in memory at once. Everything
here is pure parsing — no runtime-under-test is imported, so profiling stays cheap and
dependency-free.

Stable identity rule (plan §4): a page is addressed by its numeric ``page_id`` (titles
move; ids do not). A revision is addressed by its ``rev_id``; ``parent_id`` gives the
edit chain; ``sha1`` is MediaWiki's own content digest and lets us detect reverts
(a revision whose sha1 re-appears an earlier content state).
"""
from __future__ import annotations

import bz2
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
from xml.etree.ElementTree import iterparse

# MediaWiki export 0.11 namespace — every tag is namespaced, so we match on local name.
_MW = "{http://www.mediawiki.org/xml/export-0.11/}"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _open(path: str | Path):
    p = str(path)
    if p.endswith(".bz2"):
        return bz2.open(p, "rt", encoding="utf-8")
    if p.endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "rt", encoding="utf-8")


@dataclass(frozen=True)
class Revision:
    rev_id: int
    parent_id: Optional[int]
    timestamp: str          # ISO-8601, e.g. 2009-07-23T15:53:26Z
    contributor: str        # username, or "ip:1.2.3.4", or "" when suppressed
    comment: str
    sha1: str               # MediaWiki base-36 content digest ("" if absent)
    text: str               # revision wikitext ("" if omitted/suppressed)
    minor: bool


@dataclass
class Page:
    page_id: int
    title: str
    ns: int
    revisions: list[Revision] = field(default_factory=list)


@dataclass(frozen=True)
class LogItem:
    log_id: int
    timestamp: str
    contributor: str
    type: str               # protect | delete | move | block | ...
    action: str             # protect | unprotect | modify | ...
    logtitle: str           # the affected page title
    params: str


def _text(elem, name: str, default: str = "") -> str:
    child = elem.find(_MW + name)
    if child is None or child.text is None:
        return default
    return child.text


def iter_pages(history_path: str | Path, *, include_text: bool = True) -> Iterator[Page]:
    """Yield one Page (with all its revisions, oldest-first) per <page> element.

    Set ``include_text=False`` to skip carrying revision wikitext — much cheaper when
    only the revision graph/metadata is needed (e.g. profiling, revert detection).
    """
    with _open(history_path) as stream:
        page: Optional[Page] = None
        # State for the revision currently being assembled.
        for event, elem in iterparse(stream, events=("start", "end")):
            name = _local(elem.tag)
            if event == "start":
                if name == "page":
                    page = Page(page_id=-1, title="", ns=0)
                continue
            # end events
            if name == "title" and page is not None and page.title == "":
                page.title = elem.text or ""
            elif name == "ns" and page is not None:
                page.ns = int(elem.text or 0)
            elif name == "id" and page is not None and page.page_id == -1:
                # The first <id> after <page> (before any <revision>) is the page id.
                page.page_id = int(elem.text or -1)
            elif name == "revision" and page is not None:
                page.revisions.append(_parse_revision(elem, include_text=include_text))
                elem.clear()
            elif name == "page" and page is not None:
                yield page
                page = None
                elem.clear()


def _parse_revision(rev_elem, *, include_text: bool) -> Revision:
    contrib = ""
    c = rev_elem.find(_MW + "contributor")
    if c is not None:
        uname = c.find(_MW + "username")
        ip = c.find(_MW + "ip")
        if uname is not None and uname.text:
            contrib = uname.text
        elif ip is not None and ip.text:
            contrib = "ip:" + ip.text
    parent = _text(rev_elem, "parentid", "")
    return Revision(
        rev_id=int(_text(rev_elem, "id", "-1")),
        parent_id=int(parent) if parent else None,
        timestamp=_text(rev_elem, "timestamp"),
        contributor=contrib,
        comment=_text(rev_elem, "comment"),
        sha1=_text(rev_elem, "sha1"),
        text=_text(rev_elem, "text") if include_text else "",
        minor=rev_elem.find(_MW + "minor") is not None,
    )


def iter_logs(logging_path: str | Path) -> Iterator[LogItem]:
    """Yield one LogItem per <logitem> — protection/deletion/move/block events."""
    with _open(logging_path) as stream:
        for event, elem in iterparse(stream, events=("end",)):
            if _local(elem.tag) != "logitem":
                continue
            contrib = ""
            c = elem.find(_MW + "contributor")
            if c is not None:
                uname = c.find(_MW + "username")
                if uname is not None and uname.text:
                    contrib = uname.text
            yield LogItem(
                log_id=int(_text(elem, "id", "-1")),
                timestamp=_text(elem, "timestamp"),
                contributor=contrib,
                type=_text(elem, "type"),
                action=_text(elem, "action"),
                logtitle=_text(elem, "logtitle"),
                params=_text(elem, "params"),
            )
            elem.clear()
