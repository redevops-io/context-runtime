"""Deterministic selection of real strategywiki pages/revisions for the benchmark arms.

Everything downstream must be reproducible, so selection is a pure, ordered scan of the frozen
corpus (no randomness, no clock): pages are taken in ascending page_id, revisions in dump order
(oldest→newest). Each arm asks for a small, fixed slice — a handful of multi-revision content pages —
so the small-corpus benchmark runs in seconds and reproduces identically across the ≥3 clean reruns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.corpus import Page, iter_pages, iter_logs

DATA = Path(__file__).resolve().parent.parent / "data"
HISTORY = DATA / "strategywiki-20260801-pages-meta-history.xml.bz2"
LOGGING = DATA / "strategywiki-20260801-pages-logging.xml.gz"

# Content namespaces worth pinning as evidence (articles + Proposal space); skip talk/thread noise.
CONTENT_NS = {0, 4, 106, 12}


@dataclass
class RevisionPair:
    """Two real consecutive revisions of one page — the A→B evidence advance most arms need."""
    page_id: int
    title: str
    a_revid: str
    a_text: str
    a_ts: str
    b_revid: str
    b_text: str
    b_ts: str


def select_multi_revision_pages(limit: int, *, min_revs: int = 3,
                                min_len: int = 200, max_len: int = 8000) -> list[Page]:
    """First ``limit`` content pages (ascending page_id) with ≥``min_revs`` revisions whose current
    text is a reasonable size — deterministic, so every run picks the same pages."""
    out: list[Page] = []
    for page in iter_pages(HISTORY, include_text=True):
        if page.ns not in CONTENT_NS or len(page.revisions) < min_revs:
            continue
        cur = page.revisions[-1].text or ""
        if not (min_len <= len(cur) <= max_len):
            continue
        out.append(page)
        if len(out) >= limit:
            break
    return out


def select_revision_pairs(limit: int, *, min_len: int = 200, max_len: int = 8000) -> list[RevisionPair]:
    """``limit`` (revision A, revision B) pairs from distinct content pages — A is the page's first
    substantive revision, B a strictly-later one with different content (a real edit)."""
    pairs: list[RevisionPair] = []
    for page in iter_pages(HISTORY, include_text=True):
        if page.ns not in CONTENT_NS or len(page.revisions) < 2:
            continue
        revs = [r for r in page.revisions if r.text and min_len <= len(r.text) <= max_len]
        if len(revs) < 2:
            continue
        a, b = revs[0], None
        for r in revs[1:]:
            if r.sha1 and a.sha1 and r.sha1 != a.sha1 and r.text != a.text:
                b = r
                break
        if b is None:
            continue
        pairs.append(RevisionPair(
            page_id=page.page_id, title=page.title,
            a_revid=str(a.rev_id), a_text=a.text, a_ts=a.timestamp,
            b_revid=str(b.rev_id), b_text=b.text, b_ts=b.timestamp))
        if len(pairs) >= limit:
            break
    return pairs


def select_protected_page_trajectories(limit: int) -> list[dict]:
    """Pages that received a real protection event, with their revision timestamps + revert markers.

    Kept for the (deferred) E/F governance arms — the evidence trajectory (revisions/reverts) that
    precedes a real moderation action. Returns [] work for A–D/G; wired here so the corpus selection
    for E/F is ready when the v0.3.0 governance engine exists.
    """
    protected: dict[str, str] = {}
    for item in iter_logs(LOGGING):
        if item.type == "protect" and item.action in ("protect", "modify"):
            protected.setdefault(item.logtitle, item.timestamp)
    out: list[dict] = []
    if not protected:
        return out
    for page in iter_pages(HISTORY, include_text=False):
        if page.title in protected and len(page.revisions) >= 3:
            seen: set[str] = set()
            reverts = 0
            for r in page.revisions:
                if r.sha1 and r.sha1 in seen:
                    reverts += 1
                if r.sha1:
                    seen.add(r.sha1)
            out.append({
                "page_id": page.page_id, "title": page.title,
                "protected_at": protected[page.title],
                "n_revisions": len(page.revisions), "n_reverts": reverts,
                "revision_timestamps": [r.timestamp for r in page.revisions],
            })
            if len(out) >= limit:
                break
    return out
