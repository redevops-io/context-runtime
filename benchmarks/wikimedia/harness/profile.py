"""S2 — profile the frozen corpus (plan §12).

Counts pages, revisions, revisions/page, changed pages, reverts, and
protection/log events, plus the temporal span. Pure parsing; imports no runtime.

Revert detection uses MediaWiki's own ``sha1`` content digest: a revision is a
revert iff its content sha1 equals the sha1 of some strictly-earlier revision of the
same page (the edit restored a prior content state). This is deterministic and needs
no model — it is also the signal Test E/F build their evidence trajectories on.

Run:  PYTHONPATH=. python -m harness.profile
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from harness.corpus import iter_logs, iter_pages

DATA = Path(__file__).resolve().parent.parent / "data"
HISTORY = DATA / "strategywiki-20260801-pages-meta-history.xml.bz2"
LOGGING = DATA / "strategywiki-20260801-pages-logging.xml.gz"


def profile() -> dict:
    pages = 0
    revisions = 0
    reverts = 0
    pages_with_revert = 0
    revs_per_page_max = 0
    ns_counter: Counter = Counter()
    min_ts = "9999"
    max_ts = ""
    multi_edit_pages = 0  # pages with > 1 revision ("changed pages")

    for page in iter_pages(HISTORY, include_text=False):
        pages += 1
        ns_counter[page.ns] += 1
        n = len(page.revisions)
        revisions += n
        revs_per_page_max = max(revs_per_page_max, n)
        if n > 1:
            multi_edit_pages += 1
        seen_sha1: set[str] = set()
        page_had_revert = False
        for rev in page.revisions:
            if rev.timestamp:
                min_ts = min(min_ts, rev.timestamp)
                max_ts = max(max_ts, rev.timestamp)
            if rev.sha1 and rev.sha1 in seen_sha1:
                reverts += 1
                page_had_revert = True
            if rev.sha1:
                seen_sha1.add(rev.sha1)
        if page_had_revert:
            pages_with_revert += 1

    log_types: Counter = Counter()
    protect_events = 0
    protected_titles: set[str] = set()
    for item in iter_logs(LOGGING):
        log_types[item.type] += 1
        if item.type == "protect" and item.action in ("protect", "modify"):
            protect_events += 1
            protected_titles.add(item.logtitle)

    return {
        "pages": pages,
        "revisions": revisions,
        "revisions_per_page_mean": round(revisions / pages, 3) if pages else 0,
        "revisions_per_page_max": revs_per_page_max,
        "changed_pages_gt1_rev": multi_edit_pages,
        "reverts_sha1_match": reverts,
        "pages_with_revert": pages_with_revert,
        "namespaces_top": dict(ns_counter.most_common(8)),
        "temporal_span": {"first_revision": min_ts, "last_revision": max_ts},
        "log_events_total": sum(log_types.values()),
        "log_types_top": dict(log_types.most_common(10)),
        "protection_events": protect_events,
        "distinct_protected_pages": len(protected_titles),
    }


def main() -> int:
    if not HISTORY.exists():
        print(f"missing corpus: {HISTORY}", file=sys.stderr)
        return 2
    prof = profile()
    print(json.dumps(prof, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
