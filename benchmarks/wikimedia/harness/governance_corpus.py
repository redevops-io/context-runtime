"""Real evidence↔action governance trajectories from strategywiki (for arms E/F, v0.3.0).

Each case is one page's timeline: the timestamps of its **reverts** (revisions whose sha1 restores an
earlier content state — an edit-war signal) and, if the page was ever protected, the **protection**
timestamp (the external moderation label). Protected pages are positives; multi-revert pages that were
never protected are the negative controls. Timestamps are epoch seconds so they drop straight into the
governance engine's integer-timestamp `RuntimeEvent` ledger. Pure parsing; deterministic ordering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from harness.corpus import iter_logs, iter_pages
from harness.evidence_corpus import CONTENT_NS, HISTORY, LOGGING


def _epoch(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=timezone.utc).timestamp())


@dataclass
class GovCase:
    page_id: int
    title: str
    revert_epochs: list[int] = field(default_factory=list)   # timestamps of revert revisions
    editor_by_revert: list[str] = field(default_factory=list)  # contributor per revert (for distinct_on)
    protected_at: int | None = None                          # epoch of first protection, or None (control)

    @property
    def is_protected(self) -> bool:
        return self.protected_at is not None


def _protection_epochs() -> dict[str, int]:
    prot: dict[str, int] = {}
    for item in iter_logs(LOGGING):
        if item.type == "protect" and item.action in ("protect", "modify") and item.timestamp:
            e = _epoch(item.timestamp)
            prot[item.logtitle] = min(prot.get(item.logtitle, e), e)   # first protection
    return prot


def select_governance_cases(n_protected: int = 30, n_controls: int = 60) -> list[GovCase]:
    """Deterministic slice: the first ``n_protected`` content pages with ≥2 reverts that were later
    protected, plus the first ``n_controls`` content pages with ≥2 reverts never protected."""
    prot = _protection_epochs()
    protected: list[GovCase] = []
    controls: list[GovCase] = []
    for page in iter_pages(HISTORY, include_text=False):
        if page.ns not in CONTENT_NS or len(page.revisions) < 2:
            continue
        seen: set[str] = set()
        rev_epochs: list[int] = []
        editors: list[str] = []
        for r in page.revisions:
            if r.sha1 and r.sha1 in seen and r.timestamp:
                rev_epochs.append(_epoch(r.timestamp))
                editors.append(r.contributor or "?")
            if r.sha1:
                seen.add(r.sha1)
        if len(rev_epochs) < 2:
            continue
        p_at = prot.get(page.title)
        case = GovCase(page_id=page.page_id, title=page.title, revert_epochs=rev_epochs,
                       editor_by_revert=editors, protected_at=p_at)
        if case.is_protected and len(protected) < n_protected:
            # keep only reverts that precede the protection (the storm that *led to* it)
            case.revert_epochs = [e for e in rev_epochs if e <= p_at] or rev_epochs
            protected.append(case)
        elif not case.is_protected and len(controls) < n_controls:
            controls.append(case)
        if len(protected) >= n_protected and len(controls) >= n_controls:
            break
    return protected + controls
