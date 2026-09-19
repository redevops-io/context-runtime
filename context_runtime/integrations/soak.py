"""Longitudinal collection soak — one thin, replayable collection pass + a durable per-run record.

The scheduler around this is deliberately trivial; the value is the accumulated evidence. One scheduled
execution is exactly the plan's linear flow:

    run_id → INFORMS list (+ detail enrichment) + Future Solicitations → EvidenceStore
           → NEW/UPDATED/UNCHANGED → qualification → handoff when warranted (→ outbox for the Mission side)

Crucially a `CollectionRun` record is written on EVERY execution, even a zero-change day, capturing per
source: attempted, ok, HTTP status, record count, detail fetches, error. Without it a quiet procurement
day is indistinguishable later from a collector that silently stopped working — so the 30-day report can
separate **source reliability** (did we reach the source?) from **procurement activity** (did anything
actually change?). This module stops at the handoff/outbox boundary; it never imports the Mission side.

Read-only: this consumes publicly-exposed opportunity information only — it never submits/registers/bids.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .evidence_store import ChangeKind, EvidenceStore
from .local_gov import (
    BusinessProfile, JurisdictionRegistry, default_registry, qualification_version, qualify, to_handoff,
)
from .procurement_sources import (
    BROWSER_METHODS, COLLECTOR_VERSION, Fetcher, Observation, ProcurementSource, SourceMethod, _now_iso,
    _observe, default_source_registry, enrich_informs, merge_detail, normalize, parse_source,
    resolve_fetch_url,
)


@dataclass
class SourceOutcome:
    """What happened for one source on one run — the source-reliability signal."""
    source_id: str
    method: str
    ok: bool                 # did we reach it and get a usable body?
    http_status: int
    record_count: int        # rows parsed from the list/feed
    detail_fetches: int = 0  # detail pages successfully enriched (INFORMS)
    error: str = ""          # failure reason when not ok (transport error / blocked / non-200)
    fetch_ms: int = 0        # request latency: the list/feed fetch alone
    collect_ms: int = 0      # collection latency: fetch + parse + detail enrichment for this source


@dataclass
class CollectionRun:
    """A durable record of one scheduled collection pass — written even when nothing changed."""
    run_id: str
    started_at: str
    ended_at: str
    collector_version: str
    zip_code: str
    radius_miles: float
    sources: list[SourceOutcome]
    scheduled_at: str = ""            # when the tick was SUPPOSED to run (scheduler-health signal)
    qualification_version: str = ""   # rules+profile version the funnel below was produced under
    observation_count: int = 0
    new_count: int = 0
    updated_count: int = 0
    unchanged_count: int = 0
    pursue_count: int = 0
    review_count: int = 0
    reject_count: int = 0
    handoff_count: int = 0

    @property
    def sources_ok(self) -> int:
        return sum(1 for s in self.sources if s.ok)

    @property
    def had_change(self) -> bool:
        return (self.new_count + self.updated_count) > 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CollectionResult:
    run: CollectionRun
    handoffs: list[dict]
    changes: list
    observations: list[Observation]


class CollectionRunStore:
    """Append-only JSONL of CollectionRun records — the soak's spine, one line per scheduled pass."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, run: CollectionRun) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(run.to_dict(), separators=(",", ":")) + "\n")

    def runs(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_text().splitlines() if ln.strip()]


def _append_outbox(outbox: Path, handoffs: list[dict]) -> None:
    """Durable handoff queue the Mission side drains (idempotently). Keeps the two repos decoupled: this
    side writes JSON, the consumer reads JSON — neither imports the other."""
    outbox.parent.mkdir(parents=True, exist_ok=True)
    with outbox.open("a") as fh:
        for h in handoffs:
            fh.write(json.dumps(h, separators=(",", ":")) + "\n")


def run_collection(*, zip_code: str, radius_miles: float, profile: BusinessProfile, fetcher: Fetcher,
                   run_id: Optional[str] = None, now: Optional[str] = None,
                   registry: Optional[JurisdictionRegistry] = None,
                   sources: Optional[list[ProcurementSource]] = None,
                   evidence_store: Optional[EvidenceStore] = None,
                   run_store: Optional[CollectionRunStore] = None,
                   outbox: Optional[str | Path] = None, scheduled_at: Optional[str] = None,
                   enrich: bool = True, default_categories: tuple[str, ...] = (),
                   browser_fetcher: Optional[Fetcher] = None) -> CollectionResult:
    """Run one collection pass and record a CollectionRun (always, even on a zero-change day).

    Persists observations + EvidenceChanges to ``evidence_store`` when given, the CollectionRun to
    ``run_store``, and warranted handoffs to ``outbox``. Returns everything for the caller (the thin
    scheduler) to inspect; it does NOT open missions — that is the Mission side's job, reached via the
    decoupled outbox. ``scheduled_at`` (when the tick was meant to run) is recorded alongside the actual
    ``started_at`` so scheduler lag/missed-runs are measurable separately from source reliability."""
    reg = registry or default_registry()
    started = now or _now_iso()
    run_id = run_id or f"run-{started.replace(':', '').replace('-', '')}"
    qual_version = qualification_version(profile)

    monitored = reg.within(zip_code, radius_miles)
    monitored_ids = {m.jurisdiction.id for m in monitored}
    all_sources = sources or list(default_source_registry().values())
    region_sources = [s for s in all_sources if s.jurisdiction_id in monitored_ids or s.jurisdiction_id == "*"]

    outcomes: list[SourceOutcome] = []
    observations: list[Observation] = []
    handoffs: list[dict] = []
    changes: list = []
    counts = {"NEW": 0, "UPDATED": 0, "UNCHANGED": 0, "PURSUE": 0, "REVIEW": 0, "REJECT": 0}

    for src in region_sources:
        # SAM.gov needs a keyed URL built from env SAM_API_KEY; a missing key is a recorded source
        # failure (not a silent skip), and the key never enters an observation (obs uses src.url).
        fetch_url, url_err = resolve_fetch_url(src, profile=profile, now=started)
        if fetch_url is None:
            outcomes.append(SourceOutcome(
                source_id=src.source_id, method=src.method.value, ok=False, http_status=0,
                record_count=0, error=url_err))
            continue
        # portals that block plain HTTP / render via JS use the browser fetcher when one is provided
        use_fetcher = browser_fetcher if (browser_fetcher and src.method in BROWSER_METHODS) else fetcher
        t0 = time.monotonic()
        res = use_fetcher.fetch(fetch_url)
        fetch_ms = int((time.monotonic() - t0) * 1000)
        ok = res.status == 200 and bool(res.text)
        records = parse_source(src.method, res.text) if ok else []
        detail_fetches = 0
        for rec in records:
            obs = _observe(src.source_id, src.url, started, res.status, rec, kind="list")
            observations.append(obs)
            if evidence_store is not None:
                evidence_store.put_observation(obs, now=started)
            opp = normalize(obs, src, categories=default_categories, zip_code=zip_code)

            if enrich and src.method is SourceMethod.INFORMS:
                detail = enrich_informs(obs, src, fetcher, now=started)
                if detail is not None:
                    detail_fetches += 1
                    observations.append(detail)
                    if evidence_store is not None:
                        evidence_store.put_observation(detail, now=started)
                    merge_detail(opp, detail)

            if evidence_store is not None:
                ch = evidence_store.ingest(opp, now=started)
                changes.append(ch)
                counts[ch.kind.value.upper()] += 1

            q = qualify(opp, profile, reg)
            counts[q.decision.value] += 1
            if q.decision.value in ("PURSUE", "REVIEW"):
                h = to_handoff(opp, q, monitored)
                h["run_id"] = run_id
                h["qualification_version"] = qual_version
                handoffs.append(h)

        outcomes.append(SourceOutcome(
            source_id=src.source_id, method=src.method.value, ok=ok, http_status=res.status,
            record_count=len(records), detail_fetches=detail_fetches,
            error="" if ok else (res.content_type or f"http {res.status}"),
            fetch_ms=fetch_ms, collect_ms=int((time.monotonic() - t0) * 1000)))

    if outbox is not None and handoffs:
        _append_outbox(Path(outbox), handoffs)

    run = CollectionRun(
        run_id=run_id, started_at=started, ended_at=_now_iso(), collector_version=COLLECTOR_VERSION,
        zip_code=zip_code, radius_miles=radius_miles, sources=outcomes,
        scheduled_at=scheduled_at or started, qualification_version=qual_version,
        observation_count=len(observations), new_count=counts["NEW"], updated_count=counts["UPDATED"],
        unchanged_count=counts["UNCHANGED"], pursue_count=counts["PURSUE"],
        review_count=counts["REVIEW"], reject_count=counts["REJECT"], handoff_count=len(handoffs))
    if run_store is not None:
        run_store.append(run)
    return CollectionResult(run, handoffs, changes, observations)


def _parse_iso(ts: str) -> Optional[float]:
    # self-consistent seconds for computing DELTAS (both operands parsed the same way); not a true epoch.
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):   # full timestamp, or a source's date-only posted field
        try:
            return time.mktime(time.strptime(ts, fmt))
        except (ValueError, TypeError):
            continue
    return None


def _epoch_utc(ts: str) -> Optional[float]:
    # a TRUE UTC epoch (needed when comparing against timezone-aware expected slots).
    from datetime import datetime, timezone
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc).timestamp()
        except (ValueError, TypeError):
            continue
    return None


def schedule_adherence(runs: list[dict], *, expected_slots_local=("08:15", "20:15"),
                       tz: str = "America/New_York", tolerance_minutes: int = 120,
                       now: Optional[str] = None) -> dict:
    """Compare recorded runs against the EXPECTED schedule slots — the direct "did the run happen?" signal.

    For each expected slot from the first recorded run through ``now``, a run whose started_at falls within
    ±``tolerance_minutes`` counts as a hit; a slot with no run near it is missing. Anchored strictly AFTER
    the first run so a manual day-1 seed (which won't sit on a cron slot) never makes the first slots look
    missed, and DST-correct via the local timezone. Returns expected/observed/missing counts, an adherence
    ratio, and the exact missing slot timestamps (UTC)."""
    from datetime import datetime, timedelta, timezone
    try:
        from zoneinfo import ZoneInfo
        zone = ZoneInfo(tz)
    except Exception:  # noqa: BLE001 — no tz database → interpret the slots as UTC
        zone = timezone.utc
        tz = "UTC"

    run_epochs = sorted(e for e in (_epoch_utc(r.get("started_at", "")) for r in runs) if e is not None)
    if not run_epochs:
        return {"expected": 0, "observed": 0, "missing": 0, "adherence": None, "missing_slots": []}
    first = run_epochs[0]
    now_epoch = _epoch_utc(now) if now else datetime.now(timezone.utc).timestamp()
    tol = tolerance_minutes * 60

    expected: list[float] = []
    day = datetime.fromtimestamp(first, zone).date()
    end = datetime.fromtimestamp(now_epoch, zone).date()
    while day <= end:
        for hhmm in expected_slots_local:
            hh, mm = (int(x) for x in hhmm.split(":"))
            slot = datetime(day.year, day.month, day.day, hh, mm, tzinfo=zone).timestamp()
            if first < slot <= now_epoch:            # only slots after the seed and already due
                expected.append(slot)
        day += timedelta(days=1)

    missing = [datetime.fromtimestamp(s, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
               for s in expected if not any(abs(r - s) <= tol for r in run_epochs)]
    n = len(expected)
    return {
        "expected": n, "observed": n - len(missing), "missing": len(missing),
        "adherence": round((n - len(missing)) / n, 3) if n else None,
        "tolerance_minutes": tolerance_minutes, "expected_slots_local": list(expected_slots_local),
        "tz": tz, "missing_slots": missing,
    }


def soak_report(runs: list[dict], *, expected_slots_local=None, tz: str = "America/New_York",
                tolerance_minutes: int = 120, now: Optional[str] = None) -> dict:
    """Summarize N CollectionRun records into a report giving THREE distinct reliability measurements
    plus the qualification funnel — the separations that make the numbers defensible.

      * scheduler reliability — did the collection run happen? (runs seen + scheduled-vs-actual lag, and,
        when ``expected_slots_local`` is given, an explicit expected-vs-observed slot check with the exact
        missing slots)
      * source reliability — could we observe each source, over the runs SINCE it was onboarded? (so a
        source added on day 12 is scored on its ~18 days of coverage, never a phantom 30/30)
      * procurement activity — did anything actually change? (NEW/UPDATED + runs-with-change)
      * funnel — PURSUE/REVIEW/REJECT under each qualification_version (day-1 vs day-30 comparable only
        within a version).
    """
    if not runs:
        return {"runs": 0}
    per_source: dict[str, dict] = {}
    for r in runs:
        for s in r.get("sources", []):
            agg = per_source.setdefault(s["source_id"], {
                "onboarded_at": r.get("started_at"),       # first run this source was attempted in
                "attempts": 0, "ok": 0, "records_total": 0, "detail_fetches_total": 0,
                "fetch_ms_total": 0, "collect_ms_total": 0, "last_status": None, "last_error": ""})
            agg["attempts"] += 1
            agg["ok"] += 1 if s["ok"] else 0
            agg["records_total"] += s.get("record_count", 0)
            agg["detail_fetches_total"] += s.get("detail_fetches", 0)
            agg["fetch_ms_total"] += s.get("fetch_ms", 0)
            agg["collect_ms_total"] += s.get("collect_ms", 0)
            agg["last_status"] = s.get("http_status")
            if not s["ok"]:
                agg["last_error"] = s.get("error", "")
    for sid, agg in per_source.items():
        n = agg["attempts"]
        # uptime is over runs SINCE onboarding (= attempts), not the whole soak window
        agg["scheduled_runs_since_onboarding"] = n
        agg["uptime"] = round(agg["ok"] / n, 3) if n else 0.0
        agg["avg_fetch_ms"] = round(agg.pop("fetch_ms_total") / n) if n else 0
        agg["avg_collect_ms"] = round(agg.pop("collect_ms_total") / n) if n else 0

    lags = [(_parse_iso(r.get("started_at", "")) or 0) - (_parse_iso(r.get("scheduled_at", "")) or 0)
            for r in runs if r.get("scheduled_at") and r.get("started_at")]
    runs_with_change = sum(1 for r in runs if (r.get("new_count", 0) + r.get("updated_count", 0)) > 0)
    scheduler = {                                          # did the run happen, and on time?
        "runs_recorded": len(runs),
        "max_lag_seconds": round(max(lags)) if lags else 0,
        "avg_lag_seconds": round(sum(lags) / len(lags)) if lags else 0,
    }
    if expected_slots_local:                               # explicit expected-vs-observed slot check
        scheduler["adherence"] = schedule_adherence(
            runs, expected_slots_local=expected_slots_local, tz=tz,
            tolerance_minutes=tolerance_minutes, now=now)
    return {
        "runs": len(runs),
        "window": {"first": runs[0].get("started_at"), "last": runs[-1].get("ended_at")},
        "collector_versions": sorted({r.get("collector_version", "") for r in runs}),
        "qualification_versions": sorted({r.get("qualification_version", "") for r in runs if r.get("qualification_version")}),
        "scheduler": scheduler,
        "reliability": per_source,                         # could we reach each source (since onboarding)?
        "activity": {                                      # did anything actually change?
            "new_total": sum(r.get("new_count", 0) for r in runs),
            "updated_total": sum(r.get("updated_count", 0) for r in runs),
            "unchanged_total": sum(r.get("unchanged_count", 0) for r in runs),
            "handoffs_total": sum(r.get("handoff_count", 0) for r in runs),
            "runs_with_change": runs_with_change,
            "zero_change_runs": len(runs) - runs_with_change,
        },
        "funnel": {                                        # is qualification too permissive? (per version)
            "pursue_total": sum(r.get("pursue_count", 0) for r in runs),
            "review_total": sum(r.get("review_count", 0) for r in runs),
            "reject_total": sum(r.get("reject_count", 0) for r in runs),
        },
    }


def _days_between(later_iso: str, earlier_iso: str) -> Optional[float]:
    a, b = _parse_iso(later_iso), _parse_iso(earlier_iso)
    return round((a - b) / 86400.0, 2) if (a is not None and b is not None) else None


def discovery_report(opportunities: list[dict]) -> dict:
    """Discovery performance from the durable opportunity records: how quickly we detected opportunities
    relative to when the source posted them (posting → discovery latency), and detail-enrichment coverage.

    Pass ``EvidenceStore(path).opportunities()``. Latency is only computed for opportunities whose source
    stated a posted date (source_posted_at) — we never infer one."""
    total = len(opportunities)
    lat_days = []
    for o in opportunities:
        d = _days_between(o.get("first_seen_at", ""), o.get("source_posted_at", ""))
        if d is not None and d >= 0:
            lat_days.append(d)
    enriched = sum(1 for o in opportunities if o.get("detail_observed"))
    lat_days.sort()
    return {
        "opportunities": total,
        "detail_observed": enriched,
        "detail_coverage": round(enriched / total, 3) if total else 0.0,
        "posting_to_discovery_days": {
            "measured": len(lat_days),                     # opps where the source stated a posted date
            "min": lat_days[0] if lat_days else None,
            "median": lat_days[len(lat_days) // 2] if lat_days else None,
            "max": lat_days[-1] if lat_days else None,
        },
    }
