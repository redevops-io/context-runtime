"""Arm D — freshness and REFRESH (plan §8).

Uses the real revision timestamps as evidence age. Each real revision is evaluated at TWO points in
time, so both freshness regimes come from real data: shortly after the revision (fresh → serve) and as
of the frozen 2026 dump date (years later → stale → REFRESH). With the policy disabled, both serve
unchanged. Retrieval is driven by a snippet of the revision's own text so the evidence is reliably
returned (freshness is computed from what was actually retrieved).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from harness.evidence_corpus import RevisionPair

# Fixed "now" for the stale regime: the frozen dump date. Deterministic, no clock.
DUMP_AS_OF = "2026-08-01T00:00:00Z"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _query(text: str) -> str:
    return " ".join(text.split()[:12])


def _doc(pair: RevisionPair, revid: str, text: str, ts: str) -> dict:
    from runtime_contracts.canonical import content_hash as rcv1
    ref = f"strategywiki/page/{pair.page_id}"
    return {"chunk_id": f"{ref}@{revid}::0", "filename": pair.title, "text": text,
            "observed_at": ts, "version": revid, "content_hash": rcv1(text), "source_ref": ref}


def run(pairs: list[RevisionPair]) -> dict:
    from context_runtime import ContextRuntime, FreshnessPolicy

    policy = FreshnessPolicy(enabled=True, mode="age_decay", half_life_days=30.0, min_freshness=0.5)

    refresh_when_stale = stale_not_refreshed = 0
    served_when_fresh = unnecessary_refresh = 0
    retrieval_miss = disabled_unchanged = 0
    n_stale = n_fresh = n_disabled = 0

    for pair in pairs:
        for revid, text, ts in ((pair.a_revid, pair.a_text, pair.a_ts),
                                (pair.b_revid, pair.b_text, pair.b_ts)):
            doc = _doc(pair, revid, text, ts)
            q = _query(text)
            rev_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            fresh_as_of = _iso(rev_dt + timedelta(days=1))   # one day after the revision → fresh

            # --- FRESH regime: evaluated just after the revision existed ---
            rf = ContextRuntime.default([doc], freshness_policy=policy).run(q, as_of=fresh_as_of)
            if rf.refresh:
                unnecessary_refresh += 1
            else:
                served_when_fresh += 1
            n_fresh += 1

            # --- STALE regime: evaluated as of the 2026 dump (years later) ---
            rs = ContextRuntime.default([doc], freshness_policy=policy).run(q, as_of=DUMP_AS_OF)
            if rs.freshness == 1.0:
                retrieval_miss += 1        # nothing retrieved → freshness defaulted; not a real stale test
            elif rs.refresh:
                refresh_when_stale += 1
            else:
                stale_not_refreshed += 1
            n_stale += 1

            # --- DISABLED: no policy ⇒ serve, freshness 1.0, regardless of age ---
            ro = ContextRuntime.default([doc]).run(q, as_of=DUMP_AS_OF)
            if not ro.refresh and ro.freshness == 1.0 and ro.answer:
                disabled_unchanged += 1
            n_disabled += 1

    passed = (n_stale > 0 and n_fresh > 0
              and retrieval_miss == 0
              and stale_not_refreshed == 0
              and unnecessary_refresh == 0
              and refresh_when_stale == n_stale
              and served_when_fresh == n_fresh
              and disabled_unchanged == n_disabled)
    return {
        "arm": "D", "name": "freshness and REFRESH", "passed": passed,
        "n_cases": n_stale + n_fresh,
        "metrics": {
            "stale_evaluations": n_stale,
            "fresh_evaluations": n_fresh,
            "refresh_when_stale": refresh_when_stale,       # == n_stale
            "served_when_fresh": served_when_fresh,         # == n_fresh
            "unnecessary_refresh": unnecessary_refresh,     # HARD = 0
            "stale_not_refreshed": stale_not_refreshed,     # HARD = 0
            "retrieval_miss": retrieval_miss,               # HARD = 0 (harness sanity)
            "legacy_disabled_unchanged": disabled_unchanged,  # == n_disabled
            "stale_detection_recall": round(refresh_when_stale / n_stale, 3) if n_stale else 1.0,
        },
    }
