"""Evidence-sourced freshness — turn a retrieval Hit's source timestamp/version into a freshness
score, and surface the exact evidence lineage in EXPLAIN.

The scoring dimension (``PlanScore.freshness``), the staleness penalty, the REFRESH abstention
outcome, and the EXPLAIN lineage *rendering* all already exist. What was missing — and what this
module adds — is the **source**: a way to derive freshness from the actual retrieved/versioned
evidence rather than a static prior, plus a producer that names the exact evidence ref/version/hash
in EXPLAIN. It is entirely opt-in: with no :class:`FreshnessPolicy` (or ``enabled=False``) every
value is ``1.0`` and behavior is byte-for-byte the legacy path.

Domain-neutral: freshness is computed from a Hit's ``observed_at`` (source time) against an ``as_of``
reference, under a configurable policy. It does not assume "newer is always better" — a deployment
picks age-decay or a hard TTL.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from .types import Hit


@dataclass(frozen=True)
class FreshnessPolicy:
    """How to turn evidence age into a freshness score in [0,1] (1=fresh, 0=stale).

    ``enabled=False`` (the default) disables sourcing entirely → freshness is always 1.0. ``mode``:
    ``age_decay`` (exponential half-life) or ``ttl`` (fresh until ``ttl_seconds``, then 0).
    ``min_freshness`` is the REFRESH threshold the serving gate compares against.
    """

    enabled: bool = False
    mode: str = "age_decay"          # "age_decay" | "ttl"
    half_life_days: float = 90.0
    ttl_seconds: float = 0.0
    min_freshness: float = 0.5
    min_confidence: float = 0.0      # confidence bar for the same serving gate; 0 = freshness-only


def _parse_ts(ts) -> datetime | None:
    if ts is None or ts == "":
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        d = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def freshness_of(observed_at, *, as_of=None, policy: FreshnessPolicy) -> float:
    """Freshness in [0,1] of one piece of evidence observed at ``observed_at``, as of ``as_of``.

    Unknown ``observed_at`` → 1.0 (never penalize evidence whose age we can't determine — keeps
    legacy corpora that carry no source time unchanged). ``as_of=None`` uses the current time.
    """
    if not policy.enabled:
        return 1.0
    obs = _parse_ts(observed_at)
    if obs is None:
        return 1.0
    ref = _parse_ts(as_of) or datetime.now(timezone.utc)
    age_s = max((ref - obs).total_seconds(), 0.0)
    if policy.mode == "ttl":
        if policy.ttl_seconds <= 0:
            return 1.0
        return 1.0 if age_s <= policy.ttl_seconds else 0.0
    half_life_s = max(policy.half_life_days, 1e-9) * 86400.0
    return max(0.0, min(1.0, 0.5 ** (age_s / half_life_s)))


def score_hits(hits: Iterable[Hit], *, as_of=None, policy: FreshnessPolicy) -> float:
    """Aggregate freshness of a served evidence set — the **worst** (min) piece governs, so a single
    stale citation makes the answer stale. 1.0 when disabled or no evidence carries a timestamp."""
    if not policy.enabled:
        return 1.0
    vals = [freshness_of(h.observed_at, as_of=as_of, policy=policy) for h in hits]
    return min(vals) if vals else 1.0


def lineage_from_hits(hits: Iterable[Hit], *, as_of=None, policy: FreshnessPolicy | None = None,
                      capability_version: str = "") -> list[dict]:
    """Build the EXPLAIN ``lineage`` rows (the versioned-refs section) from served evidence.

    Names the exact source ref, version and content hash of each hit — the "populate, don't merely
    render" half of evidence-lineage EXPLAIN. Freshness is included when a policy is given.
    """
    pol = policy or FreshnessPolicy()
    rows: list[dict] = []
    for h in hits:
        ref = (h.meta or {}).get("source_ref") or h.chunk_id
        row: dict = {"ref": ref, "version": h.version, "content_hash": h.content_hash,
                     "source": h.source}
        if capability_version:
            row["capability_version"] = capability_version
        if pol.enabled:
            row["freshness"] = freshness_of(h.observed_at, as_of=as_of, policy=pol)
        rows.append(row)
    return rows


def explain_hit_row(h: Hit, *, served: bool = True) -> dict:
    """One EXPLAIN retrieval-trace row that carries the hit's revision + content hash so the
    per-hit ``⟨@version #hash⟩`` annotation renders."""
    return {
        "served": served, "chunk_id": h.chunk_id, "filename": h.filename,
        "score": float(h.score), "version": h.version, "content_hash": h.content_hash,
    }
