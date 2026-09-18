"""Durable evidence store — the append-only lineage that turns discovery from an experiment into a
longitudinal dataset.

The plan's durable chain, persisted so it survives restarts:

    immutable source artifact → content hash → Observation → EvidenceChange → (qualification → handoff →
    Mission events → owner action → outcome, on the consumer side)

Every record is one JSONL line, append-only; nothing is ever mutated in place. Two record types live here:

  * ``observation`` — an immutable, content-addressed fetch: source id + url, collector version, the raw
    record, its raw-content digest, ``observed_at`` (when the source was read) and ``known_at`` (when it
    entered our knowledge). Re-writing the same observation is a no-op (same id) — observations are facts.
  * ``opportunity`` / ``evidence_change`` — the normalized-opportunity digest and its lineage. Ingesting
    the same tender again with no material change is UNCHANGED (idempotent); a moved deadline / new status
    is UPDATED with the exact fields that changed — the signal that can reawaken an existing mission.

On construction the store replays its log to rebuild the latest digest per opportunity and the set of
known observation ids, so a fresh process pointed at the same file continues the same dataset — restart
and replay are deterministic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from .local_gov import RevenueOpportunity, correlation_key, opportunity_digest, opportunity_material


class ChangeKind(str, Enum):
    NEW = "new"              # first time we have seen this opportunity id
    UPDATED = "updated"      # material change vs the last known version (deadline, status, docs, …)
    UNCHANGED = "unchanged"  # rediscovered, no material change → idempotent no-op


@dataclass
class EvidenceChange:
    opportunity_id: str
    kind: ChangeKind
    prior_digest: str
    new_digest: str
    changed_fields: list[str] = field(default_factory=list)
    at: str = ""


def _now_iso() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class EvidenceStore:
    """Append-only JSONL evidence store. Immutable observations + versioned opportunity digests, reloaded
    on construction so lineage survives restarts."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._obs_ids: set[str] = set()
        self._opp_digest: dict[str, str] = {}
        self._opp_material: dict[str, dict] = {}
        self._load()

    # ── replay the durable log to rebuild in-memory state ──
    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if rec.get("rec") == "observation":
                self._obs_ids.add(rec["observation_id"])
            elif rec.get("rec") == "opportunity":
                self._opp_digest[rec["opportunity_id"]] = rec["normalized_digest"]
                self._opp_material[rec["opportunity_id"]] = rec.get("material", {})

    def _append(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")

    # ── observations: immutable, content-addressed, written once ──
    def put_observation(self, obs, *, now: Optional[str] = None) -> bool:
        """Persist an Observation if unseen. Returns True if it was newly written, False if already known
        (observations are immutable facts — the same content is never duplicated)."""
        oid = obs.observation_id
        if oid in self._obs_ids:
            return False
        self._obs_ids.add(oid)
        self._append({
            "rec": "observation", "observation_id": oid,
            "source_id": obs.source_id, "source_url": obs.url, "kind": obs.kind,
            "collector_version": obs.collector_version, "raw_digest": obs.content_hash,
            "observed_at": obs.fetched_at, "known_at": obs.known_at or (now or _now_iso()),
            "status": obs.status, "raw": obs.raw,
        })
        return True

    # ── opportunities: versioned by material digest, with per-field change detection ──
    def ingest(self, opp: RevenueOpportunity, *, now: Optional[str] = None) -> EvidenceChange:
        """Record a normalized opportunity and classify it against the last known version.

        NEW/UPDATED append a new opportunity version + an evidence_change record; UNCHANGED writes nothing
        (idempotent) but still returns the classification. UPDATED carries the exact material fields that
        changed — the reawaken signal for a downstream mission."""
        oid = opp.opportunity_id
        digest = opportunity_digest(opp)
        material = opportunity_material(opp)
        prior = self._opp_digest.get(oid)
        ts = now or _now_iso()

        if prior is None:
            kind, changed = ChangeKind.NEW, sorted(k for k, v in material.items() if v)
        elif prior == digest:
            return EvidenceChange(oid, ChangeKind.UNCHANGED, prior, digest, [], ts)
        else:
            before = self._opp_material.get(oid, {})
            changed = sorted(k for k in material if material.get(k) != before.get(k))
            kind = ChangeKind.UPDATED

        self._opp_digest[oid] = digest
        self._opp_material[oid] = material
        self._append({
            "rec": "opportunity", "opportunity_id": oid, "normalized_digest": digest,
            "opportunity_kind": "gov_forecast" if opp.solicitation_type.value == "FORECAST" else "gov_solicitation",
            "correlation_key": correlation_key(opp), "response_due_at": opp.response_due_at,
            "evidence_ids": list(opp.evidence_ids), "detail_observed": opp.detail_observed,
            "material": material, "known_at": ts,
        })
        self._append({
            "rec": "evidence_change", "opportunity_id": oid, "kind": kind.value,
            "prior_digest": prior or "", "new_digest": digest, "changed_fields": changed, "at": ts,
        })
        return EvidenceChange(oid, kind, prior or "", digest, changed, ts)

    # ── read-side helpers ──
    def known_opportunity_ids(self) -> set[str]:
        return set(self._opp_digest)

    def observation_count(self) -> int:
        return len(self._obs_ids)

    def opportunity_digest_of(self, opportunity_id: str) -> Optional[str]:
        return self._opp_digest.get(opportunity_id)
