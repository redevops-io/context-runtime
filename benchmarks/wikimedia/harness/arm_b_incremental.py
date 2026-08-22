"""Arm B — incremental Discovery vs full rescan (plan §6).

At T0, one sealed conclusion per content page (its evidence = the page's revision). Between T0 and T1
a subset of pages receive a real later revision → typed ``EvidenceChange[]``. The incremental path
must reach the SAME valid state as a full rescan while re-deriving only the affected conclusions.

Required semantic gate (plan §6): valid_state(incremental) == valid_state(full). Only then is the
work reduction (conclusions recomputed / model calls / bytes) reported.
"""
from __future__ import annotations

from harness.evidence_corpus import RevisionPair


def _build(pairs, changed_idx):
    """Return (conclusions_at_T0, changes, current_values). Conclusions value = the revid that is
    'current' as of T0 (a_revid); a changed page advances to b_revid at T1."""
    from runtime_contracts import (
        VerifiedIntent, IntentField, DecisionEvidence, ReaderKind, Author,
    )
    conclusions = []
    current_values = {}
    changes = []
    from runtime_contracts import EvidenceChange, UPDATED
    for i, pair in enumerate(pairs):
        ref = f"strategywiki/page/{pair.page_id}"
        ev = DecisionEvidence("r1", ReaderKind.RULE, pair.a_revid, source_ref=f"{ref}#{pair.a_revid}")
        vi = VerifiedIntent(
            objective="page-current-revision",
            fields={"f": IntentField(value=pair.a_revid, author=Author.READER, evidence=[ev])},
            policy_version="pol@1", capability_version="readers@1").seal()
        conclusions.append(vi)
        if i in changed_idx:
            current_values[ref] = pair.b_revid           # the page really advanced A->B
            changes.append(EvidenceChange(ref, UPDATED))
        else:
            current_values[ref] = pair.a_revid           # unchanged
    return conclusions, changes, current_values


def _rediscover(current_values, counter):
    from runtime_contracts import (
        VerifiedIntent, IntentField, DecisionEvidence, ReaderKind, Author,
    )
    from discovery_runtime import evidence_ids

    def rediscover(vi):
        counter.append(1)
        rid = next(iter(evidence_ids(vi)))
        val = current_values[rid]
        ev = DecisionEvidence("r1", ReaderKind.RULE, val, source_ref=f"{rid}#cur")
        return VerifiedIntent(
            objective="page-current-revision",
            fields={"f": IntentField(value=val, author=Author.READER, evidence=[ev])},
            policy_version="pol@1", capability_version="readers@1").seal()
    return rediscover


def _valid_map(concls):
    from discovery_runtime import evidence_ids
    from runtime_contracts import IntentState
    return {next(iter(evidence_ids(c))): c.fields["f"].value
            for c in concls if c.state is IntentState.VERIFIED}


def run(pairs: list[RevisionPair]) -> dict:
    from discovery_runtime import DiscoveryCheckpoint, discover_incremental, discover_full

    # half the pages change between T0 and T1 (deterministic: even indices)
    changed_idx = {i for i in range(len(pairs)) if i % 2 == 0}
    checkpoint = DiscoveryCheckpoint(evidence_position="pos-1", policy_version="pol@1",
                                     capability_set_version="readers@1")

    # FULL rescan — re-derive every conclusion regardless of what changed
    conclusions, changes, current = _build(pairs, changed_idx)
    full_calls: list = []
    full = discover_full(conclusions, checkpoint, rediscover=_rediscover(current, full_calls))

    # INCREMENTAL — re-derive only the conclusions the change set touches
    conclusions2, changes2, current2 = _build(pairs, changed_idx)
    inc_calls: list = []
    inc = discover_incremental(conclusions2, changes2, checkpoint,
                               rediscover=_rediscover(current2, inc_calls))

    vm_full, vm_inc = _valid_map(full.conclusions), _valid_map(inc.conclusions)
    equivalent = vm_full == vm_inc and len(vm_full) == len(pairs)

    fr, ir = full.report, inc.report
    passed = equivalent and len(inc_calls) < len(full_calls) and len(inc_calls) == len(changed_idx)
    return {
        "arm": "B", "name": "incremental Discovery vs full rescan", "passed": passed,
        "n_cases": len(pairs),
        "metrics": {
            "valid_state_equivalent": equivalent,          # HARD GATE
            "pages_changed": len(changed_idx),
            "full_conclusions_recomputed": len(full_calls),
            "incremental_conclusions_recomputed": len(inc_calls),
            "full_model_calls": fr.model_calls,
            "incremental_model_calls": ir.model_calls,
            "full_examined": fr.examined,
            "incremental_examined": ir.examined,
            "recompute_reduction_ratio": round(len(full_calls) / max(1, len(inc_calls)), 3),
        },
    }
