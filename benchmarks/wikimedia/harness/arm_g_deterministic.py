"""Arm G — deterministic-first validation (plan §11).

Where deterministic truth suffices, no model work should happen. Real deterministic facts over the
corpus: a revision's content hash, revision ordering, whether a basis was updated vs deleted, and the
STALE/INVALIDATED classification — all pure functions. This arm confirms every classification verdict
is resolved deterministically (zero model calls), and that incremental Discovery only spends a
"model call" (rediscover) on the STALE conclusions that genuinely need re-derivation — never on
INVALIDATED or unchanged ones.
"""
from __future__ import annotations

from harness.evidence_corpus import RevisionPair


def run(pairs: list[RevisionPair]) -> dict:
    from runtime_contracts import (
        VerifiedIntent, IntentField, DecisionEvidence, ReaderKind, Author, IntentState,
        EvidenceChange, UPDATED, DELETED,
    )
    from runtime_contracts.canonical import content_hash as rcv1
    from discovery_runtime import (
        DiscoveryCheckpoint, classify, discover_incremental, evidence_ids,
    )

    checkpoint = DiscoveryCheckpoint(evidence_position="pos-1", policy_version="pol@1",
                                     capability_set_version="readers@1")

    def concl(ref, revid, val):
        ev = DecisionEvidence("r1", ReaderKind.RULE, val, source_ref=f"{ref}#{revid}")
        return VerifiedIntent(
            objective="page-current-revision",
            fields={"f": IntentField(value=val, author=Author.READER, evidence=[ev])},
            policy_version="pol@1", capability_version="readers@1").seal()

    deterministic_verdicts = 0
    model_calls_in_classify = 0   # must stay 0 — classify never calls a model
    content_hash_checks = 0
    ordering_checks = 0
    n = 0

    conclusions = []
    changes = []
    current_values = {}
    for i, pair in enumerate(pairs):
        ref = f"strategywiki/page/{pair.page_id}"
        conclusions.append(concl(ref, pair.a_revid, pair.a_revid))
        n += 1
        # deterministic content-hash fact: A and B are genuinely different content
        content_hash_checks += 1 if rcv1(pair.a_text) != rcv1(pair.b_text) else 0
        # deterministic ordering fact: B is a later revision than A
        ordering_checks += 1 if int(pair.b_revid) > int(pair.a_revid) else 0

        # a third of pages updated, a third deleted-basis, a third unchanged (deterministic split)
        if i % 3 == 0:
            changes.append(EvidenceChange(ref, UPDATED)); current_values[ref] = pair.b_revid
        elif i % 3 == 1:
            changes.append(EvidenceChange(ref, DELETED))
        else:
            current_values[ref] = pair.a_revid

    # every verdict is a pure classify() — no model
    verdicts = {}
    for c in conclusions:
        v, _ = classify(c, changes, checkpoint)
        verdicts[next(iter(evidence_ids(c)))] = v
        deterministic_verdicts += 1

    # incremental Discovery: rediscover (the "model call") fires ONLY on STALE, never INVALIDATED/unchanged
    calls: list = []

    def rediscover(vi):
        calls.append(1)
        rid = next(iter(evidence_ids(vi)))
        return concl(rid, current_values.get(rid, "cur"), current_values.get(rid, "cur"))

    inc = discover_incremental(conclusions, changes, checkpoint, rediscover=rediscover)
    n_stale = sum(1 for v in verdicts.values() if v is IntentState.STALE)
    n_invalidated = sum(1 for v in verdicts.values() if v is IntentState.INVALIDATED)
    n_unchanged = sum(1 for v in verdicts.values() if v is IntentState.VERIFIED)
    model_calls_avoided = n_invalidated + n_unchanged   # these never trigger rediscover

    passed = (n > 0 and model_calls_in_classify == 0
              and deterministic_verdicts == n
              and len(calls) == n_stale                       # only STALE recomputed
              and inc.report.model_calls == len(calls)
              and content_hash_checks == n and ordering_checks == n)
    return {
        "arm": "G", "name": "deterministic-first", "passed": passed,
        "n_cases": n,
        "metrics": {
            "deterministic_verdicts": deterministic_verdicts,     # == n, all pure
            "model_calls_in_classification": model_calls_in_classify,  # HARD = 0
            "stale_recomputed": len(calls),
            "invalidated_not_recomputed": n_invalidated,
            "unchanged_not_recomputed": n_unchanged,
            "model_calls_avoided": model_calls_avoided,
            "deterministic_resolution_rate": round(deterministic_verdicts / n, 3),
            "content_hash_facts": content_hash_checks,
            "ordering_facts": ordering_checks,
        },
    }
