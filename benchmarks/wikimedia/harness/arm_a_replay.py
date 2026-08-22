"""Arm A — point-in-time context and exact replay (plan §5).

For each real (revision A → revision B) pair: ingest A into RAG as canonical evidence, create a Mission
bound to A's EvidenceRef, advance the RAG source to B, then restart the runtime (fresh process-level
MissionRuntime on the same on-disk event log) and prove exact replay still resolves A — same intent
content hash, same ContextEpoch, reproduced plan fingerprint — while explicit re-evaluation resolves B.

Hard gates (plan §5): wrong-version substitution = 0, silent divergent replay = 0.
"""
from __future__ import annotations

from harness.embedder import StubEmbedder
from harness.evidence_corpus import RevisionPair

GRANTS = ["billing:write", "support:write", "books:write", "compliance:write"]


def _intent_from_rag(store, body, version):
    from redevops_rag.evidence import evidence_ref_from_hit
    hits = store.semantic_search(body, top_k=1, threshold=0.0, source_version=version)
    if not hits:
        return None, None
    er = evidence_ref_from_hit(hits[0])
    if er is None:
        return None, None
    return {"content_hash": er.content_hash, "produced_by": "discovery@0.3.0",
            "evidence": [{"field": "page", "source_ref": er.pin()}]}, er


def run(pairs: list[RevisionPair], workdir: str) -> dict:
    from redevops_rag.store import Store
    from redevops_rag.evidence import EvidenceRevision, ingest_revision, evidence_ref_from_hit
    from agentic_os.mission.demo import build_fleet
    from agentic_os.mission.executor import Executor
    from agentic_os.mission.runtime import MissionRuntime, ReplayError, ReplayDivergence
    from agentic_os.mission.store import EventStore
    from agentic_os.mission.context_view import epoch_from_refs

    def runtime(store):
        reg, client = build_fleet()
        return MissionRuntime(reg, Executor(client), store=store)

    def plan_meta(rt, mid):
        return next(e for e in rt.store.for_mission(mid) if e.type == "PlanCreated").payload

    n = 0
    replay_success = exact_recovery = wrong_version = 0
    fp_reproduced = epoch_reproduced = reeval_resolved_B = 0
    replay_divergences = 0
    latencies_ms: list[float] = []
    import time

    for i, pair in enumerate(pairs):
        ref = f"strategywiki/page/{pair.page_id}"
        rag = Store(StubEmbedder(), ":memory:")
        ingest_revision(rag, rag.embedder, EvidenceRevision(
            ref=ref, version=pair.a_revid, content=pair.a_text,
            observed_at=pair.a_ts, source="wikimedia"))
        rag.reindex_fts()
        intent_A, er_A = _intent_from_rag(rag, pair.a_text, pair.a_revid)
        if intent_A is None:
            continue
        n += 1

        path = f"{workdir}/arm_a_{i}.jsonl"
        rt = runtime(EventStore(path=path))
        m = rt.create_mission("Summarize the page", policy_refs=GRANTS, template="onboarding",
                              verified_intent=intent_A)
        mid = m.id
        sealed = plan_meta(rt, mid)
        epoch_A, fp_A = sealed["context_epoch_id"], sealed["plan_fingerprint"]

        # advance the RAG source A -> B (retention keeps A addressable)
        ingest_revision(rag, rag.embedder, EvidenceRevision(
            ref=ref, version=pair.b_revid, content=pair.b_text,
            observed_at=pair.b_ts, source="wikimedia"))
        rag.reindex_fts()

        # crash + restart: fresh runtime, same on-disk log
        t0 = time.perf_counter()
        try:
            rt2 = runtime(EventStore(path=path))
            m2 = rt2.rehydrate(mid)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            replay_success += 1
            # exact-revision recovery: the replayed evidence is A, never B
            if m2.intent_content_hash == er_A.content_hash and m2.evidence_refs == [f"page:{er_A.pin()}"]:
                exact_recovery += 1
            # a true substitution = the replayed evidence's VERSION is B's revid. The version sits
            # between '@' and '#' in the pin (ref@version#hash), so match the delimited form — a bare
            # substring would false-positive on a short revid appearing inside the content-hash hex.
            if any(f"@{pair.b_revid}#" in ref for ref in m2.evidence_refs):
                wrong_version += 1
            if m2.context_epoch_id == epoch_A and \
               m2.context_epoch_id == epoch_from_refs([f"page:{er_A.pin()}"], pins=[er_A.content_hash]).id:
                epoch_reproduced += 1
            if plan_meta(rt2, mid)["plan_fingerprint"] == fp_A:
                fp_reproduced += 1
        except (ReplayError, ReplayDivergence):
            replay_divergences += 1  # a fail-closed divergence — recorded, not silent

        # the point-in-time guarantee: A's evidence is STILL retrievable from RAG after B
        pinned = rag.semantic_search(pair.a_text, top_k=1, threshold=0.0, source_version=pair.a_revid)
        if pinned and evidence_ref_from_hit(pinned[0]).pin() != er_A.pin():
            wrong_version += 1

        # explicit re-evaluation resolves B
        intent_B, er_B = _intent_from_rag(rag, pair.b_text, pair.b_revid)
        if intent_B is not None:
            m3 = rt.re_evaluate(mid, verified_intent=intent_B, cause="source revised A->B")
            if m3.intent_content_hash == er_B.content_hash and m3.context_epoch_id != epoch_A:
                reeval_resolved_B += 1

    p50 = sorted(latencies_ms)[len(latencies_ms) // 2] if latencies_ms else 0.0
    passed = (n > 0 and wrong_version == 0 and replay_divergences == 0
              and exact_recovery == n and fp_reproduced == n and epoch_reproduced == n
              and reeval_resolved_B == n)
    return {
        "arm": "A", "name": "point-in-time context + exact replay", "passed": passed,
        "n_cases": n,
        "metrics": {
            "replay_success": replay_success,
            "exact_revision_recovery": exact_recovery,
            "wrong_version_substitution": wrong_version,   # HARD GATE = 0
            "silent_replay_divergence": replay_divergences,  # HARD GATE = 0
            "fingerprint_reproduced": fp_reproduced,
            "epoch_reproduced": epoch_reproduced,
            "reevaluation_resolved_B": reeval_resolved_B,
            "replay_latency_ms_p50": round(p50, 3),
        },
    }
