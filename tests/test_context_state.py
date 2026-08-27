"""Contract tests for ContextState (F1).

F1 is wiring: the runtime's evidence becomes the canonical runtime-contracts ContextView. The properties
that matter: reproducible identity (same pinned evidence → same view_hash), an honest REQUIRED/OPTIONAL/
EXCLUDED decision record, and correct incremental evolution under EvidenceChange — an UPDATE moves a pin
(replay honestly diverges), a DELETE removes the basis (the item is EXCLUDED and a REQUIRED plan turns
infeasible). Requires runtime_contracts; skipped cleanly when absent.
"""
from __future__ import annotations

import pytest

pytest.importorskip("runtime_contracts")

from runtime_contracts.models.context import Necessity, PlanFeasibility  # noqa: E402
from runtime_contracts.protocol.evidence import DELETED, UPDATED, EvidenceChange, EvidenceRef  # noqa: E402

from context_runtime.context_state import (  # noqa: E402
    apply_change, build_context_view, state_only_handles,
)

HITS = [
    {"chunk_id": "doc1", "version": "1", "content_hash": "rcv1:aaa", "text": "alpha " * 20},
    {"chunk_id": "doc2", "version": "1", "content_hash": "rcv1:bbb", "text": "beta " * 20},
    {"chunk_id": "doc3", "version": "1", "content_hash": "rcv1:ccc", "text": "gamma " * 20},
]


def test_maps_hits_to_a_context_view_with_pins():
    v = build_context_view(HITS, required_ids=["doc1"])
    necs = {i.handle.artifact_id.split("@")[0]: i.necessity for i in v.plan.items}
    assert necs["doc1"] is Necessity.REQUIRED and necs["doc2"] is Necessity.OPTIONAL
    assert v.version_pins["doc1@1"] == "rcv1:aaa"           # identity-transparent: pin = content hash
    assert v.plan.is_feasible


def test_view_hash_is_reproducible_and_pin_sensitive():
    a = build_context_view(HITS, view_id="x", materialized_at="2026-01-01T00:00:00Z")
    b = build_context_view(HITS, view_id="y-different-id", materialized_at="2026-09-09T00:00:00Z")
    assert a.view_hash == b.view_hash                       # view_id + clock are excluded from identity
    # move one pin → the view is a different reproducibility unit, and divergence names the artifact
    moved = build_context_view(
        [{**HITS[0], "version": "2", "content_hash": "rcv1:zzz"}, HITS[1], HITS[2]])
    assert moved.view_hash != a.view_hash
    assert a.divergence_from(moved) == ["doc1@1", "doc1@2"]


def test_excluded_candidates_are_on_the_record():
    v = build_context_view(HITS[:1], excluded=[HITS[2]])
    ex = [i for i in v.plan.items if i.necessity is Necessity.EXCLUDED]
    assert len(ex) == 1 and ex[0].handle.artifact_id.startswith("doc3")
    assert ex[0].reason                                     # the omission carries a reason


def test_update_moves_the_pin():
    v = build_context_view(HITS)
    change = EvidenceChange(ref="doc2", change_type=UPDATED,
                            prior=EvidenceRef(ref="doc2", content_hash="rcv1:bbb", version="1"),
                            new=EvidenceRef(ref="doc2", content_hash="rcv1:bbb2", version="2"))
    v2 = apply_change(v, change)
    assert "doc2@1" not in v2.version_pins and v2.version_pins["doc2@2"] == "rcv1:bbb2"
    assert v2.view_hash != v.view_hash                      # replay honestly diverges
    assert v.version_pins["doc2@1"] == "rcv1:bbb"           # the original view is immutable


def test_delete_removes_basis_and_makes_a_required_plan_infeasible():
    v = build_context_view(HITS, required_ids=["doc2"], budget_tokens=100000)
    assert v.plan.feasibility is PlanFeasibility.FEASIBLE
    change = EvidenceChange(ref="doc2", change_type=DELETED,
                            prior=EvidenceRef(ref="doc2", content_hash="rcv1:bbb", version="1"))
    assert change.removes_basis
    v2 = apply_change(v, change)
    gone = next(i for i in v2.plan.items if i.handle.artifact_id.startswith("doc2"))
    assert gone.necessity is Necessity.EXCLUDED and "basis" in gone.reason.lower()
    assert "doc2@1" not in v2.version_pins
    # the REQUIRED evidence is gone → the plan must not silently answer without it
    assert not v2.plan.required                             # doc2 dropped out of REQUIRED (now EXCLUDED)


def test_state_only_is_the_required_set():
    v = build_context_view(HITS, required_ids=["doc1", "doc3"])
    ids = {h.artifact_id.split("@")[0] for h in state_only_handles(v)}
    assert ids == {"doc1", "doc3"}


def test_tenant_scope_is_carried_onto_handles():
    v = build_context_view(HITS, tenant="acme")
    assert all(i.handle.tenancy.tenant_id == "acme" for i in v.plan.items)
