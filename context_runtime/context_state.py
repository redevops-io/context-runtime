"""ContextState (F1) — represent the runtime's working set as the canonical runtime-contracts ContextView.

F1 is **wiring, not a new subsystem** (audit): the canonical types already exist in `runtime-contracts`
(`ContextView`, `ContextPreviewPlan`, `PlannedItem`/`Necessity`, `EvidenceChange.removes_basis`). Forking a
parallel "context state" type is forbidden (IMPLEMENTATION_SPLIT.md §15). So this module *consumes* those
types: it maps the runtime's retrieved evidence (`Hit`s — objects or the benchmark's dicts) onto a
`ContextView`, and evolves it incrementally under typed `EvidenceChange`s.

What that buys the rest of the stack:
  • a **reproducible** working set — `view_hash` is stable across re-materializations of the same pinned
    evidence, and `divergence_from` says exactly which pins moved when a replay differs;
  • an **honest** decision record — REQUIRED / OPTIONAL / EXCLUDED per item, so "why did the model not see
    the refuting document?" is answerable after the fact, and a plan that can't fit its REQUIRED set is
    reported infeasible rather than silently dropping it;
  • the **ContextState** that F2's materialization ladder reads: STATE_ONLY is the view's REQUIRED set, and
    an `EvidenceChange` whose `removes_basis` is true (a DELETE) is exactly when the state must re-materialize.

The view carries *pins* (content hashes), not bytes — materializing the actual text stays the store's job;
F1 owns the identity/decision layer. Requires the neutral `runtime_contracts` package (an optional extra);
importing this module without it raises, so callers that want ContextState opt into that dependency.
"""
from __future__ import annotations

from typing import Any, Iterable

from runtime_contracts.models.context import (
    ContextPreviewPlan, ContextView, Necessity, PlannedItem,
)
from runtime_contracts.models.handle import ArtifactHandle
from runtime_contracts.models.visibility import Tenancy, Visibility
from runtime_contracts.protocol.evidence import EvidenceChange

_TOK = 4  # ~chars per token (matches the benchmark suite's estimate)


def _tenancy(tenant: str | None) -> Tenancy:
    # A tenant-scoped view carries PRIVATE tenancy so the runtime-contracts authorization checks bind it to
    # its tenant; an unscoped (single-tenant/public) view is PUBLIC. Keeps F1 on the same isolation footing
    # as F5/F4/F3.
    return Tenancy(visibility=Visibility.PRIVATE, tenant_id=tenant) if tenant else Tenancy(visibility=Visibility.PUBLIC)


def _get(hit: Any, field: str, default=None):
    return hit.get(field, default) if isinstance(hit, dict) else getattr(hit, field, default)


def _handle(hit: Any, tenant: str | None = None) -> ArtifactHandle:
    cid = str(_get(hit, "chunk_id") or _get(hit, "id") or _get(hit, "document_id"))
    ver = str(_get(hit, "version") or "1")
    chash = str(_get(hit, "content_hash") or "")
    text = _get(hit, "text") or ""
    return ArtifactHandle(
        artifact_id=f"{cid}@{ver}",          # runtime-contracts requires a version-pinned id
        artifact_type="chunk",
        version=ver,
        artifact_content_hash=chash,
        tenancy=_tenancy(tenant),
        estimated_expansion_tokens=len(text) // _TOK,
        projections=("full", "summary"),
    )


def build_context_view(hits: Iterable[Any], *, required_ids: Iterable[str] = (),
                       excluded: Iterable[Any] = (), budget_tokens: int | None = None, tenant: str | None = None,
                       view_id: str = "cv", plan_id: str = "cp", materialized_at: str | None = None) -> ContextView:
    """Map retrieved evidence onto a ContextView. ``required_ids`` (by chunk/doc id) are REQUIRED — their
    absence invalidates the answer and is never a budget drop; the rest are OPTIONAL; ``excluded`` are
    candidates that were considered and dropped, retained as EXCLUDED items so the omission is on the record.
    ``version_pins`` come from each hit's content hash → the view is identity-transparent and replay-honest."""
    required = {str(r) for r in required_ids}
    items: list[PlannedItem] = []
    pins: dict[str, str] = {}
    est = 0
    for hit in hits:
        h = _handle(hit, tenant)
        base = h.artifact_id.split("@", 1)[0]
        nec = Necessity.REQUIRED if base in required else Necessity.OPTIONAL
        items.append(PlannedItem(handle=h, necessity=nec, projection="full"))
        pins[h.artifact_id] = h.artifact_content_hash
        est += h.estimated_expansion_tokens or 0
    for ex in excluded:
        items.append(PlannedItem(handle=_handle(ex, tenant), necessity=Necessity.EXCLUDED,
                                 projection="summary", reason="considered, not selected"))
    plan = ContextPreviewPlan(plan_id=plan_id, items=items, budget_tokens=budget_tokens,
                              estimated_tokens=est, omitted_count=0)
    return ContextView(view_id=view_id, plan=plan, version_pins=pins, materialized_at=materialized_at)


def apply_change(view: ContextView, change: EvidenceChange, *, view_id: str | None = None) -> ContextView:
    """Evolve a ContextView under one typed evidence delta, returning a NEW view (the old one is immutable
    and stays replayable). An UPDATE moves the item's version pin → view_hash changes → replay is honestly
    reported as divergent. A DELETE (``change.removes_basis``) marks the item EXCLUDED and drops its pin —
    the evidentiary basis is gone, so a REQUIRED item can no longer be satisfied and the plan turns
    infeasible rather than silently answering without it."""
    items: list[PlannedItem] = []
    pins = dict(view.version_pins)
    for item in view.plan.items:
        base = item.handle.artifact_id.split("@", 1)[0]
        if base != change.ref:
            items.append(item)
            continue
        if change.removes_basis:                          # DELETE: basis removed
            pins.pop(item.handle.artifact_id, None)
            items.append(PlannedItem(handle=item.handle, necessity=Necessity.EXCLUDED,
                                     projection=item.projection, authorization=item.authorization,
                                     reason="evidence deleted — basis removed"))
        elif change.new is not None:                      # UPDATE: pin moves to the new revision
            new = change.new
            h = ArtifactHandle(artifact_id=f"{change.ref}@{new.version or '1'}", artifact_type="chunk",
                               version=str(new.version or "1"), artifact_content_hash=new.content_hash,
                               tenancy=item.handle.tenancy,   # tenancy is preserved across a revision bump
                               estimated_expansion_tokens=item.handle.estimated_expansion_tokens,
                               projections=item.handle.projections)
            pins.pop(item.handle.artifact_id, None)
            pins[h.artifact_id] = h.artifact_content_hash
            items.append(PlannedItem(handle=h, necessity=item.necessity, projection=item.projection,
                                     authorization=item.authorization))
        else:
            items.append(item)
    plan = ContextPreviewPlan(plan_id=view.plan.plan_id, items=items, budget_tokens=view.plan.budget_tokens,
                              estimated_tokens=view.plan.estimated_tokens, omitted_count=view.plan.omitted_count)
    return ContextView(view_id=view_id or view.view_id, plan=plan, version_pins=pins,
                       materialized_at=view.materialized_at)


def state_only_handles(view: ContextView) -> list[ArtifactHandle]:
    """The STATE_ONLY materialization: the REQUIRED items — the minimal set whose absence invalidates the
    answer. F2's ladder materializes exactly these at its cheapest depth."""
    return [i.handle for i in view.plan.required]
