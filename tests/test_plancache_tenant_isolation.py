"""SnapshotPlanCache — cross-tenant isolation (F5 security regression, Scenario E).

The plan-cache key must bind a sealed plan to the *tenant* and *effective permission set* that produced
it, not only to sensitivity. Before the v2-scope fix, ``build_key`` set ``policy_fingerprint =
_h(sensitivity)`` — so two tenants issuing the same intent against the same pinned sources at the same
sensitivity collided on one key. Per-tenant policy is evaluated on the cache-MISS path only, so on the
second (HIT) call tenant B was served tenant A's authorized plan, silently bypassing B's policy. This is
a cross-tenant isolation failure, not a caching nicety; these tests fail closed if it ever regresses.

There was no cross-tenant harness before this file — it is net-new (plan Scenario E).
"""
from __future__ import annotations

from context_runtime.runtime.runtime import ContextRuntime
from context_runtime.types import Constraints, SourceRef
from context_runtime.plancache.cache import SnapshotPlanCache, build_key

DOCS = [
    {"chunk_id": "deploy.md::0", "filename": "deploy.md",
     "text": "Deployment X failed: readiness probe timed out after the Cloudflare cert expired.",
     "created_at": None},
]
SRC = [SourceRef(name="kb", version="v1")]
Q = "why did deployment X fail"


def _c(tenant=None, permissions=(), sensitivity="public"):
    return Constraints(tenant=tenant, permissions=tuple(permissions), sensitivity=sensitivity)


def test_different_tenants_do_not_share_a_cache_key():
    """Same intent + same pinned sources + same sensitivity, differing only by tenant ⇒ distinct keys."""
    rt = ContextRuntime.default(DOCS)
    g_a = rt._coerce_goal(Q, SRC, _c(tenant="acme"))
    g_b = rt._coerce_goal(Q, SRC, _c(tenant="globex"))
    _, _, i_a = rt._make_plan(g_a)
    _, _, i_b = rt._make_plan(g_b)
    k_a, k_b = build_key(i_a, g_a), build_key(i_b, g_b)
    assert k_a != k_b
    assert k_a.policy_fingerprint != k_b.policy_fingerprint


def test_tenant_b_never_receives_tenant_a_cached_plan():
    """End-to-end: tenant A seeds the cache, tenant B's identical request MISSES (re-plans under B)."""
    rt = ContextRuntime.default(DOCS)
    assert isinstance(rt.plan_cache, SnapshotPlanCache)
    a1 = rt.plan(Q, sources=SRC, constraints=_c(tenant="acme"))
    assert a1.cache in ("miss", "bypass")
    a2 = rt.plan(Q, sources=SRC, constraints=_c(tenant="acme"))     # A replays its own plan
    assert a2.cache == "hit"
    b1 = rt.plan(Q, sources=SRC, constraints=_c(tenant="globex"))   # B must NOT hit A's entry
    assert b1.cache in ("miss", "bypass")


def test_differing_permissions_within_a_tenant_do_not_share_a_key():
    """Two principals in one tenant with different effective permissions get distinct plans."""
    rt = ContextRuntime.default(DOCS)
    g_lo = rt._coerce_goal(Q, SRC, _c(tenant="acme", permissions=("kb:read",)))
    g_hi = rt._coerce_goal(Q, SRC, _c(tenant="acme", permissions=("kb:read", "pii:read")))
    _, _, i_lo = rt._make_plan(g_lo)
    _, _, i_hi = rt._make_plan(g_hi)
    assert build_key(i_lo, g_lo) != build_key(i_hi, g_hi)


def test_permission_order_is_canonical():
    """Permission-set identity is order-independent (same set ⇒ same key)."""
    rt = ContextRuntime.default(DOCS)
    g1 = rt._coerce_goal(Q, SRC, _c(tenant="acme", permissions=("a", "b")))
    g2 = rt._coerce_goal(Q, SRC, _c(tenant="acme", permissions=("b", "a")))
    _, _, i1 = rt._make_plan(g1)
    _, _, i2 = rt._make_plan(g2)
    assert build_key(i1, g1) == build_key(i2, g2)


def test_single_tenant_default_still_replays():
    """Backward-compat: with no tenant/permissions set, identical requests still replay (HIT)."""
    rt = ContextRuntime.default(DOCS)
    p1 = rt.plan(Q, sources=SRC)
    assert p1.cache in ("miss", "bypass")
    p2 = rt.plan(Q, sources=SRC)
    assert p2.cache == "hit"
