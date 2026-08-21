"""SnapshotPlanCache — deterministic replay keyed on the pinned evidence identity (v0.2.x Slice 2).

source_fingerprint is now load-bearing: same intent + same pinned source versions ⇒ replay HIT (reuse
the sealed plan); mutate a source's version ⇒ new source_fingerprint ⇒ MISS (re-plan = re-evaluation).
"""
from __future__ import annotations

from context_runtime.runtime.runtime import ContextRuntime
from context_runtime.types import SourceRef
from context_runtime.plancache.cache import SnapshotPlanCache, NullPlanCache, build_key

DOCS = [
    {"chunk_id": "deploy.md::0", "filename": "deploy.md",
     "text": "Deployment X failed: readiness probe timed out after the Cloudflare cert expired.",
     "created_at": None},
]


def test_default_is_snapshot_cache_and_replays_identical_requests():
    rt = ContextRuntime.default(DOCS)
    assert isinstance(rt.plan_cache, SnapshotPlanCache)
    src = [SourceRef(name="kb", version="v1")]
    p1 = rt.plan("why did deployment X fail", sources=src)
    assert p1.cache in ("miss", "bypass")
    p2 = rt.plan("why did deployment X fail", sources=src)   # same pinned evidence → replay HIT
    assert p2.cache == "hit"


def test_mutating_a_source_version_misses_the_cache():
    rt = ContextRuntime.default(DOCS)
    q = "why did deployment X fail"
    rt.plan(q, sources=[SourceRef(name="kb", version="v1")])                 # seed
    hit = rt.plan(q, sources=[SourceRef(name="kb", version="v1")])
    assert hit.cache == "hit"
    # evidence mutates v1 → v2: source_fingerprint changes, so replay must NOT reuse the v1 plan
    miss = rt.plan(q, sources=[SourceRef(name="kb", version="v2")])
    assert miss.cache in ("miss", "bypass")


def test_source_fingerprint_distinguishes_versions():
    rt = ContextRuntime.default(DOCS)
    g1 = rt._coerce_goal("q", [SourceRef(name="kb", version="v1")], None)
    g2 = rt._coerce_goal("q", [SourceRef(name="kb", version="v2")], None)
    _, _, i1 = rt._make_plan(g1)
    _, _, i2 = rt._make_plan(g2)
    assert build_key(i1, g1).source_fingerprint != build_key(i2, g2).source_fingerprint


def test_learning_runtime_opts_out_of_caching():
    # an online optimizer must re-select every call, so a learning runtime keeps the always-miss stub
    rt = ContextRuntime.default(DOCS, learning=True)
    assert isinstance(rt.plan_cache, NullPlanCache)
