"""DSpark-inspired additions: score calibration, load meter, cost profile, load-aware
sizing, and their opt-in wiring into the LibreChat tenant. Legacy behaviour (no map, no
meter) must be byte-for-byte unchanged — asserted alongside the new paths."""
from __future__ import annotations

from context_runtime.costmodel.profile import CostProfile
from context_runtime.integrations.calibration import (
    CalibrationLog,
    CalibrationMap,
    _isotonic_fit,
    fit_from_log,
)
from context_runtime.integrations.librechat import (
    DEFAULT_STRATEGIES,
    LibreChatTenant,
    reward_from_judgment,
)
from context_runtime.integrations.loadmeter import LoadMeter
from context_runtime.scheduler.load_aware import size_expensive_stage
from context_runtime.types import Hit


# ──────────────────────────── calibration ────────────────────────────


def test_isotonic_is_monotone():
    bps = _isotonic_fit([(0.9, 1.0), (0.1, 0.0), (0.3, 1.0), (0.2, 0.0), (0.15, 1.0)])
    ys = [y for _, y in bps]
    assert ys == sorted(ys)  # non-decreasing


def test_fit_apply_roundtrip(tmp_path):
    log = CalibrationLog(tmp_path / "log.jsonl")
    for i in range(30):
        hi, lo = 0.6 + (i % 5) * 0.05, 0.1 + (i % 5) * 0.03
        log.append("hybrid", "lookup", 0.8,
                   [{"chunk_id": "a", "score": hi, "rel": 1.0},
                    {"chunk_id": "b", "score": lo, "rel": 0.0}])
    cmap = fit_from_log(log, min_samples=20)
    out = tmp_path / "cal.json"
    cmap.save(out)
    loaded = CalibrationMap.load(out)
    assert loaded.has("hybrid")
    assert loaded.apply("hybrid", 0.1) <= loaded.apply("hybrid", 0.5) <= loaded.apply("hybrid", 0.85)
    assert loaded.apply("hybrid", 0.85) > loaded.apply("hybrid", 0.1)


def test_unfit_method_is_identity():
    cmap = CalibrationMap.load  # sanity: empty map returns identity
    m = CalibrationMap({})
    assert m.apply("bm25", 0.42) == 0.42
    assert not m.has("bm25")


def test_calibration_log_append_and_read(tmp_path):
    log = CalibrationLog(tmp_path / "l.jsonl")
    log.append("bm25", "lookup", 0.5, [{"chunk_id": "x", "score": 0.3, "rel": None}])
    log.append("vector", "howto", 0.9, [{"chunk_id": "y", "score": 0.8, "rel": 1.0}])
    rows = log.rows()
    assert len(rows) == 2 and rows[0]["method"] == "bm25" and rows[1]["hits"][0]["rel"] == 1.0


# ──────────────────────────── load meter ────────────────────────────


def test_load_meter_bands():
    m = LoadMeter(mid=2, hi=4)
    assert m.band() == "lo"
    m.enter(); m.enter()
    assert m.band() == "mid"
    m.enter(); m.enter()
    assert m.band() == "hi"
    m.leave(); m.leave(); m.leave(); m.leave()
    assert m.band() == "lo" and m.inflight() == 0


def test_load_meter_track_scope():
    m = LoadMeter(mid=1, hi=2)
    with m.track():
        assert m.inflight() == 1
    assert m.inflight() == 0


# ──────────────────────────── cost profile ────────────────────────────


def test_cost_profile_observe_and_bucket_fallback(tmp_path):
    p = CostProfile(tmp_path / "prof.json")
    p.observe("rerank", 8, 0.20)
    p.observe("rerank", 8, 0.30)
    assert abs(p.latency("rerank", 8) - 0.25) < 1e-6      # online mean
    assert p.latency("rerank", 6) is not None             # falls back to nearest smaller bucket
    assert p.latency("verify", 4) is None                 # unprofiled stage → None
    # persisted + reloaded
    p2 = CostProfile(tmp_path / "prof.json")
    assert p2.samples("rerank", 8) == 2


# ──────────────────────────── load-aware sizing ────────────────────────────


def test_sizer_idle_keeps_all_busy_prunes():
    probs = [0.95, 0.9, 0.8, 0.5, 0.2]
    lo = size_expensive_stage(probs, load_band="lo", requested_k=5, requested_rerank=True)
    hi = size_expensive_stage(probs, load_band="hi", requested_k=5, requested_rerank=True)
    assert lo.final_k == 5                    # idle: admit the whole block
    assert hi.final_k < lo.final_k            # busy: prune the low-survival tail
    assert hi.rerank is False                 # rerank dropped under heavy load
    assert lo.rerank is True


def test_sizer_never_exceeds_requested_k_and_keeps_one():
    probs = [0.99] * 10
    d = size_expensive_stage(probs, load_band="lo", requested_k=3, requested_rerank=False)
    assert d.final_k == 3                      # capped by the arm — never enlarges depth
    zero = size_expensive_stage([0.01, 0.01], load_band="hi", requested_k=2, requested_rerank=False)
    assert zero.final_k >= 1                    # always keep at least the top hit


def test_sizer_budget_guard_trims_to_latency(tmp_path):
    prof = CostProfile(tmp_path / "p.json")
    prof.observe("rerank", 4, 5.0)             # 4 candidates cost 5s
    prof.observe("rerank", 2, 1.0)             # 2 candidates cost 1s
    d = size_expensive_stage([0.99, 0.99, 0.99, 0.99], load_band="lo", requested_k=4,
                             requested_rerank=True, cost_profile=prof, max_latency_seconds=1.5)
    assert d.final_k <= 2                       # trimmed to fit the 1.5s ceiling


# ──────────────────────────── reward uses served relevance ────────────────────────────


def test_reward_legacy_when_no_rel_signal():
    s = DEFAULT_STRATEGIES[1]
    assert reward_from_judgment(0.8, s) == reward_from_judgment(0.8, s, rel_signal=None, beta=0.5)


def test_reward_blends_calibrated_relevance():
    s = DEFAULT_STRATEGIES[1]
    base = reward_from_judgment(0.5, s, rel_signal=None, beta=0.0)
    # a lukewarm judge (0.5) but highly relevant served hits (0.95) → reward rises with beta
    hi = reward_from_judgment(0.5, s, rel_signal=0.95, beta=0.5)
    lo = reward_from_judgment(0.5, s, rel_signal=0.05, beta=0.5)
    assert hi > base > lo


def test_tenant_reward_uses_hit_score():
    """With calibration + reward_beta, served-hit relevance actually moves the bandit reward."""
    strong = _StubRetriever([0.9, 0.85, 0.8])
    t = LibreChatTenant(retriever=strong, calibration=_linear_map(None), reward_beta=0.5)
    ctx = t.retrieve("q")
    r_relevant = t.record_judgment("q", 0.5)
    weak = _StubRetriever([0.1, 0.05, 0.02])
    t2 = LibreChatTenant(retriever=weak, calibration=_linear_map(None), reward_beta=0.5)
    t2.retrieve("q2")
    r_irrelevant = t2.record_judgment("q2", 0.5)
    assert r_relevant > r_irrelevant   # same judge score, different served relevance → different reward
    assert ctx.probs                    # sanity: calibration ran


# ──────────────────────────── tenant integration ────────────────────────────


class _StubRetriever:
    """Returns hits whose scores decay with rank, so calibration/abstention are testable."""

    def __init__(self, scores):
        self._scores = scores

    def index(self, path):  # pragma: no cover - unused
        return {}

    def search(self, query, k, method):
        return [Hit(chunk_id=f"c{i}", filename=f"f{i%2}.txt", text=f"passage {i} {query}",
                    score=s) for i, s in enumerate(self._scores[:k])]


def _linear_map(method):
    # identity-ish calibration: P(rel) == score, per method
    from context_runtime.integrations.calibration import _MethodCal
    bps = [(x / 10, x / 10) for x in range(0, 11)]
    return CalibrationMap({m: _MethodCal(breakpoints=bps, n=100) for m in
                           ("bm25", "vector", "hybrid", "graph", "community")})


def test_tenant_legacy_unchanged():
    """No calibration, no meter → probs empty, never abstains, plain bucket ctx."""
    t = LibreChatTenant(retriever=_StubRetriever([0.9, 0.8, 0.7, 0.6, 0.5]))
    ctx = t.retrieve("what is the operating margin")
    assert ctx.probs == () and ctx.abstain is False and ctx.max_p_rel == 1.0


def test_tenant_calibration_and_abstention():
    weak = _StubRetriever([0.1, 0.05, 0.02])       # everything looks irrelevant
    t = LibreChatTenant(retriever=weak, calibration=_linear_map(None), abstain_threshold=0.5)
    ctx = t.retrieve("obscure unanswerable query")
    assert ctx.probs and ctx.max_p_rel < 0.5 and ctx.abstain is True
    strong = _StubRetriever([0.9, 0.8, 0.7])
    t2 = LibreChatTenant(retriever=strong, calibration=_linear_map(None), abstain_threshold=0.5)
    ctx2 = t2.retrieve("answerable query")
    assert ctx2.max_p_rel >= 0.5 and ctx2.abstain is False


def test_tenant_logs_calibration_rows(tmp_path):
    log = CalibrationLog(tmp_path / "log.jsonl")
    t = LibreChatTenant(retriever=_StubRetriever([0.8, 0.6]), calib_log=log)
    t.handle("a question about revenue")           # retrieve → judge → learn → log
    rows = log.rows()
    assert rows and "hits" in rows[0] and rows[0]["hits"][0]["score"] > 0


def test_tenant_load_aware_bandit_ctx():
    """Load-aware mode must key the bandit by bucket:band and update the SAME key."""
    meter = LoadMeter(mid=1, hi=2)
    t = LibreChatTenant(retriever=_StubRetriever([0.9, 0.8, 0.7, 0.6]),
                        calibration=_linear_map(None), load_meter=meter, load_aware=True)
    with meter.track():                            # inflight=1 → band "mid"
        ctx = t.retrieve("q")
        t.record_judgment("q", 0.9)
    keys = list(t.bandit.stats.keys())
    assert any(":" in k for k in keys)             # ctx carries a load band
