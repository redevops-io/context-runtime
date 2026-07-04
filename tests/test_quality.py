"""Quality ledger — quality tracked apart from cost; explore-then-exploit routing that prefers a
genuinely better choice at equal cost (the 'provider-agnostic ≠ provider-equal' point)."""
from __future__ import annotations

from context_runtime.quality import QualityLedger, QualityStat


def test_running_means_and_blended():
    led = QualityLedger(cost_weight=0.2)
    for q in (0.8, 1.0):
        led.observe("ctx", "A", q, 0.5)
    s = led.stat("ctx", "A")
    assert s.n == 2 and abs(s.quality - 0.9) < 1e-6 and abs(s.cost - 0.5) < 1e-6
    assert abs(s.blended(0.2) - (0.9 - 0.2 * 0.5)) < 1e-6
    assert led.stat("ctx", "never") is None


def test_route_explores_then_exploits_quality_over_cost():
    # A: best quality, cheap. B: mediocre. C: high quality but expensive.
    led = QualityLedger(cost_weight=0.2)
    choices = ["A", "B", "C"]

    # ── exploration phase: every choice is visited to min_samples before any exploitation ──
    seen = set()
    for _ in range(9):
        c = led.route("ctx", choices, min_samples=3)
        seen.add(c)
        q = {"A": 0.9, "B": 0.5, "C": 0.9}[c]
        cost = {"A": 0.1, "B": 0.1, "C": 0.9}[c]
        led.observe("ctx", c, q, cost)
    assert seen == {"A", "B", "C"}                       # explored all before exploiting
    assert all(led.stat("ctx", c).n >= 3 for c in choices)

    # ── exploitation: A wins (equal-cost-to-B but higher quality; beats C on blended cost) ──
    assert led.route("ctx", choices, min_samples=3) == "A"
    assert led.best("ctx", choices) == "A"
    # C is genuinely high quality but its cost sinks the blend below A
    ranked = led.stats("ctx", choices)
    assert ranked[0].choice == "A"
    assert next(s for s in ranked if s.choice == "C").quality == 0.9   # quality preserved, not hidden by cost


def test_higher_quality_at_equal_cost_is_preferred():
    """The core claim: at equal cost, the better-quality choice wins — a scalar reward that folds
    cost in can miss this; the ledger does not."""
    led = QualityLedger(cost_weight=0.2)
    for _ in range(5):
        led.observe("q", "gemini", 0.70, 0.3)
        led.observe("q", "claude", 0.85, 0.3)   # same cost, higher quality
    assert led.best("q", ["gemini", "claude"]) == "claude"


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "q.json"
    led = QualityLedger(path=str(p))
    led.observe("c", "A", 0.8, 0.2)
    led.observe("c", "A", 0.9, 0.2)
    again = QualityLedger(path=str(p))          # reload
    s = again.stat("c", "A")
    assert s is not None and s.n == 2 and abs(s.quality - 0.85) < 1e-6


# ── edge cases / invariants added for the v2 release audit ──

def test_observe_clamps_out_of_range():
    led = QualityLedger()
    led.observe("c", "A", quality=1.7, cost=-0.5)
    s = led.stat("c", "A")
    assert s.quality == 1.0 and s.cost == 0.0          # clamped into [0,1] (EXPLAIN/routing depend on it)


def test_route_empty_choices_returns_none():
    assert QualityLedger().route("ctx", []) is None     # cold-start fallback to the bandit


def test_best_respects_min_samples():
    led = QualityLedger()
    led.observe("c", "A", 0.9, 0.1)
    led.observe("c", "A", 0.9, 0.1)
    assert led.best("c", ["A"], min_samples=5) is None  # under-sampled → not trusted
    assert led.best("c", ["A"], min_samples=1) == "A"


def test_ledger_tolerates_corrupt_persist_file(tmp_path):
    p = tmp_path / "q.json"
    p.write_text("{ not valid json", encoding="utf-8")
    led = QualityLedger(path=p)                          # must not raise on load
    assert led.stat("c", "A") is None
    led.observe("c", "A", 0.8, 0.2)                      # still usable
    assert led.stat("c", "A").n == 1


def test_stats_sorted_by_blended_and_cost_weight_override():
    led = QualityLedger()
    led.observe("c", "A", 0.9, 0.9)                      # high quality, expensive
    led.observe("c", "B", 0.8, 0.1)                      # slightly lower quality, cheap
    assert [s.choice for s in led.stats("c", cost_weight=0.0)] == ["A", "B"]   # pure quality: A first
    assert led.stats("c", cost_weight=1.0)[0].choice == "B"                    # heavy cost weight: B wins
