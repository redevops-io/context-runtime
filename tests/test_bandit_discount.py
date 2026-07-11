"""Discounted (recency-weighted) bandit updates — the non-stationarity property (Whitepaper v3):
a constant step size fades stale evidence so the learned value tracks a drifting reward."""
from __future__ import annotations

from context_runtime.integrations.bandit import EpsilonGreedyBandit


class _Arm:
    def __init__(self, key):
        self.key = key


def _after_flip(discount: float) -> float:
    b = EpsilonGreedyBandit(arms=(_Arm("x"),), discount=discount)
    for _ in range(50):
        b.update("c", _Arm("x"), 0.9)      # long history at 0.9
    for _ in range(15):
        b.update("c", _Arm("x"), 0.1)      # world flips to 0.1
    return b.value("c", "x")[1]


def test_sample_average_is_the_default_and_barely_moves():
    # 0 discount = 1/n sample average: 50 old samples dominate 15 new ones
    assert _after_flip(0.0) > 0.6


def test_discount_tracks_the_new_regime():
    # constant step size weights recent rewards → tracks the flip toward 0.1
    assert _after_flip(0.2) < 0.2


def test_discount_beats_sample_average_under_drift():
    assert _after_flip(0.2) < _after_flip(0.0)
