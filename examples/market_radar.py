"""market-radar × Context Runtime — offline competitive-intel benchmark.

Simulates 72 intel questions where Context Runtime picks which competitor watches to
sweep per question bucket, learns from a signal-quality proxy minus scrape cost, and
beats a fixed full-sweep baseline. Run with:

    PYTHONPATH=. python examples/market_radar.py
"""
from __future__ import annotations

from statistics import mean

from context_runtime.integrations.market_radar import (
    _BUCKET_KEY,
    DEFAULT_RADAR,
    MarketRadarTenant,
    RadarArm,
    reward_from_news,
)

ROUNDS = 72
BASELINE_ARM = DEFAULT_RADAR[-1]  # full sweep — always catches it, always most expensive

INTEL_STREAM = [
    ("Did a competitor change pricing?", "pricing"),
    ("Any new product release from rivals?", "product"),
    ("Is a competitor hiring aggressively?", "hiring"),
    ("Breaking news about the market?", "news"),
    ("New pricing plan on their site?", "pricing"),
    ("Feature shipped in their changelog?", "product"),
    ("Headcount growth on their careers page?", "hiring"),
    ("Blog post signalling a strategy shift?", "news"),
]

_STATE = [0x1CEB00DA]


def _rand() -> float:
    x = _STATE[0]
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    _STATE[0] = x & 0xFFFFFFFF
    return _STATE[0] / 0x100000000


def _value(chosen: RadarArm, bucket: str) -> float:
    """Signal quality: high only when the sweep includes the watch that holds the answer."""
    decisive = _BUCKET_KEY[bucket]
    base = 6.5 if getattr(chosen, decisive) else 2.0
    noise = (_rand() - 0.5) * 0.6
    return max(0.0, base + noise)


def run(rounds: int = ROUNDS) -> None:
    tenant = MarketRadarTenant(epsilon=0.15)
    learned_rewards: list[float] = []
    baseline_rewards: list[float] = []

    print("First few intel decisions (watch set → value → reward):\n")

    for i in range(rounds):
        question, bucket = INTEL_STREAM[i % len(INTEL_STREAM)]
        chosen = tenant.choose(question, bucket=bucket)
        value = _value(chosen, bucket)
        reward = tenant.record_outcome(question, value)
        learned_rewards.append(reward)

        baseline_value = _value(BASELINE_ARM, bucket)
        baseline_rewards.append(reward_from_news(baseline_value, BASELINE_ARM))

        if i < 6:
            print(f"  {question[:40]:<40} → {chosen.key:<22} value={value:4.1f} reward={reward:5.2f}")
        elif i == 6:
            print("  ...")

    avg_learned = mean(learned_rewards[-18:])
    avg_baseline = mean(baseline_rewards[-18:])

    print("\nreward = intel signal quality − scrape cost\n")
    print(f"Context Runtime (learned): {avg_learned:.3f}")
    print(f"baseline ({BASELINE_ARM.key}): {avg_baseline:.3f}")

    print("\nlearned policy per bucket:\n")
    policy = tenant.policy()
    if not policy:
        print("  (unlearned)")
    else:
        for bucket_key in sorted(policy):
            print(f"  {bucket_key:<10} → {policy[bucket_key]}")


if __name__ == "__main__":
    run()
