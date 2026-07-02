"""agentic-support × Context Runtime — offline support-context benchmark.

Simulates 72 support tickets where Context Runtime picks a context bundle per ticket
bucket, learns from a resolution-quality proxy minus retrieval cost, and beats a fixed
full-context baseline. Run with:

    PYTHONPATH=. python examples/agentic_support.py
"""
from __future__ import annotations

from statistics import mean

from context_runtime.integrations.agentic_support import (
    DECISIVE_BY_BUCKET,
    DEFAULT_SUPPORT,
    AgenticSupportTenant,
    SupportContextBundle,
    reward_from_resolution,
)

ROUNDS = 72
BASELINE_BUNDLE = DEFAULT_SUPPORT[0]  # full_context — always correct, always most expensive

TICKET_STREAM = [
    ("How do I configure SSO for my team?", "howto"),
    ("Export button throws a 500 error", "bug"),
    ("Why was I charged twice this month?", "billing_q"),
    ("Dashboard is down and unavailable", "outage"),
    ("Where do I enable webhooks?", "howto"),
    ("Sync crashes on large imports", "bug"),
    ("Refund for the duplicate invoice", "billing_q"),
    ("API returning 500, service incident?", "outage"),
]

_STATE = [0xC0FFEE]


def _rand() -> float:
    x = _STATE[0]
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    _STATE[0] = x & 0xFFFFFFFF
    return _STATE[0] / 0x100000000


def _value(chosen: SupportContextBundle, bucket: str) -> float:
    """Resolution quality: high only when the chosen bundle holds the bucket's decisive source."""
    decisive = DECISIVE_BY_BUCKET[bucket]
    base = 6.5 if getattr(chosen, decisive) else 2.0
    noise = (_rand() - 0.5) * 0.6
    return max(0.0, base + noise)


def run(rounds: int = ROUNDS) -> None:
    tenant = AgenticSupportTenant(epsilon=0.15)
    learned_rewards: list[float] = []
    baseline_rewards: list[float] = []

    print("First few ticket decisions (bundle → value → reward):\n")

    for i in range(rounds):
        ticket, bucket = TICKET_STREAM[i % len(TICKET_STREAM)]
        chosen = tenant.choose(ticket, bucket=bucket)
        value = _value(chosen, bucket)
        reward = tenant.record_outcome(ticket, value)
        learned_rewards.append(reward)

        baseline_value = _value(BASELINE_BUNDLE, bucket)
        baseline_rewards.append(reward_from_resolution(baseline_value, BASELINE_BUNDLE))

        if i < 6:
            print(f"  {ticket[:44]:<44} → {chosen.key:<18} value={value:4.1f} reward={reward:5.2f}")
        elif i == 6:
            print("  ...")

    avg_learned = mean(learned_rewards[-18:])
    avg_baseline = mean(baseline_rewards[-18:])

    print("\nreward = resolution quality − context retrieval cost\n")
    print(f"Context Runtime (learned): {avg_learned:.3f}")
    print(f"baseline ({BASELINE_BUNDLE.key}): {avg_baseline:.3f}")

    print("\nlearned policy per bucket:\n")
    policy = tenant.policy()
    if not policy:
        print("  (unlearned)")
    else:
        for bucket_key in sorted(policy):
            print(f"  {bucket_key:<10} → {policy[bucket_key]}")


if __name__ == "__main__":
    run()
