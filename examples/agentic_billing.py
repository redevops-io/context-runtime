"""agentic-billing × Context Runtime — offline collections benchmark.

Simulates 72 credit/collections decisions where Context Runtime picks a signal bundle
per account bucket, learns from a repayment value proxy, and beats a fixed baseline.
Run with:

    PYTHONPATH=. python examples/agentic_billing.py
"""
from __future__ import annotations

from statistics import mean

from context_runtime.integrations.agentic_billing import (
    DEFAULT_BILLING,
    AgenticBillingTenant,
    BillingSignalBundle,
    agentic_billing_bucket,
    reward_from_value,
)

ROUNDS = 72
BASELINE_BUNDLE = DEFAULT_BILLING[0]

ACCOUNT_STREAM = [
    ("ACME Corp overdue 45 days", "delinquent", "usage_dunning"),
    ("Globex subscription at churn risk", "at_risk", "usage_history"),
    ("Initech paid early last quarter", "healthy", "usage_invoice"),
    ("Soylent renewal in warning status", "at_risk", "usage_history"),
    ("Umbrella invoices past due", "delinquent", "usage_dunning"),
    ("Stark Industries engagement steady", "healthy", "usage_invoice"),
]

BUNDLES_BY_KEY = {b.key: b for b in DEFAULT_BILLING}

_STATE = [0xDEADBEEF]


def _rand() -> float:
    x = _STATE[0]
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    _STATE[0] = x & 0xFFFFFFFF
    return _STATE[0] / 0x100000000


def _value(chosen: BillingSignalBundle, latent: BillingSignalBundle) -> float:
    base = 2.0
    decisive = {
        "delinquent": "include_dunning",
        "at_risk": "include_payment_history",
        "healthy": "include_usage",
    }[agentic_billing_bucket(latent.key)]
    if getattr(chosen, decisive):
        base = 6.5
    else:
        if chosen.include_usage == latent.include_usage:
            base += 1.0
        if chosen.include_invoice == latent.include_invoice:
            base += 0.8
        if chosen.include_dunning == latent.include_dunning:
            base += 0.9
        if chosen.include_payment_history == latent.include_payment_history:
            base += 0.7
    noise = (_rand() - 0.5) * 0.6
    return max(0.0, base + noise)


def run(rounds: int = ROUNDS) -> None:
    tenant = AgenticBillingTenant(epsilon=0.15)
    learned_rewards: list[float] = []
    baseline_rewards: list[float] = []

    print("First few account decisions (bundle → value → reward):\n")

    for i in range(rounds):
        account, bucket, latent_key = ACCOUNT_STREAM[i % len(ACCOUNT_STREAM)]
        latent_bundle = BUNDLES_BY_KEY[latent_key]
        chosen = tenant.choose(account, bucket=bucket)
        value = _value(chosen, latent_bundle)
        reward = tenant.record_outcome(account, value)
        learned_rewards.append(reward)

        baseline_value = _value(BASELINE_BUNDLE, latent_bundle)
        baseline_rewards.append(reward_from_value(baseline_value, BASELINE_BUNDLE))

        if i < 6:
            print(f"  {account[:44]:<44} → {chosen.key:<14} value={value:4.1f} reward={reward:5.2f}")
        elif i == 6:
            print("  ...")

    avg_learned = mean(learned_rewards[-18:])
    avg_baseline = mean(baseline_rewards[-18:])

    print("\nreward = repayment value − data fetch cost\n")
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
