"""agentic-compliance × Context Runtime — offline evidence-selection benchmark.

Simulates 72 compliance findings where Context Runtime picks an evidence bundle per
finding bucket, learns from a remediation-correctness proxy minus evidence cost, and
beats a fixed full-evidence baseline. Run with:

    PYTHONPATH=. python examples/agentic_compliance.py
"""
from __future__ import annotations

from statistics import mean

from context_runtime.integrations.agentic_compliance import (
    DECISIVE_BY_BUCKET,
    DEFAULT_COMPLIANCE,
    AgenticComplianceTenant,
    ComplianceEvidenceBundle,
    reward_from_remediation,
)

ROUNDS = 72
BASELINE_BUNDLE = DEFAULT_COMPLIANCE[0]  # full_evidence — always correct, always most expensive

FINDING_STREAM = [
    ("Weak password policy on the login service", "access"),
    ("TLS cipher suite is deprecated", "crypto"),
    ("Audit logging disabled on journald", "logging"),
    ("Outdated kernel, CVE pending patch", "patch"),
    ("Privileged sudo access unrestricted", "access"),
    ("SSH key encryption below policy", "crypto"),
    ("rsyslog forwarding not configured", "logging"),
    ("Package versions behind on updates", "patch"),
]

_STATE = [0x5CA1AB1E]


def _rand() -> float:
    x = _STATE[0]
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    _STATE[0] = x & 0xFFFFFFFF
    return _STATE[0] / 0x100000000


def _value(chosen: ComplianceEvidenceBundle, bucket: str) -> float:
    decisive = DECISIVE_BY_BUCKET[bucket]
    base = 6.5 if getattr(chosen, decisive) else 2.0
    noise = (_rand() - 0.5) * 0.6
    return max(0.0, base + noise)


def run(rounds: int = ROUNDS) -> None:
    tenant = AgenticComplianceTenant(epsilon=0.15)
    learned_rewards: list[float] = []
    baseline_rewards: list[float] = []

    print("First few finding decisions (bundle → value → reward):\n")

    for i in range(rounds):
        finding, bucket = FINDING_STREAM[i % len(FINDING_STREAM)]
        chosen = tenant.choose(finding, bucket=bucket)
        value = _value(chosen, bucket)
        reward = tenant.record_outcome(finding, value)
        learned_rewards.append(reward)

        baseline_value = _value(BASELINE_BUNDLE, bucket)
        baseline_rewards.append(reward_from_remediation(baseline_value, BASELINE_BUNDLE))

        if i < 6:
            print(f"  {finding[:44]:<44} → {chosen.key:<16} value={value:4.1f} reward={reward:5.2f}")
        elif i == 6:
            print("  ...")

    avg_learned = mean(learned_rewards[-18:])
    avg_baseline = mean(baseline_rewards[-18:])

    print("\nreward = remediation correctness − evidence pull cost\n")
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
