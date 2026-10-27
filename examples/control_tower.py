# Copyright (C) 2024 Context Runtime
#
# Licensed under AGPL-3.0-or-later with the Commons Clause (see LICENSE.md).
# Distributed WITHOUT ANY WARRANTY; see the GNU Affero General Public License for details.
"""control-tower × Context Runtime — offline Metabase refresh benchmark."""
from __future__ import annotations

from context_runtime.integrations.control_tower import (
    ControlTowerTenant,
    ControlTowerArm,
    DEFAULT_TOWER,
    control_tower_bucket,
    reward_from_cash,
    reward_from_ops,
    reward_from_pipeline,
    reward_from_revenue,
)

ROUNDS = 72
BASELINE_ARM = DEFAULT_TOWER[0]

GOAL_STREAM = [
    ("How did ARR trend last week?", "revenue", 10.5),
    ("Are MQLs converting in the enterprise segment?", "pipeline", 9.8),
    ("Are on-call escalations down after the automation rollout?", "ops", 8.4),
    ("Do we still have 14 months runway?", "cash", 8.7),
]

BUCKET_REWARD = {
    "revenue": reward_from_revenue,
    "pipeline": reward_from_pipeline,
    "ops": reward_from_ops,
    "cash": reward_from_cash,
}


def _rand(state: list[int]) -> float:
    x = state[0]
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    state[0] = x & 0xFFFFFFFF
    return state[0] / 0x100000000


def latent_value(chosen: ControlTowerArm, bucket: str, state: list[int]) -> float:
    base = next(v for q, b, v in GOAL_STREAM if b == bucket)
    if chosen.key == LATENT_POLICY[bucket]:
        base *= 1.3
    else:
        base *= 0.8
    noise = (_rand(state) - 0.5) * 0.6
    return max(0.0, base + noise)


LATENT_POLICY = {
    "revenue": "daily_revenue_core",
    "pipeline": "growth_pipeline_full",
    "ops": "ops_latency_focus",
    "cash": "cashflow_variance",
}

ARM_BY_KEY = {arm.key: arm for arm in DEFAULT_TOWER}


def run(rounds: int = ROUNDS) -> None:
    tenant = ControlTowerTenant()
    rng = [0xA5F1523C]
    learned: list[float] = []
    baseline: list[float] = []

    print("First few dashboard refreshes (arm → value → reward):\n")

    for i in range(rounds):
        question, bucket, _ = GOAL_STREAM[i % len(GOAL_STREAM)]
        arm = tenant.choose(question)
        value = latent_value(arm, bucket, rng)
        reward = tenant.record_outcome(question, value)
        learned.append(reward)

        baseline_value = latent_value(BASELINE_ARM, bucket, rng)
        baseline.append(BUCKET_REWARD[bucket](baseline_value, BASELINE_ARM))

        if i < 6:
            print(f"  {question[:46]:<46} → {arm.key:<24} value={value:4.1f} reward={reward:5.2f}")
        elif i == 6:
            print("  ...")

    avg_learned = sum(learned[-18:]) / 18
    avg_baseline = sum(baseline[-18:]) / 18

    print("\nreward = dashboard delta − compute minutes\n")
    print(f"Context Runtime (learned): {avg_learned:.3f}")
    print(f"baseline ({BASELINE_ARM.key}): {avg_baseline:.3f}")

    print("\nlearned policy per bucket:\n")
    policy = tenant.policy()
    if not policy:
        print("  (unlearned)")
    else:
        for bucket in sorted(policy):
            print(f"  {bucket:<10} → {policy[bucket]}")


if __name__ == "__main__":
    run()
