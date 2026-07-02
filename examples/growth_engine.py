# SPDX-License-Identifier: AGPL-3.0-or-later
"""growth-engine × Context Runtime — offline attribution benchmark."""
from __future__ import annotations

from statistics import mean

from context_runtime.integrations.growth_engine import (
    DEFAULT_GROWTH,
    GrowthEngineTenant,
    reward_from_direct,
    reward_from_organic,
    reward_from_paid,
    reward_from_referral,
)

ROUNDS = 72
EPSILON = 0.15
BASELINE_ARM = DEFAULT_GROWTH[2]


def deterministic_rng(seed: int) -> callable[[], float]:
    state = [seed & 0xFFFFFFFF]

    def _next() -> float:
        x = state[0]
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        state[0] = x & 0xFFFFFFFF
        return state[0] / 0x100000000

    return _next


RNG = deterministic_rng(0x715394A3)


def latent_reward(bucket: str, arm_key: str) -> float:
    base = {
        "paid": 10.0,
        "organic": 8.5,
        "referral": 9.0,
        "direct": 7.5,
    }[bucket]
    if bucket == "paid" and arm_key.startswith("24h:utm"):
        base += 3.2
    if bucket == "organic" and arm_key.startswith("7d:referrer"):
        base += 2.6
    if bucket == "referral" and arm_key.startswith("30d:first_touch"):
        base += 3.4
    if bucket == "direct" and arm_key.startswith("session:session"):
        base += 2.3
    return base + (RNG() - 0.5) * 1.4


GOAL_STREAM = [
    ("How did the paid AI webinar campaign perform yesterday?", "paid"),
    ("Attribute conversions from the blog article launch", "organic"),
    ("Measure partner referrals this week", "referral"),
    ("Direct traffic uplift after homepage refresh", "direct"),
]

REWARD_FN = {
    "paid": reward_from_paid,
    "organic": reward_from_organic,
    "referral": reward_from_referral,
    "direct": reward_from_direct,
}


def run(rounds: int = ROUNDS) -> None:
    tenant = GrowthEngineTenant(epsilon=EPSILON)
    learned_rewards: list[float] = []
    baseline_rewards: list[float] = []

    print("First few attribution decisions (arm → value → reward):\n")
    for i in range(rounds):
        question, bucket = GOAL_STREAM[i % len(GOAL_STREAM)]
        arm = tenant.choose(question)
        value = latent_reward(bucket, arm.key)
        reward = REWARD_FN[bucket](value, arm)
        tenant_reward = tenant.record_outcome(question, value)
        learned_rewards.append(tenant_reward)

        baseline_value = latent_reward(bucket, BASELINE_ARM.key)
        baseline_reward = REWARD_FN[bucket](baseline_value, BASELINE_ARM)
        baseline_rewards.append(baseline_reward)

        if i < 6:
            print(f"  {question[:46]:<46} → {arm.key:<28} value={value:4.1f} reward={tenant_reward:5.2f}")
        elif i == 6:
            print("  ...")

    avg_learned = mean(learned_rewards[-18:])
    avg_baseline = mean(baseline_rewards[-18:])

    print("\nreward = attributed value − attribution cost\n")
    print(f"Context Runtime (learned): {avg_learned:.3f}")
    print(f"baseline ({BASELINE_ARM.key}): {avg_baseline:.3f}")

    policy = tenant.policy()
    print("\nlearned policy per bucket:\n")
    for bucket, arm_key in sorted(policy.items()):
        print(f"  {bucket:<8} → {arm_key}")

    assert avg_learned >= avg_baseline, "Learned policy should outperform baseline"


if __name__ == "__main__":
    run()
