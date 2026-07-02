"""social-autopilot × Context Runtime — offline Postiz benchmark.

Simulates 72 posting decisions where Context Runtime picks a channel/timing/content
strategy per goal bucket, learns from an engagement proxy, and beats a fixed baseline.
Run with:

    PYTHONPATH=. python examples/social_autopilot.py
"""
from __future__ import annotations

from context_runtime.integrations.social_autopilot import (
    DEFAULT_STRATEGIES,
    SocialAutopilotTenant,
    SocialStrategy,
    reward_from_engagement,
)

ROUNDS = 72
BASELINE_STRATEGY = DEFAULT_STRATEGIES[1]  # linkedin midday carousel announcement

# Stream of goals with their latent best strategy (what truly resonates).
GOAL_STREAM = [
    ("Announce the AI release ship", "linkedin:midday:carousel:announcement"),
    ("Share a customer community spotlight", "instagram:evening:reel:community"),
    ("Thread on our automation metrics", "twitter:morning:thread:educational"),
    ("Promote the limited-time upgrade offer", "tiktok:evening:short:community"),
    ("Deep-dive tutorial on orchestrations", "linkedin:morning:article:educational"),
    ("Event reminder for tomorrow's webinar", "twitter:midday:thread:announcement"),
]

STRATEGY_BY_KEY = {s.key: s for s in DEFAULT_STRATEGIES}


def _rand(state: list[int]) -> float:
    x = state[0]
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    state[0] = x & 0xFFFFFFFF
    return state[0] / 0x100000000


def simulate_engagement(chosen: SocialStrategy, latent: SocialStrategy, state: list[int]) -> float:
    base = 4.5
    if chosen.key == latent.key:
        base = 9.6
    else:
        if chosen.channel == latent.channel:
            base += 1.6
        if chosen.timing == latent.timing:
            base += 0.9
        if chosen.format == latent.format:
            base += 1.2
        if chosen.tone == latent.tone:
            base += 0.7
    noise = (_rand(state) - 0.5) * 0.8
    return max(0.0, base + noise)


def run(rounds: int = ROUNDS) -> None:
    tenant = SocialAutopilotTenant()
    rng = [0xA5F1523C]
    learned_rewards: list[float] = []
    baseline_rewards: list[float] = []

    print("First few campaign decisions (strategy → engagement → reward):\n")

    for i in range(rounds):
        goal_text, latent_key = GOAL_STREAM[i % len(GOAL_STREAM)]
        latent_strategy = STRATEGY_BY_KEY[latent_key]
        strategy = tenant.choose(goal_text)
        engagement = simulate_engagement(strategy, latent_strategy, rng)
        reward = tenant.record_outcome(goal_text, engagement)
        learned_rewards.append(reward)

        baseline_engagement = simulate_engagement(BASELINE_STRATEGY, latent_strategy, rng)
        baseline_rewards.append(reward_from_engagement(baseline_engagement, BASELINE_STRATEGY))

        if i < 6:
            print(f"  {goal_text[:46]:<46} → {strategy.key:<38} engagement={engagement:4.1f} reward={reward:5.2f}")
        elif i == 6:
            print("  ...")

    avg_learned = sum(learned_rewards[-18:]) / 18
    avg_baseline = sum(baseline_rewards[-18:]) / 18

    print("\nreward = engagement − posting cost\n")
    print(f"Context Runtime (learned): {avg_learned:.3f}")
    print(f"baseline ({BASELINE_STRATEGY.key}): {avg_baseline:.3f}")

    print("\nlearned policy per bucket:\n")
    policy = tenant.policy()
    if not policy:
        print("  (unlearned)")
    else:
        for bucket in sorted(policy):
            print(f"  {bucket:<10} → {policy[bucket]}")


if __name__ == "__main__":
    run()
