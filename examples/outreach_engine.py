#!/usr/bin/env python3
"""outreach-engine — Context Runtime learns which outreach PLAY lands pilots per account signal.

The decision: for each account, which buying **signal** to lead with, which **channel**, and how
deep to **personalize** (template → company research → an EXPLAIN/redevops-rag *artifact teardown*).
The teardown converts technical buyers but is expensive, so it only pays on high-signal accounts —
the exact cost/quality trade the runtime is built for. Baseline = a fixed "spray" (cold email
template) on every account; learned = the bandit adapting per account bucket.

Seeded/deterministic. Run:  PYTHONPATH=. python examples/outreach_engine.py   (exits 0)
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from context_runtime.integrations.outreach_engine import (
    DEFAULT_PLAYS, OutreachEngineTenant, OutreachPlay, reward_from_pilot,
)

# account buckets and the real signal each one should be led with
BUCKETS = ("funding", "hiring", "tech_pain", "leadership", "cold")
BUCKET_SIGNAL = {"funding": "funding", "hiring": "hiring", "tech_pain": "tech_pain",
                 "leadership": "leadership", "cold": "cold"}
# how convertible each account bucket is at all (signal beats spray: cold barely converts)
CEILING = {"cold": 0.5, "leadership": 0.85, "hiring": 0.95, "funding": 1.1, "tech_pain": 1.2}
DEPTH_V = {"template": 1.0, "company": 2.6, "artifact": 4.2}
CHANNEL_V = {"email": 1.0, "linkedin": 1.0, "multi": 1.15, "video": 1.3}


def _rng(seed: int):
    s = {"x": (seed * 2654435761 + 12345) & 0xFFFFFFFF}

    def nxt() -> float:
        s["x"] = (1103515245 * s["x"] + 12345) & 0x7FFFFFFF
        return s["x"] / 0x7FFFFFFF
    return nxt


def pilot_value(play: OutreachPlay, bucket: str, noise: float) -> float:
    """Simulated outcome value (reply → meeting → pilot, weighted). High only when effort meets a
    real, matching signal; artifact effort on a cold account is wasted."""
    match = 1.0 if play.signal == BUCKET_SIGNAL[bucket] else 0.55
    v = DEPTH_V[play.depth] * match * CEILING[bucket] * CHANNEL_V[play.channel]
    return v * (0.9 + 0.2 * noise)


def run(seed: int, learned: bool) -> float:
    rng = _rng(seed)
    baseline = OutreachPlay("cold", "email", "template")  # the spray
    tenant = OutreachEngineTenant(epsilon=0.12) if learned else None
    rewards = []
    n = 600
    for i in range(n):
        bucket = BUCKETS[int(rng() * len(BUCKETS)) % len(BUCKETS)]
        acct = f"{bucket} account {i}"
        if learned:
            play = tenant.choose(acct, bucket=bucket)
            r = tenant.record_outcome(acct, pilot_value(play, bucket, rng()))
        else:
            r = reward_from_pilot(pilot_value(baseline, bucket, rng()), baseline)
        if i >= n // 2:                       # measure after the policy has converged
            rewards.append(r)
    return sum(rewards) / len(rewards) if rewards else 0.0


def main() -> int:
    seeds = range(1, 21)
    learned = sum(run(s, True) for s in seeds) / len(seeds)
    baseline = sum(run(s, False) for s in seeds) / len(seeds)
    print("=" * 64)
    print("outreach-engine — learned outreach play vs fixed spray (cold email template)")
    print("=" * 64)
    print("reward = pilot-conversion value (reply→meeting→pilot) − outreach effort cost,")
    print("averaged over the 2nd half of a 600-account stream, 20 seeds.\n")
    print(f"  learned (per-signal play)  {learned:6.3f}")
    print(f"  baseline (fixed spray)     {baseline:6.3f}   ({learned - baseline:+.3f})\n")
    print("The runtime learns to spend the expensive artifact teardown on high-signal accounts")
    print("(funded / hiring / tech-pain) and stay cheap on cold ones — signal beats spray.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
