"""Resolution Closure tenant — offline learning benchmark.

    python examples/resolution_closure.py

Context Runtime learns, per capability, which composition recovers the dependency closure best under budget —
and routes the disputed tail to review. The per-(capability, composition) closure-recall surface is the FROZEN
Gate-2 measurement from redevops-benchmarks (the offline stand-in for a live retrieval eval): code rewards
structural promotion (α≳0.4 → 0.960), legal is content-saturated (α=0 → 0.589, structure *hurts*). No single
global composition wins both — so a per-capability bandit should beat any fixed one. The reward also charges a
small cost for structural promotion, so the learner prefers the *cheapest* α that reaches the closure.
"""
from __future__ import annotations

from context_runtime.integrations.resolution_closure import (
    CompositionArm,
    ResolutionClosureTenant,
    _closure_bandit,
    reward_closure,
)

# Frozen Gate-2 closure recall per (capability, composition arm key) — measured, not invented.
CELL: dict[str, dict[str, float]] = {
    "code":  {"content_only": 0.727, "structure_first": 0.693,
              "content_led:0": 0.860, "content_led:0.4": 0.960, "content_led:0.6": 0.960},
    "legal": {"content_only": 0.441, "structure_first": 0.138,
              "content_led:0": 0.589, "content_led:0.4": 0.582, "content_led:0.6": 0.582},
}

# Small synthetic per-task payloads (exercise the composition + ensemble seams; recall comes from CELL).
TASKS = [
    {"request": "code: signature change to Mission.run", "capability": "code",
     "seed": ["run"], "content_score": {"run": 0.5, "handle": 1.0, "dispatch": 0.7},
     "struct_score": {"run": 1.0, "dispatch": 0.9, "approve": 0.8}, "budget": 300,
     "tokens": {"run": 100, "handle": 100, "dispatch": 100, "approve": 100}},
    {"request": "code: add field to ExecutionIntent", "capability": "code",
     "seed": ["intent"], "content_score": {"intent": 0.4, "build": 0.9},
     "struct_score": {"intent": 1.0, "build": 0.7, "select": 0.9}, "budget": 200,
     "tokens": {"intent": 50, "build": 50, "select": 50}},
    {"request": "legal: NDA confidentiality carve-out", "capability": "legal",
     "seed": ["p12"], "content_score": {"p12": 0.9, "p13": 0.8, "p33": 0.3},
     "struct_score": {"p12": 1.0, "p33": 0.6}, "budget": 200,
     "tokens": {"p12": 60, "p13": 60, "p33": 60, "p36": 60},
     "votes": {"p13": {"local": True, "kimi": True, "openai": False},   # a genuine split → routed to review
               "p12": {"local": True, "kimi": True, "openai": True}}},
    {"request": "legal: term/termination survival", "capability": "legal",
     "seed": ["p82"], "content_score": {"p82": 1.0, "p93": 0.7, "p94": 0.4},
     "struct_score": {"p82": 1.0, "p94": 0.8}, "budget": 180,
     "tokens": {"p82": 55, "p93": 55, "p94": 55}},
]


def _xorshift(seed: int):
    s = seed & 0xFFFFFFFF
    while True:
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        yield (s & 0xFFFFFFFF) / 0xFFFFFFFF


def _recall(capability: str, arm_key: str, rng) -> float:
    base = CELL[capability][arm_key]
    return max(0.0, min(1.0, base + (next(rng) - 0.5) * 0.04))   # tiny deterministic noise


def run(rounds: int = 96) -> None:
    tenant = ResolutionClosureTenant(bandit=_closure_bandit(0.12))
    baseline = CompositionArm("content_led", 0.6)   # the naive "always structure-promote" global policy
    rng = _xorshift(0xC105)
    learned, base, routed_total = [], [], 0

    print("reward = closure recall − structural-promotion cost (higher = fuller closure, cheaper)\n")
    print(f"  {'#':>3} {'capability':10} {'chosen arm':18} {'recall':>6} {'reward':>7} {'status':14} routed")
    for i in range(rounds):
        task = TASKS[i % len(TASKS)]
        res = tenant.resolve(task)
        rec = _recall(task["capability"], res.arm.key, rng)
        r = tenant.record_outcome(res.request, rec)
        learned.append(r)
        base.append(reward_closure(_recall(task["capability"], baseline.key, rng), baseline))
        routed_total += len(res.routed)
        if i < 6:
            print(f"  {i:>3} {res.capability:10} {res.arm.key:18} {rec:6.3f} {r:7.3f} {res.status:14} "
                  f"{','.join(res.routed) or '—'}")

    w = 24
    lw, bw = sum(learned[-w:]) / w, sum(base[-w:]) / w
    print(f"\n── mean reward over last {w} rounds ──")
    print(f"  Context Runtime (learned per-capability composition): {lw:.3f}")
    print(f"  baseline (fixed content_led:0.6 everywhere):          {bw:.3f}")
    print(f"  delta: {lw - bw:+.3f}   (learned should win — it avoids legal's structural-promotion penalty)")

    print("\n── learned composition policy per capability ──")
    for cap, arm_key in sorted(tenant.policy().items()):
        print(f"  {cap:8} → {arm_key}")

    print(f"\n── Gate-4 ensemble routing ──")
    print(f"  disputed dependencies routed to review over {rounds} rounds: {routed_total} "
          f"(these are the semantic-uncertainty tail, not auto-resolved)")

    stats = tenant.runtime.estimator.statistics()
    print(f"\n── cost-model calibration after {rounds} observed resolutions ──")
    for fs in list(stats.fields)[:4]:
        print(f"  {fs.field:18} samples={fs.sample_count} calibration={fs.calibration:.3f}")


if __name__ == "__main__":
    run()
