#!/usr/bin/env python3
"""Online (Gen-4) vs static planner under drift — "remaining correct in a non-stationary world".

The static planner (v2 cost model) commits to the plan it estimated best and never revisits it. The
online planner (BanditOptimizer) learns from measured reward. We run a seeded environment where plan A
is best — until it drifts and C becomes best (a model upgrade, a corpus shift). We compare three:

  • static           — always the highest-prior plan (A); never adapts.
  • online, plain     — sample-average learning; 200 stale samples of A drown the new signal.
  • online, discount  — recency-weighted learning (the whitepaper's non-stationarity property);
                        stale evidence fades, so the planner tracks the drift to C.

    PYTHONPATH=. python examples/online_vs_static_bench.py
"""
from __future__ import annotations

from context_runtime.optimizer.online import BanditOptimizer
from context_runtime.types import Candidate, Goal, PlanScore, StepSpec

CTX, GOAL = "synthesis", Goal(text="q")
STEPS, DRIFT = 400, 200
PRIOR = {"A": 0.80, "C": 0.20}               # cost model's fixed estimate (calibrated to t<DRIFT)
PRE = {"A": 0.80, "C": 0.20}                 # true reward before drift
POST = {"A": 0.20, "C": 0.80}                # true reward after drift — C now wins


def cand(arm):
    return Candidate(steps=(StepSpec(type="retrieve", params={"method": arm}),), model_tier="cheap")


def reward(arm, step):
    return (PRE if step < DRIFT else POST)[arm]


def scored():
    return [(cand(a), PlanScore(total=PRIOR[a], feasible=True)) for a in PRIOR]


def run_static():
    return [reward("A", t) for t in range(STEPS)]     # cost model never updates → always A


def run_online(discount):
    opt = BanditOptimizer(None, epsilon=0.2, discount=discount)
    rewards = []
    for t in range(STEPS):
        plan = opt.select(scored(), GOAL, context=CTX)
        arm = next(s.params["method"] for s in plan.chosen.steps if s.type == "retrieve")
        r = reward(arm, t)
        opt.learn_from_plan(plan, r)          # learn keyed by the plan's own arm (method:tier)
        rewards.append(r)
    return rewards


def avg(xs):
    return sum(xs) / len(xs)


def main():
    s, plain, disc = run_static(), run_online(0.0), run_online(0.2)
    oracle = avg([max((PRE if i < DRIFT else POST).values()) for i in range(DRIFT, STEPS)])
    print("Online (Gen-4 bandit) vs static planner — best plan drifts A→C at t=200\n")
    print(f"| {'window':24} | static | online plain | online discounted |")
    print(f"| {'-'*24} | ------ | ------------ | ----------------- |")
    for name, lo, hi in [("pre-drift  (t<200)", 0, DRIFT), ("post-drift (t>=200)", DRIFT, STEPS),
                         ("overall", 0, STEPS)]:
        print(f"| {name:24} | {avg(s[lo:hi]):.3f}  | {avg(plain[lo:hi]):.3f}        "
              f"| {avg(disc[lo:hi]):.3f}             |")
    print(f"\nPost-drift oracle (best possible) = {oracle:.2f}. Static is pinned to the now-stale A "
          f"({avg(s[DRIFT:]):.2f}). Plain online adapts poorly ({avg(plain[DRIFT:]):.2f}) — the sample "
          f"average is dominated by 200 pre-drift observations. With discounting, stale evidence fades "
          f"and the planner tracks the drift to C ({avg(disc[DRIFT:]):.2f}).")


if __name__ == "__main__":
    main()
