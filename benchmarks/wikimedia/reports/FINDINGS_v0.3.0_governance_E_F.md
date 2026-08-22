# Evidence-Native Benchmark — Arms E/F findings (v0.3.0 governance)

**The two deferred arms, now live.** Arms E (evidence/action trajectory governance) and F
(OBSERVE→ENFORCE) were absent in the v0.2.x run because the cross-series governance engine was not yet
reachable. They now run against the **shipped ReDevOps governance engine** (`agentic_os_enterprise.
governance`) plus the **v0.3.0 `PopulationRule`** (hard "≥K in a window" rate-cap mechanics), on real
strategywiki revert→protection trajectories. With them the full **A–G** benchmark passes.

- **Date:** 2026-08-22
- **Engine:** `agentic_os_enterprise.governance` — `CrossSeriesRule` + `OBSERVE/ENFORCE` (shipped) +
  **`PopulationRule`** (v0.3.0, `agentic-os-enterprise` `v0.3.0` branch).
- **Corpus:** strategywiki 20260801 — 27 pages that received a real **protection** event and had ≥2
  reverts before it (positives), + 60 multi-revert pages never protected (negative controls).
- **Label caveat:** a later protection event is an external moderation label, **not proof of
  causality** — reported as precision/recall against that label with the false-positive rate explicit.

## Result: both arms PASS (full A–G exit gate PASSED, 3 reproducible runs)

### Arm E — evidence/action trajectory governance

| metric | value | meaning |
|---|---|---|
| protected pages / controls | 27 / 60 | positives / negatives |
| trajectory false-positive rate | **0.217** | revert-storm rule flags 22% of controls |
| per-event baseline false-positive rate | **1.0** | the naive "any revert flags the page" gate flags *every* control |
| cross-series findings on controls | **0** | HARD gate — no action series ⇒ no finding (negative by construction) |
| trajectory recall vs label | 0.333 | storms precede 1/3 of protections (reported honestly, not tuned) |
| cross-series findings on protected | 10 | revert→protection correlated + cited (both series) |

The headline is the **false-positive collapse**: the trajectory/population rule flags edit-war storms at
a 0.217 FPR where the per-event baseline flags **everything** (1.0). The cross-series rule fires on **no**
control page — it is negative by construction because a control has no action (protection) series.

### Arm F — OBSERVE→ENFORCE lifecycle

Over the 9 pages with a detected revert-storm, on the **same** rule and events:

| metric | value |
|---|---|
| detection identical (OBSERVE vs ENFORCE) | 9 / 9 |
| OBSERVE decision = ALLOW | 9 / 9 |
| ENFORCE decision = REQUIRE_REVIEW | 9 / 9 |
| negative controls clean (wrong key / outside window / below threshold) | ✅ all 0 |

Promotion changes the **disposition**, never the **detection** — the invariant the plan requires.

## What changed vs v0.2.x

- v0.2.x run: A, B, C, D, G pass; E, F absent (governance engine not reachable).
- v0.3.0: added `PopulationRule` to the governance engine (the "≥K in a window" mechanic the fixed
  sequence rules could not express — the same shape benchmark-E needs), then wired E/F to the engine.
- **Full A–G now passes, reproducible across 3 runs**, every hard gate satisfied.

## Reproduce

```bash
cd benchmarks/wikimedia
# with agentic_os (public) + agentic_os_enterprise.governance (v0.3.0) on the path:
PYTHONPATH=.:/path/to/agentic-os-src python -m harness.run_benchmark --runs 3 --pairs 12
# → arms A B C D G E F all PASS; results/small/summary.json exit_gate_passed: true
```
