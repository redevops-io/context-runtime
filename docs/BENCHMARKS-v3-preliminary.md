# Benchmarks — v3 (preliminary)

> **Preliminary.** This is a forward-looking axis distinct from — and not a replacement for — the
> shipped v1→v2 calibration results in [`BENCHMARKS.md`](../BENCHMARKS.md). It is kept in its own
> file until promoted. Reproduce with `PYTHONPATH=. python examples/consolidated_benchmark.py --v3-doc docs/BENCHMARKS-v3-preliminary.md`.

## Online optimization under drift (Generation 4)

The best plan drifts mid-run (a model upgrade, a corpus shift). A **static** v1/v2 planner is pinned
to the now-stale plan; the **v3 online** planner re-explores, and recency-weighted (discounted)
learning lets it track the shift. We report the post-drift average served-plan reward, seed-averaged.

| Metric | v2 (static) | v3 (online) | Δ |
| --- | --- | --- | --- |
| Post-drift served-plan reward | 0.20 | 0.67 | ▲ +0.47 |
| Post-drift reward, online w/o discounting | 0.20 | 0.34 | +0.14 |

_Seeded drift simulation, 24-seed average; post-drift oracle = 0.80. Discounting is what converts online learning from a marginal gain into recovery of most of the
oracle reward after the drift. This axis is orthogonal to (and preserves) the v2 calibration gains._
