# Geospatial End-to-End Benchmark: Python vs Mojo 1.1

Reproducible comparison of the production pure-Python geospatial kernel
(`context_runtime/geospatial/engine.py`) against a behaviorally-equivalent **Mojo 1.1** port.
Implements the spec in `REDEVOPS_GEOSPATIAL_PYTHON_VS_MOJO_E2E_BENCHMARK.md`.

## Run

```bash
# needs Mojo 1.1 (pixi global install -c conda-forge -c https://conda.modular.com/max mojo)
# and shapely (optional reality-check arm)
python scripts/run_all.py --profile smoke      # correctness + tiny cases
python scripts/run_all.py --profile dev        # correctness + reduced matrix (default)
python scripts/run_all.py --profile publish    # full matrix + repetitions
python scripts/run_all.py --arms python,mojo   # subset of arms
```
Each run writes `results/<ts>-<host>/` with `environment.json`, `correctness.json`,
`benchmark_raw.jsonl` (every observation), `summary.json`, and `summary.md` (article tables).

## Arms
- **python** — the unmodified production engine. **Semantic reference** (§2).
- **mojo** — a faithful 1.1 port (`mojo/geospatial.mojo`), same epsilon/tie-break (`1e-9`, `1e-12`). Built optimized (`mojo build`) before timing.
- **shapely** — reality-check baseline (§4 Arm C); known differing boundary semantics, so it is **not** a parity target.

## Integration boundary (important)
Mojo is driven through `integration/mojo_bridge.py` as **one native-binary subprocess per batch**, exchanging a flat float protocol (`mojo/runner.mojo` parses it natively). Two numbers are reported separately:
- **T1 native kernel** — timed *inside* the Mojo process (geometry loop only).
- **T3 integrated** — the whole Python round-trip (serialize → subprocess → parse), i.e. what the Runtime would feel.

This boundary is **batch subprocess + text serialization**, deliberately the conservative (worst-case) integration. A compiled shared-library / in-process FFI or a binary protocol would shrink the boundary substantially and move the integrated crossover down. Do not read T1 as the product number (§5.2).

## Results (dev run, this machine — see `results/` for the authoritative JSON)
- **Correctness gate: Mojo 0 mismatches** across the general corpus **and** the adversarial epsilon suite → deterministic parity holds (H3). Shapely diverges (≈20) as expected.
- **Native kernel: Mojo 5.7×–42× faster** than Python (raw compute; H1 confirmed).
- **Integrated: workload-dependent** (H2/H4):
  - `polygon_intersection` (O(n²) edge test): Mojo wins from ~64 vertices, up to **~8.9×** at 256.
  - `spatial_join` / `runtime_batch`: crossover around **L300×R100** (~1.2×); Python faster below.
  - `point_in_polygon`: Python wins — the fixed subprocess floor (~7 ms) plus text serialization of large coordinate arrays (up to ~195 ms at 1024-vertex × 200 queries) erases the kernel win.
- Shapely is fast on small point-in-polygon but **slower than Mojo on the large joins**, which is exactly why it is included (prevents a misleading "Mojo vs Python" headline).
- Mojo clean optimized build (T6): sub-second.

### Classification (§22): **CONDITIONAL**
The Mojo kernel is dramatically faster and passes deterministic parity, but the current text/subprocess boundary means it only wins **end-to-end above a measured per-workload crossover** (compute-heavy geometry). Dispatch heavy `polygon_intersection` / large `spatial_join` to Mojo; keep Python below the crossover. A binary/FFI boundary is the obvious next step to lower the crossover.

## Reproducibility / evidence (§19)
Corpus is seeded (`0x5EED2026`, see `corpus/manifest.json`); inputs are persisted so both arms consume identical data. Every raw observation (with a `result_digest` to catch drift) is kept in `benchmark_raw.jsonl` — the fastest run is never retained alone. Environment (CPU/OS/Python/Mojo versions, commit) is captured per run.

## Known limitations
- Integration is batch-subprocess with **text** serialization (not per-call FFI / binary) — the dominant boundary cost; integrated numbers are therefore a conservative lower bound on Mojo's product potential.
- `dev`/`smoke` use a reduced matrix; the largest spec sizes (1M queries, 100k×1k joins) need corpus scaling in `generate_corpus.py` and are gated by the per-case timeout (§13).
- Shapely uses `covers()` (interior∪boundary) as the closest analogue to the reference tie-break; it is a baseline, not a parity oracle.
- One geospatial kernel — **do not generalize to "Mojo vs Python" as languages** (§18).
