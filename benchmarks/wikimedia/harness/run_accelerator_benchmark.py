"""Run the v0.3.0 accelerator crossover arms (H/I/J) on the GPU host.

Loads the real-data inputs prepared by ``accelerators.prepare_inputs`` (shipped from the dev box), then for
each arm: asserts CPU-reference == GPU semantic equivalence at every size, measures the CPU-vs-GPU latency
crossover, and writes a result bundle. Correctness gates performance — an arm only "passes" if every GPU
result matched its CPU reference; the timings then say *where*, if anywhere, the accelerator is worth it.

    PYTHONPATH=. python -m harness.run_accelerator_benchmark --inputs /root/accel_inputs.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from accelerators import arm_h_evidence as H, arm_i_retrieval as I, arm_j_planning as J
from accelerators.common import gpu_available, gpu_info

RESULTS = Path(__file__).resolve().parent.parent / "results" / "accel"


def _sizes(cap: int, ladder) -> list[int]:
    out = [s for s in ladder if s < cap] + [cap]
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=Path("/root/accel_inputs.npz"))
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    if not gpu_available():
        print("NO GPU — the accelerator arms require cudf/cuvs/cuopt + a CUDA device. Aborting.")
        return 2
    info = gpu_info()
    print(f"GPU: {info.get('name')} cc{info.get('compute_capability')} "
          f"{info.get('vram_free_gb')}/{info.get('vram_total_gb')} GB free")

    d = np.load(args.inputs)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # --- Arm H: cuDF evidence/temporal processing -----------------------------------------------------
    h_arrays = (d["h_ref"], d["h_revid"], d["h_sha"], d["h_ts"])
    ncap = len(h_arrays[0])
    ts = h_arrays[3]
    split_ts, later_ts = int(np.percentile(ts, 60)), int(np.percentile(ts, 100))
    h_inc_ok = H.incremental_equals_full(h_arrays, split_ts, later_ts)
    # real dump (ncap rows) then distribution-tiled beyond it to reach the cuDF crossover
    h_sizes = sorted({s for s in [5000, 50000, ncap, 200000, 1_000_000, 2_000_000]})
    h_rows = H.sweep(h_arrays, h_sizes, repeats=args.repeats)
    h_pass = h_inc_ok and all(r["correct"] for r in h_rows)
    print(f"\narm H (cuDF): incremental==full={h_inc_ok}; "
          f"equivalence {sum(r['correct'] for r in h_rows)}/{len(h_rows)} sizes")
    for r in h_rows:
        tag = " (scaled)" if r["scaled"] else " (real) "
        print(f"  n={r['n']:8d}{tag} cpu={r['cpu_ms_median']:8.2f}ms  gpu={r['gpu_ms_median']:7.2f}ms  "
              f"speedup={r['speedup']}x  ok={r['correct']}")

    # --- Arm I: cuVS retrieval ------------------------------------------------------------------------
    emb, q_emb = d["emb"], d["q_emb"]
    mcap = emb.shape[0]
    have_hnsw = True
    try:
        import hnswlib  # noqa: F401
    except Exception:
        have_hnsw = False
    i_sizes = sorted({s for s in [1000, 5000, mcap, 50000, 200000]})
    i_rows = I.sweep(emb, q_emb, i_sizes, repeats=max(3, args.repeats - 2), have_hnsw=have_hnsw)
    # Identity gate applies to the REAL embeddings (the actual retrieval claim). Scaled sizes are
    # distribution-extended for throughput only — heavy tiling perturbs recall, so it is informational.
    real_rows = [r for r in i_rows if not r["scaled"]]
    i_pass = all(r["correct"] for r in real_rows)
    print(f"\narm I (cuVS): identity recall>= {I.RECALL_FLOOR} on real sizes "
          f"{sum(r['correct'] for r in real_rows)}/{len(real_rows)}  (query-only = amortised prebuilt index; "
          f"scaled recall informational)")
    for r in i_rows:
        tag = "scaled" if r["scaled"] else " real "
        hn = f" hnsw_q={r.get('hnsw_query_ms','-')}ms(r={r.get('hnsw_recall','-')})" if have_hnsw else ""
        print(f"  n={r['n']:7d}[{tag}] exact_q={r['cpu_exact_query_ms']:8.2f}ms{hn}  "
              f"cuvs_build={r['cuvs_build_ms']:7.2f}ms cuvs_q={r['cuvs_query_ms']:6.2f}ms(r={r['cuvs_recall']})  "
              f"q_speedup_vs_exact={r['query_speedup_vs_exact']}x  ok={r['correct']}")

    # --- Arm J: cuOpt planning ------------------------------------------------------------------------
    j_value, j_tokens = d["j_value"], d["j_tokens"]
    jcap = len(j_value)
    # explicit sizes (do NOT append jcap): context-assembly candidate pools are tens-to-hundreds, and the
    # DP capacity dimension makes a full-corpus instance meaningless (and its exact DP astronomically slow).
    jsizes = [n for n in [20, 50, 100, 200, 500] if n <= jcap]
    # cuOpt latency is dominated by fixed setup/solve overhead (low variance), so a single timed solve per
    # size is enough — and avoids paying the solver's seconds-scale cost several times over.
    # (a) realistic bounded context window: the CPU should win — GPU correctly not selected
    j_fixed = J.sweep(j_value, j_tokens, jsizes, fixed_budget=8000, repeats=1)
    # (b) capacity-scaled: grow the DP dimension until cuOpt overtakes — the measured crossover
    j_scaled = J.sweep(j_value, j_tokens, jsizes, budget_frac=0.5, repeats=1)
    j_pass = all(r["correct"] for r in j_fixed) and all(r["correct"] for r in j_scaled)
    crossover = next((r["n"] for r in j_scaled if r["gpu_wins"]), None)
    print(f"\narm J (cuOpt): objective-match "
          f"{sum(r['correct'] for r in j_fixed+j_scaled)}/{len(j_fixed)+len(j_scaled)}; "
          f"scaled GPU-wins crossover at n={crossover}")
    print("  (a) fixed 8k-token context window — CPU expected to win:")
    for r in j_fixed:
        print(f"    n={r['n']:5d} budget={r['budget']:7d}  cpu_dp={r['cpu_dp_ms_median']:8.2f}ms  "
              f"cuopt={r['cuopt_ms_median']:7.2f}ms  obj_match={r['objective_match']}  gpu_wins={r['gpu_wins']}")
    print("  (b) capacity-scaled (budget=50% of candidate tokens) — crossover:")
    for r in j_scaled:
        print(f"    n={r['n']:5d} budget={r['budget']:7d}  cpu_dp={r['cpu_dp_ms_median']:8.2f}ms  "
              f"cuopt={r['cuopt_ms_median']:7.2f}ms  speedup={r['speedup']}x  "
              f"obj_match={r['objective_match']}  gpu_wins={r['gpu_wins']}")

    summary = {
        "benchmark": "evidence-native-runtime/wikimedia/accelerators",
        "stage": "v0.3.0-accelerators", "corpus": "strategywiki-20260801",
        "gpu": info,
        "arms": {
            "H": {"name": H.NAME, "passed": bool(h_pass), "incremental_equals_full": bool(h_inc_ok),
                  "rows": h_rows},
            "I": {"name": I.NAME, "passed": bool(i_pass), "recall_floor": I.RECALL_FLOOR,
                  "have_hnsw": have_hnsw, "rows": i_rows},
            "J": {"name": J.NAME, "passed": bool(j_pass), "gpu_wins_crossover_n": crossover,
                  "rows_fixed_8k_budget": j_fixed, "rows_capacity_scaled": j_scaled},
        },
        "all_correct": bool(h_pass and i_pass and j_pass),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== ACCELERATOR SUMMARY ===")
    for a in ("H", "I", "J"):
        print(f"  arm {a}: {'PASS' if summary['arms'][a]['passed'] else 'FAIL'} — {summary['arms'][a]['name']}")
    print(f"  wrote {RESULTS/'summary.json'}")
    return 0 if summary["all_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
