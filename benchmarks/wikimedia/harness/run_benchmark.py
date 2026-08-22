"""Run the evidence-native Wikimedia benchmark (arms A–G) against the frozen v0.2.x runtimes.

Stage-1 small-corpus correctness run (plan §12/§13): select a fixed slice of real strategywiki
revisions, run each arm ≥3 times from clean state, assert the semantic outputs are identical across
runs, and write a result bundle (per-run JSON + summary) pinning dataset + runtime versions.

    PYTHONPATH=. python -m harness.run_benchmark [--runs 3] [--pairs 12]

Arms E and F (cross-series governance / OBSERVE→ENFORCE) are intentionally absent: the governance
engine they need does not exist in the v0.2.x runtimes (deferred to the v0.3.0 private release). Their
corpus selection is wired (evidence_corpus.select_protected_page_trajectories) and ready.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from importlib import metadata
from pathlib import Path

from harness import arm_a_replay, arm_b_incremental, arm_c_lineage, arm_d_freshness, arm_g_deterministic
from harness.evidence_corpus import select_revision_pairs

RESULTS = Path(__file__).resolve().parent.parent / "results" / "small"


def _runtime_pins() -> dict:
    pins = {}
    for dist in ("runtime-contracts", "redevops-rag", "context-runtime", "agentic-os", "discovery-runtime"):
        try:
            pins[dist] = metadata.version(dist)
        except Exception:
            pins[dist] = "not-installed"
    return pins


def _governance_available() -> bool:
    try:
        import agentic_os_enterprise.governance  # noqa: F401
        return True
    except Exception:
        return False


def _run_once(pairs, workdir: str) -> list[dict]:
    out = [
        arm_a_replay.run(pairs, workdir),
        arm_b_incremental.run(pairs),
        arm_c_lineage.run(pairs),
        arm_d_freshness.run(pairs),
        arm_g_deterministic.run(pairs),
    ]
    # Arms E/F need the v0.3.0 governance engine (agentic_os_enterprise.governance, PopulationRule +
    # CrossSeriesRule + OBSERVE/ENFORCE). Run them when it is importable; otherwise they stay deferred.
    if _governance_available():
        from harness import governance_arms
        out.append(governance_arms.run_arm_e())
        out.append(governance_arms.run_arm_f())
    return out


def _semantic_key(arm_result: dict) -> tuple:
    """The reproducibility fingerprint of an arm: pass/fail + every metric except timings."""
    m = {k: v for k, v in arm_result["metrics"].items() if not k.endswith("_ms_p50")}
    return (arm_result["arm"], arm_result["passed"], arm_result["n_cases"], json.dumps(m, sort_keys=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--pairs", type=int, default=12)
    args = ap.parse_args()

    pins = _runtime_pins()
    print(f"runtime pins: {pins}")
    pairs = select_revision_pairs(args.pairs)
    print(f"selected {len(pairs)} real revision pairs from strategywiki (deterministic)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    all_runs: list[list[dict]] = []
    for r in range(1, args.runs + 1):
        with tempfile.TemporaryDirectory() as wd:
            results = _run_once(pairs, wd)
        all_runs.append(results)
        run_doc = {"run": r, "runtime_pins": pins, "n_pairs": len(pairs), "arms": results}
        (RESULTS / f"run-{r:03d}.json").write_text(json.dumps(run_doc, indent=2))
        status = " ".join(f"{a['arm']}:{'PASS' if a['passed'] else 'FAIL'}" for a in results)
        print(f"run {r}: {status}")

    # reproducibility: semantic outputs identical across all runs
    keys_per_run = [tuple(_semantic_key(a) for a in run) for run in all_runs]
    reproducible = all(k == keys_per_run[0] for k in keys_per_run)

    # exit gate (plan §13): every arm passes, and reruns reproduce
    arms_pass = {a["arm"]: all(run[i]["passed"] for run in all_runs)
                 for i, a in enumerate(all_runs[0])}
    all_pass = all(arms_pass.values())

    summary = {
        "benchmark": "evidence-native-runtime/wikimedia",
        "stage": "small", "corpus": "strategywiki-20260801",
        "runtime_pins": pins, "n_pairs": len(pairs), "runs": args.runs,
        "reproducible_across_runs": reproducible,
        "arms_pass": arms_pass, "exit_gate_passed": all_pass and reproducible,
        "arms_present": [a["arm"] for a in all_runs[-1]],
        "governance_engine": ("agentic_os_enterprise.governance (v0.3.0)"
                              if _governance_available() else "not installed — E/F deferred"),
        "last_run_metrics": {a["arm"]: a["metrics"] for a in all_runs[-1]},
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== SUMMARY ===")
    for arm, ok in arms_pass.items():
        print(f"  arm {arm}: {'PASS' if ok else 'FAIL'}")
    print(f"  reproducible across {args.runs} runs: {reproducible}")
    print(f"  EXIT GATE: {'PASSED' if summary['exit_gate_passed'] else 'FAILED'}")
    return 0 if summary["exit_gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
