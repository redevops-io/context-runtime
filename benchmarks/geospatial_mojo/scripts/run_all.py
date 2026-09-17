"""Publishable benchmark runner (§9, §14). One command runs corpus gen → correctness gate → timed
benchmark (interleaved, repetitions) → statistics → summary.md, with environment capture.

  python scripts/run_all.py --profile {smoke,dev,publish} [--arms python,shapely,mojo]

Python is the semantic reference; Shapely is a reality-check; Mojo (when the bridge is available) is the
subject. No performance table is emitted for an arm that fails the correctness gate vs Python.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics as st
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CORPUS = os.path.join(ROOT, "corpus")
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "integration"))
from arms import get_arm  # noqa: E402

PROFILES = {  # (item cap per case, repetitions, sub-ms repetitions)
    "smoke": (40, 3, 5),
    "dev": (200, 7, 11),
    "publish": (10_000, 9, 21),
}


def load(name):
    p = os.path.join(CORPUS, f"{name}.jsonl")
    with open(p) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


# ---- workload execution via the uniform batch interface (Mojo = one subprocess per batch = integrated) ----
def run_pip(arm, items):
    return arm.pip_batch(items)

def run_pairs(arm, items):
    return arm.pairs_batch(items)

def run_join(arm, items):
    return arm.join_batch(items)

def run_batch(arm, items):
    # W4 runtime-style: match + serialize a stable Runtime-facing output per join item
    return [json.dumps({k: sorted(v) for k, v in sorted(res.items())}, separators=(",", ":"))
            for res in arm.join_batch(items)]

WORKLOADS = {
    "point_in_polygon": ("point_in_polygon", run_pip),
    "polygon_intersection": ("polygon_pairs", run_pairs),
    "spatial_join": ("spatial_join", run_join),
    "runtime_batch": ("spatial_join", run_batch),
}


def case_key(wl, it):
    return it.get("case", "?")


def time_case(fn, arm, items, reps):
    obs = []
    out0 = None
    for _ in range(2):  # warmups (untimed)
        out0 = fn(arm, items)
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        out = fn(arm, items)
        obs.append(time.perf_counter_ns() - t0)
    return out0, obs


def stats(obs):
    obs = sorted(obs)
    n = len(obs)
    def pct(p): return obs[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {"median": st.median(obs), "p10": pct(0.10), "p90": pct(0.90), "mean": st.fmean(obs),
            "stdev": (st.pstdev(obs) if n > 1 else 0.0), "min": obs[0], "max": obs[-1], "n": n}


def environment():
    env = {"timestamp": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
           "platform": platform.platform(), "processor": platform.processor(),
           "cpu_count": os.cpu_count()}
    try:
        env["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        env["git_commit"] = "unknown"
    try:
        from mojo_bridge import mojo_version  # noqa
        env["mojo_version"] = mojo_version()
    except Exception:
        env["mojo_version"] = None
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="dev", choices=list(PROFILES))
    ap.add_argument("--arms", default="python,shapely,mojo")
    args = ap.parse_args()
    cap, reps, reps_subms = PROFILES[args.profile]

    subprocess.check_call([sys.executable, os.path.join(HERE, "generate_corpus.py")])

    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = {}
    for a in want:
        try:
            if a == "mojo":
                from mojo_bridge import MojoArm
                arm = MojoArm()
            else:
                arm = get_arm(a)
            if getattr(arm, "available", True):
                arms[a] = arm
            else:
                print(f"[skip] arm '{a}' unavailable")
        except Exception as e:
            print(f"[skip] arm '{a}': {e}")
    assert "python" in arms, "python reference arm is required"

    runid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + platform.node()
    outdir = os.path.join(ROOT, "results", runid)
    os.makedirs(outdir, exist_ok=True)
    env = environment()
    if "mojo" in arms and getattr(arms["mojo"], "build_ns", None):
        env["mojo_build_seconds"] = round(arms["mojo"].build_ns / 1e9, 3)  # T6 clean optimized build
    json.dump(env, open(os.path.join(outdir, "environment.json"), "w"), indent=2)

    # ---- correctness gate (§8): every arm vs python reference, per corpus item ----
    print("== correctness gate ==")
    correctness = {}
    ref = arms["python"]
    for wl, (src, fn) in WORKLOADS.items():
        if wl == "runtime_batch":
            continue
        items = load(src)[:cap] if False else load(src)  # correctness uses the WHOLE corpus
        ref_out = fn(ref, items)
        for a, arm in arms.items():
            if a == "python":
                continue
            mism = []
            arm_out = fn(arm, items)
            for i, (r, x) in enumerate(zip(ref_out, arm_out)):
                if r != x:
                    mism.append({"workload": wl, "fixture": case_key(wl, items[i]), "python": _js(r), "arm": _js(x)})
            correctness.setdefault(a, {"general": 0, "mismatch": 0, "cases": []})
            correctness[a]["general"] += len(items)
            correctness[a]["mismatch"] += len(mism)
            correctness[a]["cases"] += mism[:20]
    # adversarial (point_in_polygon only)
    adv = load("adversarial")
    ref_adv = ref.pip_batch(adv)
    for a, arm in arms.items():
        if a == "python":
            continue
        am = 0
        arm_adv = arm.pip_batch(adv)
        for it, r, x in zip(adv, ref_adv, arm_adv):
            if r != x:
                am += 1
                correctness[a]["cases"].append({"workload": "adversarial", "fixture": it["case"], "python": r, "arm": x})
        correctness[a]["adversarial"] = len(adv)
        correctness[a]["adversarial_mismatch"] = am
    json.dump(correctness, open(os.path.join(outdir, "correctness.json"), "w"), indent=2)
    for a, c in correctness.items():
        print(f"   {a}: {c['general']} general ({c['mismatch']} mismatch), "
              f"{c.get('adversarial',0)} adversarial ({c.get('adversarial_mismatch',0)} mismatch)")

    parity_ok = {a: (c["mismatch"] == 0 and c.get("adversarial_mismatch", 0) == 0) for a, c in correctness.items()}
    parity_ok["python"] = True

    # ---- benchmark (§9): interleaved arms, repetitions, per case ----
    print("== benchmark ==")
    raw = open(os.path.join(outdir, "benchmark_raw.jsonl"), "w")
    summary = {}  # wl -> case -> arm -> stats(+digest)
    for wl, (src, fn) in WORKLOADS.items():
        by_case = {}
        for it in load(src):
            by_case.setdefault(case_key(wl, it), []).append(it)
        summary[wl] = {}
        for case, items in by_case.items():
            items = items[:cap]
            # sub-ms cases get more reps; decide from a quick python run
            _, probe = time_case(fn, arms["python"], items, 1)
            r = reps_subms if probe[0] < 1_000_000 else reps
            summary[wl][case] = {}
            for a, arm in arms.items():
                # §8 parity gate applies to a *replacement candidate* (mojo). Shapely is a reality-check
                # baseline (§4) with known differing boundary semantics — always reported, never gated.
                if a == "mojo" and not parity_ok.get("mojo", False):
                    continue
                out, obs = time_case(fn, arm, items, r)
                s = stats(obs); s["digest"] = digest([_js(o) for o in out]); s["items"] = len(items)
                if a == "mojo" and getattr(arm, "last_kernel_ns", None):
                    s["kernel_ns"] = arm.last_kernel_ns  # T1 native (from inside the Mojo process)
                summary[wl][case][a] = s
                for rep_i, ns in enumerate(obs):
                    raw.write(json.dumps({"schema": "redevops.geospatial-bench.v1", "workload": wl, "case": case,
                                          "implementation": a, "n": len(items), "repetition": rep_i,
                                          "elapsed_ns": ns, "result_digest": s["digest"], "status": "ok"}) + "\n")
            print(f"   {wl}/{case}: " + " ".join(
                f"{a}={summary[wl][case][a]['median']/1e6:.3f}ms" for a in summary[wl][case]))
    raw.close()
    json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=2)
    write_summary_md(os.path.join(outdir, "summary.md"), env, correctness, summary, args.profile)
    print(f"\nresults: {outdir}")


def _js(o):
    if isinstance(o, dict):
        return {k: sorted(v) if isinstance(v, list) else v for k, v in o.items()}
    return o


def write_summary_md(path, env, correctness, summary, profile):
    L = [f"# Geospatial Python vs Mojo — {profile} run", "",
         f"- {env['platform']} · Python {env['python']} · Mojo {env.get('mojo_version') or 'n/a'} · commit `{env['git_commit'][:10]}`",
         f"- Mojo clean optimized build (T6): {env.get('mojo_build_seconds','n/a')}s", ""]
    L += ["## Correctness (vs Python reference)", "",
          "| Arm | General | Adversarial | Mismatches |", "|---|---:|---:|---:|"]
    for a, c in correctness.items():
        L.append(f"| {a} | {c['general']} | {c.get('adversarial',0)} | {c['mismatch']+c.get('adversarial_mismatch',0)} |")
    L += ["", "## End-to-end performance (median ms; speedup = python/other)", "",
          "| Workload | Case | python | shapely | mojo | mojo speedup |", "|---|---|---:|---:|---:|---:|"]
    for wl, cases in summary.items():
        for case, arms in cases.items():
            py = arms.get("python", {}).get("median")
            def ms(a): return f"{arms[a]['median']/1e6:.3f}" if a in arms else "—"
            sp = (f"{py/arms['mojo']['median']:.2f}×" if "mojo" in arms and py else "—")
            L.append(f"| {wl} | {case} | {ms('python')} | {ms('shapely')} | {ms('mojo')} | {sp} |")
    # Kernel (T1 native, timed inside the Mojo process) vs integrated (T3, incl. serialization/subprocess).
    L += ["", "## Mojo: native kernel (T1) vs integrated (T3)  — exposes the Python↔Mojo boundary cost", "",
          "| Workload | Case | Python (ms) | Mojo native kernel (ms) | Mojo integrated (ms) | Boundary (ms) | Kernel speedup |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for wl, cases in summary.items():
        for case, arms in cases.items():
            m = arms.get("mojo", {}); py = arms.get("python", {}).get("median")
            if "kernel_ns" not in m:
                continue
            kern = m["kernel_ns"] / 1e6; integ = m["median"] / 1e6
            ksp = f"{(py/1e6)/kern:.2f}×" if py and kern > 0 else "—"
            L.append(f"| {wl} | {case} | {py/1e6:.3f} | {kern:.3f} | {integ:.3f} | {integ-kern:.3f} | {ksp} |")
    # crossover (§12): smallest case where integrated mojo < python, per workload
    L += ["", "## Integrated crossover (§12)", "", "| Workload | First case Mojo wins integrated |", "|---|---|"]
    for wl, cases in summary.items():
        first = "none observed"
        for case, arms in cases.items():
            py = arms.get("python", {}).get("median"); mj = arms.get("mojo", {}).get("median")
            if py and mj and mj < py:
                first = case; break
        L.append(f"| {wl} | {first} |")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
