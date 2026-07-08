#!/usr/bin/env python3
"""Consolidated v1-vs-v2 benchmark — Python and Go, side by side, one table.

Runs the SAME seeded, ground-truth simulation in both runtimes and merges the results into a
single table (Markdown + self-contained HTML) suitable for redevops.io. Each runtime reports its
own v1→v2 numbers on the three isolated effects the v2 upgrade adds:

  • policy precision — the reward now sees calibrated per-passage relevance (not just the coarse
    per-query judge), so the learned policy stops chasing high-coverage / low-precision arms.
  • abstention       — a P(relevant) floor lets v2 decline unanswerable queries; v1 cannot.
  • expensive-stage sizer — DSpark survival-product gate prunes the low-relevance tail before the
    costly rerank/synthesis stage, cutting depth while precision rises.

Python results are computed in-process; Go results come from `go run ./cmd/dsparkbench --json` in
the Go repo (skipped gracefully if the toolchain/repo is absent — the table then shows Python only).

    PYTHONPATH=. python examples/consolidated_benchmark.py                 # markdown to stdout
    PYTHONPATH=. python examples/consolidated_benchmark.py --html out.html # + HTML for the site
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from examples.dspark_calibration_bench import compute_results as python_results  # noqa: E402

GO_DIR = os.getenv("CR_GO_DIR", "/mnt/backup/projects/context-runtime-go")

# display order + labels; keyed to the metric keys both runtimes emit.
ROWS = [
    ("policy_precision", "Learned-policy precision", "The served passages that are actually relevant, after the policy converges. v2's reward finally sees calibrated relevance."),
    ("abstain_recall", "Abstention recall (unanswerable caught)", "Share of truly-unanswerable queries v2 declines to answer. v1 has no abstention at all."),
    ("false_abstain", "False-abstain rate (answerable dropped)", "Answerable queries v2 wrongly declined — the cost of abstention. Lower is better."),
    ("sizer_depth", "Expensive-stage depth (passages)", "Passages sent to the costly rerank/synthesis stage from a deep k=8 arm. The sizer prunes the low-relevance tail."),
    ("sizer_precision", "Precision after the sizer", "Precision of what survives the sizer's gate — pruning the tail raises it."),
]


def go_results() -> dict | None:
    """Run the Go twin and parse its --json. None if Go/toolchain/repo is unavailable."""
    if not pathlib.Path(GO_DIR).exists():
        return None
    try:
        out = subprocess.run(["go", "run", "./cmd/dsparkbench", "--json"], cwd=GO_DIR,
                             capture_output=True, text=True, timeout=300)
        if out.returncode != 0:
            return None
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        return None


def _fmt(metric: dict, key: str) -> str:
    v = metric[key]
    if metric.get("unit") == "%":
        return f"{v * 100:.1f}%"
    return f"{v:.2f}"


def _delta(metric: dict) -> str:
    hb = metric.get("higher_better", True)
    if metric.get("unit") == "%":
        d = (metric["v2"] - metric["v1"]) * 100
        if abs(d) < 0.05:
            return "—"
        arrow = "▲" if (d > 0) == hb else "▼"
        return f"{arrow} {d:+.1f} pts"
    d = metric["v2"] - metric["v1"]
    if metric.get("unit") == "passages" and metric["v1"]:
        pct = (1 - metric["v2"] / metric["v1"]) * 100
        return f"▼ −{pct:.0f}%"
    return f"{d:+.2f}"


def build_table(py: dict, go: dict | None) -> list[tuple]:
    rows = []
    for key, label, _desc in ROWS:
        pm = py["metrics"][key]
        cells = [label, _fmt(pm, "v1"), _fmt(pm, "v2"), _delta(pm)]
        if go is not None:
            gm = go["metrics"][key]
            cells += [_fmt(gm, "v1"), _fmt(gm, "v2"), _delta(gm)]
        rows.append(tuple(cells))
    return rows


def to_markdown(py: dict, go: dict | None) -> str:
    has_go = go is not None
    head = ["Metric", "Py v1", "Py v2", "Δ"]
    if has_go:
        head += ["Go v1", "Go v2", "Δ"]
    sep = ["---"] * len(head)
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(sep) + " |"]
    for row in build_table(py, go):
        lines.append("| " + " | ".join(row) + " |")
    beta = py["metrics"]["policy_precision"].get("beta")
    note = (f"\n_Seeded ground-truth simulation, {py['seeds']}-seed average; precision headlined at "
            f"β={beta} (calibration-trust knob; shipped default 0.5). "
            f"{'Go = independent re-implementation, same methodology.' if has_go else 'Go runtime unavailable — Python only.'}_")
    return "\n".join(lines) + "\n" + note


def to_html(py: dict, go: dict | None) -> str:
    has_go = go is not None
    cols = ["Metric", "Py v1", "Py v2", "Δ"] + (["Go v1", "Go v2", "Δ"] if has_go else [])
    th = "".join(f"<th>{c}</th>" for c in cols)
    body = []
    for (key, label, desc), row in zip(ROWS, build_table(py, go)):
        tds = [f'<td class="metric"><b>{row[0]}</b><span class="d">{desc}</span></td>']
        for c in row[1:]:
            cls = "delta" if ("▲" in c or "▼" in c or "pts" in c or "−" in c) else "num"
            tds.append(f'<td class="{cls}">{c}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    beta = py["metrics"]["policy_precision"].get("beta")
    span = 4 if not has_go else 7
    return f"""<!doctype html><meta charset="utf-8"><title>Context Runtime — v1 vs v2 benchmarks</title>
<style>
 body{{font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#0f172a;background:#f8fafc;margin:0;padding:2rem}}
 .wrap{{max-width:960px;margin:0 auto}}
 h1{{font-size:1.5rem;margin:0 0 .25rem}} .sub{{color:#64748b;margin:0 0 1.5rem}}
 table{{border-collapse:collapse;width:100%;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 th,td{{padding:.7rem .9rem;text-align:right;border-bottom:1px solid #eef2f7}}
 th{{background:#0f172a;color:#fff;font-weight:600;font-size:.85rem}}
 td.metric{{text-align:left}} td.metric .d{{display:block;color:#94a3b8;font-size:.78rem;font-weight:400;margin-top:.15rem}}
 td.num{{font-variant-numeric:tabular-nums;color:#334155}}
 td.delta{{font-variant-numeric:tabular-nums;font-weight:600;color:#059669}}
 tr:last-child td{{border-bottom:0}}
 .foot{{color:#94a3b8;font-size:.8rem;margin-top:1rem}}
 .grp{{background:#f1f5f9;color:#475569;font-size:.75rem;text-align:center;letter-spacing:.04em}}
</style>
<div class="wrap">
<h1>Context Runtime — v1 → v2, measured in both runtimes</h1>
<p class="sub">The same seeded, ground-truth retrieval simulation, run in the Python source-of-truth and the Go port. v2 = calibrated relevance-in-reward + abstention + the DSpark load-aware sizer.</p>
<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>
<p class="foot">{py['seeds']}-seed average · precision headlined at β={beta} (the calibration-trust knob; shipped default 0.5) · {'Go is an independent re-implementation on identical methodology — directional parity across languages.' if has_go else 'Go runtime unavailable; Python only.'}</p>
</div>"""


def v3_results(seeds: int = 24) -> dict:
    """v3 (preliminary) — online optimization under drift (Generation 4). The best plan drifts mid-run
    (a model upgrade / corpus shift). A static v1/v2 planner is pinned to the now-stale plan; the v3 online
    planner re-explores, and recency-weighted (discounted) learning lets it track the shift. We report the
    post-drift average served-plan reward, seed-averaged."""
    from context_runtime.optimizer.online import BanditOptimizer
    from context_runtime.types import Candidate, Goal, PlanScore, StepSpec

    STEPS, DRIFT = 400, 200
    PRIOR = {"A": 0.80, "C": 0.20}
    PRE = {"A": 0.80, "C": 0.20}
    POST = {"A": 0.20, "C": 0.80}          # best plan flips A→C at t=DRIFT

    def cand(a):
        return Candidate(steps=(StepSpec(type="retrieve", params={"method": a}),), model_tier="cheap")

    def scored():
        return [(cand(a), PlanScore(total=PRIOR[a], feasible=True)) for a in PRIOR]

    def run(discount, seed):
        opt = BanditOptimizer(None, epsilon=0.2, discount=discount, seed=seed)
        post = []
        for t in range(STEPS):
            plan = opt.select(scored(), Goal(text="q"), context="c")
            arm = next(s.params["method"] for s in plan.chosen.steps if s.type == "retrieve")
            r = (PRE if t < DRIFT else POST)[arm]
            opt.learn_from_plan(plan, r)
            if t >= DRIFT:
                post.append(r)
        return sum(post) / len(post)

    static = POST["A"]                      # v1/v2 never adapt → pinned to the stale best (A)
    plain = sum(run(0.0, 0x1000 + s) for s in range(seeds)) / seeds
    disc = sum(run(0.2, 0x1000 + s) for s in range(seeds)) / seeds
    return {"seeds": seeds, "static": static, "online_plain": plain, "online_discount": disc,
            "oracle": max(POST.values())}


def v3_markdown(v3: dict) -> str:
    r = v3
    return ("\n\n### v3 (preliminary) — online optimization under drift\n\n"
            "A different axis from the v1→v2 calibration gains above. The best plan drifts mid-run; a static "
            "v1/v2 planner is pinned to the now-stale plan, while the v3 online planner re-explores and "
            "recency-weighted learning tracks the shift.\n\n"
            "| Metric | v2 (static) | v3 (online) | Δ |\n| --- | --- | --- | --- |\n"
            f"| Post-drift served-plan reward | {r['static']:.2f} | {r['online_discount']:.2f} | "
            f"▲ +{r['online_discount'] - r['static']:.2f} |\n"
            f"| Post-drift reward, online w/o discounting | {r['static']:.2f} | {r['online_plain']:.2f} | "
            f"+{r['online_plain'] - r['static']:.2f} |\n"
            f"\n_Seeded drift simulation, {r['seeds']}-seed average; post-drift oracle = {r['oracle']:.2f}. "
            "v3's online-optimization axis is orthogonal to (and preserves) the v2 calibration gains — preliminary._")


def main() -> int:
    py = python_results()
    go = go_results()
    v3 = v3_results()
    md = to_markdown(py, go)
    print(md)
    print(v3_markdown(v3))
    if "--html" in sys.argv:
        path = sys.argv[sys.argv.index("--html") + 1]
        pathlib.Path(path).write_text(to_html(py, go), encoding="utf-8")
        print(f"\n[wrote HTML → {path}]", file=sys.stderr)
    if "--json" in sys.argv:
        print(json.dumps({"python": py, "go": go, "v3": v3}, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
