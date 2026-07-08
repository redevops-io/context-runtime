#!/usr/bin/env python3
"""Aggregate result JSONL(s) → a markdown table + the crossover plot.

The headline question is visual: plot answer-accuracy vs pollution, one line per
(model, arm). If a smaller model + Context Runtime (A2) stays flat while a bigger model
managing context natively (A0/A1) degrades — and they cross — that's the result.

  python -m harness.report results/*.jsonl --md results/summary.md --plot results/crossover.png
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict


def load(paths) -> list:
    rows = []
    for pat in paths:
        for path in glob.glob(pat):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(rows) -> dict:
    """(model_name, arm, pollution) → aggregated metrics."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model_name"], r["arm"], r["pollution"])].append(r)
    agg = {}
    for key, rs in groups.items():
        agg[key] = {
            "n": len(rs),
            "accuracy": _mean([r["correct"] for r in rs]),
            "retr_hit": _mean([r["retrieval"].get("hit") for r in rs]),
            "retr_recall": _mean([r["retrieval"].get("recall") for r in rs]),
            "retr_prec": _mean([r["retrieval"].get("precision") for r in rs]),
            "pollution_frac": _mean([r.get("pollution_frac") for r in rs]),
            "latency_s": _mean([r["latency_s"] for r in rs]),
            "prompt_tokens": _mean([r["prompt_tokens"] for r in rs]),
        }
    return agg


def to_markdown(agg) -> str:
    def pct(x): return f"{x:.0%}" if isinstance(x, (int, float)) else "—"
    def num(x, d=0): return (f"{x:.{d}f}" if isinstance(x, (int, float)) else "—")
    lines = ["| model | arm | pollution | acc | retr hit | pollution% | tok(in) | lat s | n |",
             "|---|---|--:|--:|--:|--:|--:|--:|--:|"]
    for (m, arm, pol) in sorted(agg):
        a = agg[(m, arm, pol)]
        lines.append(f"| {m} | {arm} | {pol} | {pct(a['accuracy'])} | {pct(a['retr_hit'])} "
                     f"| {pct(a['pollution_frac'])} | {num(a['prompt_tokens'])} "
                     f"| {num(a['latency_s'],1)} | {a['n']} |")
    return "\n".join(lines)


def plot(agg, out_png: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(matplotlib unavailable: {e}) — skipping plot")
        return
    series = defaultdict(list)   # (model, arm) -> [(pollution, acc)]
    for (m, arm, pol), a in agg.items():
        if a["accuracy"] is not None:
            series[(m, arm)].append((pol, a["accuracy"]))
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {"full_dump": ":", "naive_rag": "--", "context_runtime": "-"}
    for (m, arm), pts in sorted(series.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys, styles.get(arm, "-"), marker="o", label=f"{m} · {arm}")
    ax.set_xlabel("context pollution (distractor filings mixed in)")
    ax.set_ylabel("answer accuracy")
    ax.set_title("Can better context management beat a better model?")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--md", default=None)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args(argv)
    rows = load(args.paths)
    agg = aggregate(rows)
    md = to_markdown(agg)
    print(md)
    if args.md:
        with open(args.md, "w") as f:
            f.write(md + "\n")
        print(f"\nwrote {args.md}")
    if args.plot:
        plot(agg, args.plot)


if __name__ == "__main__":
    main()
