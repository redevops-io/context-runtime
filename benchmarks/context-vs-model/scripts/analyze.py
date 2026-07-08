#!/usr/bin/env python3
"""Quick crossover analysis: accuracy + retrieval + pollution by arm × pollution level.
Numeric questions only by default (authoritative grading without a judge)."""
import json
import sys
from collections import defaultdict

path = sys.argv[1]
numeric_only = "--all" not in sys.argv
rows = [json.loads(l) for l in open(path)]
if numeric_only:
    rows = [r for r in rows if r.get("is_numeric")]

agg = defaultdict(lambda: {"c": 0, "n": 0, "hit": 0.0, "poll": 0.0, "tok": 0})
for r in rows:
    k = (r.get("reasoning", "?"), r["arm"], r["pollution"])
    a = agg[k]
    a["c"] += r["correct"]; a["n"] += 1
    a["hit"] += (r["retrieval"]["hit"] or 0); a["poll"] += r["pollution_frac"]
    a["tok"] += r["prompt_tokens"]

print(f"{'think':5s} {'arm':16s} {'pol':>3s} {'acc':>5s} {'rhit':>5s} {'poll':>5s} {'tok':>6s}  n")
for k in sorted(agg):
    a = agg[k]; n = a["n"]
    print(f"{k[0]:5s} {k[1]:16s} {k[2]:>3d} {a['c']/n:>5.0%} {a['hit']/n:>5.0%} "
          f"{a['poll']/n:>5.0%} {a['tok']//n:>6d}  {n}")

# crossover: naive vs CR accuracy as pollution rises
print("\n-- accuracy vs pollution (numeric) --")
for arm in ("full_dump", "naive_rag", "context_runtime"):
    pts = sorted((k[2], agg[k]["c"] / agg[k]["n"]) for k in agg if k[1] == arm)
    print(f"  {arm:16s} " + "  ".join(f"p{p}={acc:.0%}" for p, acc in pts))
