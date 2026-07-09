#!/usr/bin/env python3
"""Aggregate the judged results of all models → table + crossover plot.
Reads results/full/*.judged.jsonl (grok-4.5 verdicts in correct_judged)."""
import glob
import json
from collections import defaultdict

rows = []
for p in glob.glob("results/full/*.judged.jsonl"):
    rows += [json.loads(l) for l in open(p)]

MODELS = ["qwen3-coder-next", "nemotron-super", "deepseek-v4-flash"]
ARMS = ["full_dump", "naive_rag", "context_runtime"]
SIZE = {"qwen3-coder-next": "80B/3B", "nemotron-super": "121B/13B", "deepseek-v4-flash": "284B/13B"}

agg = defaultdict(lambda: {"c": 0, "n": 0, "hit": 0.0, "poll": 0.0, "lat": 0.0, "tok": 0})
for r in rows:
    k = (r["model_name"], r["arm"], r["pollution"])
    a = agg[k]
    a["c"] += r.get("correct_judged", r["correct"]); a["n"] += 1
    a["hit"] += (r["retrieval"]["hit"] or 0); a["poll"] += r["pollution_frac"]
    a["lat"] += r["latency_s"]; a["tok"] += r["prompt_tokens"]

def acc(m, arm, pol):
    a = agg.get((m, arm, pol))
    return (a["c"] / a["n"]) if a and a["n"] else None

print("# Context-vs-Model on LiveRAG (grok-4.5 judged)\n")
print(f"| model | size(tot/act) | arm | acc@pol0 | acc@pol150 | Δ | retr_hit | lat s | tok(in) |")
print(f"|---|---|---|--:|--:|--:|--:|--:|--:|")
for m in MODELS:
    for arm in ARMS:
        a0, a150 = acc(m, arm, 0), acc(m, arm, 150)
        a = agg.get((m, arm, 150)) or agg.get((m, arm, 0))
        if not a:
            continue
        d = (a150 - a0) if (a0 is not None and a150 is not None) else None
        p0 = f"{a0:.0%}" if a0 is not None else "—"
        p1 = f"{a150:.0%}" if a150 is not None else "—"
        dd = f"{d:+.0%}" if d is not None else "—"
        print(f"| {m} | {SIZE[m]} | {arm} | {p0} | {p1} | {dd} | "
              f"{a['hit']/a['n']:.0%} | {a['lat']/a['n']:.1f} | {a['tok']//a['n']} |")

print("\n## Headline comparisons (grok-4.5 judged)\n")
for m in MODELS:
    for pol in (0, 150):
        wo, wi = acc(m, "naive_rag", pol), acc(m, "context_runtime", pol)
        if wo is not None and wi is not None:
            print(f"- {m:18s} @pol{pol:<3d}: without CR {wo:.0%}  →  with CR {wi:.0%}  ({wi-wo:+.0%})")

# the pitch question: small model + CR vs big model without CR, under pollution
qc = acc("qwen3-coder-next", "context_runtime", 150)
dn = acc("deepseek-v4-flash", "naive_rag", 150)
nc = acc("nemotron-super", "context_runtime", 150)
if qc is not None and dn is not None:
    print(f"\n- PITCH @pol150: Qwen-Coder(80B)+CR {qc:.0%}  vs  DeepSeek(284B) no-CR {dn:.0%}  "
          f"→ {'small+CR WINS' if qc > dn else 'big model wins'}")

# plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    styles = {"full_dump": (":", "o"), "naive_rag": ("--", "s"), "context_runtime": ("-", "D")}
    labels = {"full_dump": "native (full-dump)", "naive_rag": "without CR (naive RAG)",
              "context_runtime": "with Context Runtime"}
    for ax, m in zip(axes, MODELS):
        for arm in ARMS:
            pts = [(pol, acc(m, arm, pol)) for pol in (0, 150) if acc(m, arm, pol) is not None]
            if len(pts) < 2:
                continue
            xs, ys = zip(*pts)
            ls, mk = styles[arm]
            ax.plot(xs, ys, ls, marker=mk, label=labels[arm], linewidth=2)
        ax.set_title(f"{m}\n({SIZE[m]})", fontsize=10)
        ax.set_xlabel("context pollution (distractor docs)")
        ax.set_ylim(0.5, 1.0); ax.set_xticks([0, 150]); ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("answer accuracy (grok-4.5 judged)")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Can Context Runtime beat a bigger model? — LiveRAG, 40 Q, grok-4.5 judged", fontsize=12)
    fig.tight_layout()
    fig.savefig("results/full/crossover.png", dpi=130)
    print("\nwrote results/full/crossover.png")
except Exception as e:
    print(f"(plot skipped: {e})")
