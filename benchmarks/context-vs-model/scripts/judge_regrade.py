#!/usr/bin/env python3
"""Judge pass — re-grade a results JSONL with a frontier judge (gpt-5.5), OFF the box.

The box produces answers + a numeric grade (no key needed there). This runs on a host
that has the OpenAI key in env, re-grades every row the numeric matcher marked wrong (the
judge catches prose answers + numeric equivalents), and reports judged vs numeric-only
accuracy. Writes an updated JSONL with `correct_judged`.

  OPENAI_API_KEY=... python scripts/judge_regrade.py results/x.jsonl [--model gpt-5.5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from harness import data, grader, model as modelmod

ap = argparse.ArgumentParser()
ap.add_argument("infile")
ap.add_argument("--model", default="gpt-5.5")
ap.add_argument("--base-url", default="https://api.openai.com/v1")
ap.add_argument("--out", default=None)
args = ap.parse_args()

key = os.environ.get("OPENAI_API_KEY")
if not key:
    sys.exit("OPENAI_API_KEY not in env")
rows = [json.loads(l) for l in open(args.infile)]
# self-contained rows (question/gold_answer stored) need no dataset; else fall back to it
need_ds = not all(("question" in r and "gold_answer" in r) for r in rows)
qmap = {q.id: q for q in data.load_questions()} if need_ds else {}
jc = modelmod.make_client(args.base_url, args.model, api_key=key)
judge = modelmod.make_chat(jc, max_tokens=256)

n_judged = 0
for r in rows:
    ans = r.get("answer") or r.get("answer_preview", "")
    q = qmap.get(r["qid"])
    question = r.get("question") or (q.question if q else "")
    gold = r.get("gold_answer") or (q.answer if q else "")
    if r["correct"]:
        r["correct_judged"] = True
        continue
    r["correct_judged"] = bool(question) and grader.judge_grade(judge, question, gold, ans)
    n_judged += 1

out = args.out or args.infile.replace(".jsonl", ".judged.jsonl")
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

def acc(field):
    g = defaultdict(lambda: [0, 0])
    for r in rows:
        if not r.get("is_numeric") and field == "correct":
            pass
        k = (r["arm"], r["pollution"]); g[k][0] += r[field]; g[k][1] += 1
    return g

print(f"judged {n_judged} rows (numeric-wrong) with {args.model} → {out}\n")
gn, gj = acc("correct"), acc("correct_judged")
print(f"{'arm':16s} {'pol':>3s}  numeric-only   judged")
for k in sorted(gn):
    print(f"{k[0]:16s} {k[1]:>3d}     {gn[k][0]/gn[k][1]:>5.0%}       {gj[k][0]/gj[k][1]:>5.0%}   (n={gn[k][1]})")
