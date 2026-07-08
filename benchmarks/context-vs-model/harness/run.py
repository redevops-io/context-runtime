#!/usr/bin/env python3
"""Driver — run one served model through the 3 arms across a pollution sweep on
FinanceBench, scoring all six axes. Resumable: each (model, arm, pollution, question)
row is appended to a JSONL and skipped on re-run.

Example (smoke):
  PYTHONPATH=<repo> python -m harness.run \
      --model-name qwen3.6 --base-url http://127.0.0.1:8080/v1 --model Qwen3.6-35B-A3B \
      --pollution 0,4 --limit 20 --out results/qwen_smoke.jsonl

Add a judge for prose grading: --judge-base-url ... --judge-model ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import data, grader, metrics, model as modelmod
from .arms import arm_context_runtime, arm_full_dump, arm_naive_rag, build_pool
from .tuner import ContextRuntimeTuner

ARMS = {"full_dump": arm_full_dump, "naive_rag": arm_naive_rag, "context_runtime": arm_context_runtime}


def _done_keys(path: str) -> set:
    keys = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add((r["model_name"], r["arm"], r["pollution"], r["qid"]))
                except Exception:
                    continue
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser(description="Context-vs-model FinanceBench benchmark driver")
    ap.add_argument("--model-name", required=True, help="label for this model in results")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="served-model-name at the endpoint")
    ap.add_argument("--api-key", default="sk-noauth")
    ap.add_argument("--arms", default="full_dump,naive_rag,context_runtime")
    ap.add_argument("--pollution", default="0,2,8", help="distractor-filing counts, comma-sep")
    ap.add_argument("--limit", type=int, default=0, help="use first N questions (0 = all 150)")
    ap.add_argument("--max-context-tokens", type=int, default=6000)
    ap.add_argument("--max-answer-tokens", type=int, default=384)
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-key", default=os.environ.get("OPENAI_API_KEY", "sk-noauth"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=240.0)
    args = ap.parse_args(argv)

    root = data.default_root()
    questions = data.load_questions(root)
    print(f"▸ loading corpus …", file=sys.stderr)
    corpus = data.load_corpus(root)
    print(f"  {len(corpus.passages)} passages / {len(corpus.by_doc)} docs / "
          f"{len(corpus.companies)} companies", file=sys.stderr)
    # a question is only benchmarkable if its target filing is in the loaded corpus
    covered = [q for q in questions if q.doc_name in corpus.by_doc]
    if len(covered) < len(questions):
        print(f"  {len(covered)}/{len(questions)} questions covered by loaded corpus "
              f"(run scripts/download_data.sh for full coverage)", file=sys.stderr)
    questions = covered
    if args.limit:
        questions = questions[: args.limit]

    client = modelmod.make_client(args.base_url, args.model, api_key=args.api_key, timeout=args.timeout)
    judge_chat = None
    if args.judge_base_url and args.judge_model:
        jc = modelmod.make_client(args.judge_base_url, args.judge_model, api_key=args.judge_key)
        judge_chat = modelmod.make_chat(jc, max_tokens=6)

    arms = [a for a in args.arms.split(",") if a in ARMS]
    levels = [int(x) for x in args.pollution.split(",")]
    tuner = ContextRuntimeTuner()      # shared → learns online across the run

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = _done_keys(args.out)
    n_new = 0
    with open(args.out, "a") as out:
        for qi, q in enumerate(questions):
            for level in levels:
                pool, distractors = build_pool(corpus, q, level)
                for arm_name in arms:
                    if (args.model_name, arm_name, level, q.id) in done:
                        continue
                    if arm_name == "context_runtime":
                        ar = arm_context_runtime(pool, q, max_tokens=args.max_context_tokens, tuner=tuner)
                    else:
                        ar = ARMS[arm_name](pool, q, max_tokens=args.max_context_tokens)
                    try:
                        resp = modelmod.answer(client, q, ar.context, max_tokens=args.max_answer_tokens)
                    except Exception as e:  # endpoint hiccup → record + continue
                        resp = {"text": f"__ERROR__ {type(e).__name__}: {e}", "prompt_tokens": 0,
                                "completion_tokens": 0, "latency_s": 0.0}
                    g = grader.grade(q, resp["text"], judge_chat=judge_chat)
                    rm = metrics.retrieval_metrics(ar.retrieved, q.gold_pages)
                    poll = metrics.pollution_fraction(ar.retrieved, q)
                    if arm_name == "context_runtime" and ar.config is not None:
                        quality = 1.0 if g["correct"] else 0.5 * (rm["hit"] or 0.0)
                        tuner.record(q, ar.config, quality)
                    row = {
                        "model_name": args.model_name, "model": args.model, "arm": arm_name,
                        "pollution": level, "qid": q.id, "qtype": q.qtype, "company": q.company,
                        "correct": g["correct"], "grade_method": g["method"],
                        "is_numeric": q.is_numeric, "n_distractors": len(distractors),
                        "retrieval": rm, "pollution_frac": poll,
                        "prompt_tokens": resp["prompt_tokens"], "completion_tokens": resp["completion_tokens"],
                        "latency_s": round(resp["latency_s"], 3), "decision": ar.decision,
                        "answer_preview": resp["text"][:160],
                    }
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    n_new += 1
            if (qi + 1) % 10 == 0:
                print(f"  {qi+1}/{len(questions)} questions  (+{n_new} rows)  "
                      f"policy={tuner.policy()}", file=sys.stderr)
    print(f"done: +{n_new} rows → {args.out}", file=sys.stderr)
    print(f"CR learned policy: {tuner.policy()}", file=sys.stderr)


if __name__ == "__main__":
    main()
