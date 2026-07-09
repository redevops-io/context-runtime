#!/usr/bin/env python3
"""Driver — run one served model through the 3 arms across a pollution sweep on
FinanceBench, over the REAL redevops-rag store, scoring all six axes. Resumable.

Reasoning is a benchmark axis: pass --reasoning on|off (the model must be *served* to
match — llama.cpp `--reasoning off` vs on). Each row is tagged so on/off are comparable.

  PYTHONPATH=<repo>:. python -m harness.run \
      --model-name qwen3.6 --base-url http://127.0.0.1:8080/v1 --model Qwen3.6-35B-A3B \
      --store /cache/bench/financebench.duckdb --reasoning off \
      --pollution 0,2,8 --out results/qwen_off.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import data, grader, metrics, model as modelmod
from .arms import run_arm
from .rag_store import FinanceBenchStore
from .tuner import ContextRuntimeTuner

ARMS = ("full_dump", "naive_rag", "context_runtime")


def _done_keys(path: str) -> set:
    keys = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    keys.add((r["model_name"], r["reasoning"], r["arm"], r["pollution"], r["qid"]))
                except Exception:
                    continue
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser(description="Context-vs-model FinanceBench driver (real redevops-rag)")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="sk-noauth")
    ap.add_argument("--store", required=True, help="path to the prebuilt DuckDB embedding store")
    ap.add_argument("--reasoning", choices=("on", "off"), default="off",
                    help="tag + token budget; must match how the model is served")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--pollution", default="0,2,8")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-context-tokens", type=int, default=6000)
    ap.add_argument("--max-answer-tokens", type=int, default=0,
                    help="0 = auto (1024 if reasoning on, else 384)")
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-key", default=os.environ.get("OPENAI_API_KEY", "sk-noauth"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args(argv)

    max_ans = args.max_answer_tokens or (1024 if args.reasoning == "on" else 384)
    root = data.default_root()
    questions = data.load_questions(root)
    print("▸ loading corpus + store …", file=sys.stderr)
    corpus = data.load_corpus(root)
    doc_set = set(corpus.by_doc)
    questions = [q for q in questions if q.gold_docs & doc_set]   # gold docs present in corpus
    if args.limit:
        questions = questions[: args.limit]
    store = FinanceBenchStore(args.store)
    print(f"  corpus {len(corpus.passages)} passages / {len(corpus.by_doc)} docs; "
          f"store {store.count} chunks; {len(questions)} questions", file=sys.stderr)

    client = modelmod.make_client(args.base_url, args.model, api_key=args.api_key, timeout=args.timeout)
    judge_chat = None
    if args.judge_base_url and args.judge_model:
        jc = modelmod.make_client(args.judge_base_url, args.judge_model, api_key=args.judge_key)
        # reasoning judges (gpt-5.5) spend ~25-30 tokens thinking before the verdict; 6 → empty
        judge_chat = modelmod.make_chat(jc, max_tokens=256)

    arms = [a for a in args.arms.split(",") if a in ARMS]
    levels = [int(x) for x in args.pollution.split(",")]
    tuner = ContextRuntimeTuner()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = _done_keys(args.out)
    n_new = 0
    with open(args.out, "a") as out:
        for qi, q in enumerate(questions):
            for level in levels:
                for arm_name in arms:
                    if (args.model_name, args.reasoning, arm_name, level, q.id) in done:
                        continue
                    ar = run_arm(arm_name, store, corpus, q, level,
                                 max_tokens=args.max_context_tokens, tuner=tuner)
                    try:
                        resp = modelmod.answer(client, q, ar.context, max_tokens=max_ans)
                    except Exception as e:  # noqa: BLE001
                        resp = {"text": f"__ERROR__ {type(e).__name__}: {e}", "prompt_tokens": 0,
                                "completion_tokens": 0, "latency_s": 0.0}
                    g = grader.grade(q, resp["text"], judge_chat=judge_chat)
                    rm = metrics.retrieval_metrics(ar.retrieved, q)
                    poll = metrics.pollution_fraction(ar.retrieved, q)
                    if arm_name == "context_runtime" and ar.config is not None:
                        quality = 1.0 if g["correct"] else 0.5 * (rm["hit"] or 0.0)
                        tuner.record(q, ar.config, quality)
                    out.write(json.dumps({
                        "model_name": args.model_name, "model": args.model, "reasoning": args.reasoning,
                        "arm": arm_name, "pollution": level, "qid": q.id, "qtype": q.qtype,
                        "difficulty": q.difficulty, "correct": g["correct"], "grade_method": g["method"],
                        "is_numeric": q.is_numeric, "retrieval": rm, "pollution_frac": poll,
                        "prompt_tokens": resp["prompt_tokens"], "completion_tokens": resp["completion_tokens"],
                        "latency_s": round(resp["latency_s"], 3), "decision": ar.decision,
                        # self-contained for an off-box judge pass (no dataset access needed there)
                        "question": q.question, "gold_answer": q.answer,
                        "answer": resp["text"], "answer_preview": resp["text"][:160],
                    }) + "\n")
                    out.flush()
                    n_new += 1
            if (qi + 1) % 10 == 0:
                print(f"  {qi+1}/{len(questions)} q (+{n_new})  policy={tuner.policy()}", file=sys.stderr)
    print(f"done: +{n_new} rows → {args.out}\nCR policy: {tuner.policy()}", file=sys.stderr)


if __name__ == "__main__":
    main()
