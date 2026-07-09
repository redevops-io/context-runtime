#!/usr/bin/env python3
"""Generate grounded Russian QA from the nutrients corpus (anti-parametric).

Sample substantive passages, ask grok-4.5 to write ONE specific factual question whose
answer is stated in that passage (answerable only from it), + the answer. Gold evidence =
the source chunk. Output → .nutridata/qa_ru.jsonl for the benchmark loader.

  XAI_API_KEY=... python scripts/qa_gen_ru.py --n 80
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

from context_runtime.adapters.model_litellm import Tier
from context_runtime.adapters.model_openai import OpenAICompatibleModel
from context_runtime.types import ModelRequest

CORPUS = ".nutridata/corpus_ru.jsonl"
OUT = ".nutridata/qa_ru.jsonl"
SYS = (
    "Ты составляешь вопросы для проверки понимания текста. По данному ОТРЫВКУ придумай ОДИН "
    "конкретный фактический вопрос, ответ на который явно содержится в отрывке и не может быть "
    "угадан из общих знаний (конкретные факты, числа, определения, рекомендации). Верни СТРОГО "
    'JSON: {"question": "...", "answer": "..."} — коротко, по-русски.'
)


def chunk(text, size=1200, overlap=150):
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def seeded_sort(key, items):
    return sorted(items, key=lambda x: hashlib.sha1(f"{key}:{x[0]}".encode()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--model", default="grok-4.5")
    ap.add_argument("--base-url", default="https://api.x.ai/v1")
    args = ap.parse_args()
    key = os.environ.get("XAI_API_KEY") or sys.exit("XAI_API_KEY not set")
    client = OpenAICompatibleModel([Tier(name="chat", model=args.model,
                                         base_url=args.base_url, api_key=key)], timeout=90)

    docs = [json.loads(l) for l in open(CORPUS)]
    # candidate chunks: PDF passages, 500-1200 chars (substantive), tagged with doc_id + idx
    cands = []
    for d in docs:
        if d["source"] != "pdf":
            continue
        for ci, c in enumerate(chunk(d["text"])):
            if 500 <= len(c) <= 1300 and len(re.findall(r"[А-Яа-я]", c)) > 200:
                cands.append((f"{d['doc_id']}#c{ci}", d["doc_id"], c))
    cands = seeded_sort("qa2026", cands)
    print(f"{len(cands)} candidate passages; generating {args.n} QA", file=sys.stderr)

    out, tried = [], 0
    for cid, doc_id, text in cands:
        if len(out) >= args.n:
            break
        tried += 1
        user = f"ОТРЫВОК:\n{text}\n\nВопрос+ответ (JSON):"
        try:
            r = client.complete(ModelRequest(messages=({"role": "user", "content": user},),
                                             system=SYS, max_tokens=400, capability="draft")).text
            m = re.search(r"\{.*\}", r, re.S)
            qa = json.loads(m.group()) if m else None
        except Exception:
            qa = None
        if qa and qa.get("question") and qa.get("answer"):
            out.append({"id": f"ru{len(out):03d}", "question": qa["question"].strip(),
                        "answer": str(qa["answer"]).strip(), "gold_doc": doc_id,
                        "gold_chunk_id": cid, "gold_text": text})
        if tried % 20 == 0:
            print(f"  {len(out)}/{args.n} generated ({tried} tried)", file=sys.stderr)
    with open(OUT, "w", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"DONE: {len(out)} QA -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
