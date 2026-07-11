#!/usr/bin/env python3
"""Build the MedicalBench corpus: PubMedQA abstracts → normalized passages.

    ./download.sh && python3 deploy/medical/build_corpus.py

PubMedQA's `ori_pqal.json` is a dict keyed by PMID; each entry has QUESTION, CONTEXTS (the
abstract's labeled sections), LABELS, LONG_ANSWER and final_decision. We emit one normalized
`.txt` passage per (PMID, section) into `.medical/corpus/` — the same pre-normalized shape the
FinanceBench corpus uses (CR_CORPUS_DIR ingests it directly, no PDF step) — plus `qa.jsonl`
(question / gold long answer / decision / gold PMID) so the corpus is probe-able.

The passages carry real clinical vocabulary (discharge, chronic/acute, balance, statement, …)
that collides with 10-K language — which is the point: in the Mixed model, coverage routing
keeps a medical query on the clinical shard instead of surfacing a 10-K's "discharge of
liability". Keeps the whitepaper's cross-domain-noise-22→0 result on the live demo path.
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MED = os.path.join(ROOT, ".medical")
SRC = os.path.join(MED, "pqal.json")
CORPUS = os.path.join(MED, "corpus")


def _slug(text: str, n: int = 48) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "section"


def main() -> None:
    if not os.path.isfile(SRC):
        sys.exit(f"missing {SRC} — run deploy/medical/download.sh first")
    data = json.load(open(SRC, encoding="utf-8"))
    os.makedirs(CORPUS, exist_ok=True)
    n_docs = n_passages = 0
    with open(os.path.join(MED, "qa.jsonl"), "w", encoding="utf-8") as qa:
        for pmid, rec in data.items():
            contexts = rec.get("CONTEXTS") or []
            labels = rec.get("LABELS") or []
            if not contexts:
                continue
            n_docs += 1
            for i, ctx in enumerate(contexts):
                label = labels[i] if i < len(labels) else ""
                # one passage file per abstract section; header carries the section label so a
                # lexical index keeps the clinical framing (RESULTS / METHODS / CONCLUSIONS …).
                header = f"PubMed {pmid} — {label}".strip(" —")
                text = f"{header}\n\n{ctx.strip()}\n"
                fn = f"pmid{pmid}_{i:02d}_{_slug(label or 'abstract')}.txt"
                with open(os.path.join(CORPUS, fn), "w", encoding="utf-8") as fh:
                    fh.write(text)
                n_passages += 1
            qa.write(json.dumps({
                "pmid": pmid,
                "question": rec.get("QUESTION", ""),
                "long_answer": rec.get("LONG_ANSWER", ""),
                "decision": rec.get("final_decision", ""),
            }, ensure_ascii=False) + "\n")
    print(f"=== MedicalBench corpus built ===")
    print(f"  {n_docs} abstracts → {n_passages} passages → {CORPUS}")
    print(f"  qa.jsonl → {os.path.join(MED, 'qa.jsonl')}")


if __name__ == "__main__":
    main()
