#!/usr/bin/env python3
"""SimGraph vs HippoRAG — graph-retriever A/B on MuSiQue (supporting-paragraph recall@k).

Settles the parallel to the temporal decision. For temporal, the *simple* non-lossy approach
(document + timestamp) beat the *heavy* LLM engine (Graphiti). This asks the graph analog: does the
dependency-free ``SimGraphRetriever`` (2-hop term-spreading ≈ Personalized PageRank, no LLM OpenIE)
match the heavy ``HippoRAGRetriever``'s recall? Both are non-lossy — they return raw passages and use
the graph only to rank/traverse — so this measures whether the *learned* entity graph earns its cost
(the heavy install + per-corpus LLM OpenIE extraction).

Recall is engine-only and answer-model-independent: for each MuSiQue item we index its paragraphs,
retrieve top-k, and score the fraction of ``is_supporting`` paragraphs recovered (matched by passage
text, so the two chunk-id schemes don't matter). MuSiQue is per-question (each item ships its own
~20-paragraph corpus), so both retrievers are (re)built per item.

Usage:
  python benchmarks/graph_compare.py --demo          # smoke-test wiring (SimGraph only, no data)
  python benchmarks/graph_compare.py --data musique_ans_dev.jsonl --k 4 --limit 200 --which simgraph
  python benchmarks/graph_compare.py --data musique_ans_dev.jsonl --k 4 --which both \
      --hr-llm-base-url http://127.0.0.1:8000/v1 --hr-llm-model Qwen3.6-27B \
      --hr-embed-model BAAI/bge-large-en-v1.5 --hr-save-dir /cache/bench/hr_compare
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

# Run standalone (`python benchmarks/graph_compare.py`) or installed — put the bench root (the
# vendored `context_runtime` package) on the path either way.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_runtime.adapters.store_hipporag import SimGraphRetriever  # noqa: E402

# A 2-hop demo item: the answer passage (0) shares no query terms — it surfaces ONLY via the bridge
# entity "Lambeau" activated by the hop-0 passage (1). Exercises the multi-hop path single-hop can't.
DEMO = [{
    "question": "What team did the person born in Belgium found?",
    "texts": [
        "Curly Lambeau founded the Green Bay Packers.",   # 0: answer — bridge-only (no query overlap)
        "Curly Lambeau was born in Belgium.",             # 1: hop-0 (born, belgium)
        "The Chicago Bears play in Illinois.",            # 2: distractor
        "Belgium is a country in western Europe.",        # 3: distractor (shares 'belgium', not the bridge)
    ],
    "supporting": {0, 1},
}]


def load_musique(path: str, limit: int) -> list[dict]:
    """Load MuSiQue-Ans JSONL: each line has {question, paragraphs:[{paragraph_text, is_supporting}]}."""
    items: list[dict] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            paras = d.get("paragraphs") or []
            texts = [(p.get("paragraph_text") or p.get("text") or "") for p in paras]
            supporting = {i for i, p in enumerate(paras) if p.get("is_supporting")}
            if texts and supporting and d.get("question"):
                items.append({"question": d["question"], "texts": texts, "supporting": supporting})
            if limit and len(items) >= limit:
                break
    return items


def recall_at_k(retriever, item: dict, k: int) -> float:
    """Fraction of supporting paragraphs recovered in the top-k (matched by passage text)."""
    text_to_idx = {t: i for i, t in enumerate(item["texts"])}
    hits = retriever.search(item["question"], k=k, method="graph")
    got = {text_to_idx[h.text] for h in hits if h.text in text_to_idx}
    sup = item["supporting"]
    return len(got & sup) / len(sup) if sup else 0.0


def build_simgraph(texts: list[str]) -> SimGraphRetriever:
    return SimGraphRetriever([{"chunk_id": str(i), "filename": f"p{i}", "text": t} for i, t in enumerate(texts)])


def build_hipporag(texts: list[str], args, n: int):
    from context_runtime.adapters.store_hipporag import HippoRAGRetriever
    hr = HippoRAGRetriever(
        save_dir=f"{args.hr_save_dir}/item{n}", llm_model_name=args.hr_llm_model,
        embedding_model_name=args.hr_embed_model, llm_base_url=args.hr_llm_base_url,
        embedding_base_url=args.hr_embed_base_url, llm_api_key=args.hr_llm_api_key,
    )
    hr.index(list(texts))  # per-item corpus (LLM OpenIE builds the entity graph — the cost we're pricing)
    return hr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="MuSiQue-Ans JSONL")
    ap.add_argument("--demo", action="store_true", help="run the built-in 2-hop fixture (SimGraph only)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="max items (0 = all)")
    ap.add_argument("--which", choices=("both", "simgraph", "hipporag"), default="both")
    ap.add_argument("--verbose", action="store_true", help="per-item recall")
    # HippoRAG endpoint config (served LLM for OpenIE + embedder)
    ap.add_argument("--hr-save-dir", default="/tmp/hr_compare")
    ap.add_argument("--hr-llm-model", default="gpt-5-mini")
    ap.add_argument("--hr-llm-base-url", default=None)
    ap.add_argument("--hr-embed-model", default="nvidia/NV-Embed-v2")
    ap.add_argument("--hr-embed-base-url", default=None)
    ap.add_argument("--hr-llm-api-key", default=None)
    args = ap.parse_args()

    if args.demo:
        items, args.which = DEMO, "simgraph"
    elif not args.data:
        ap.error("--data is required (or use --demo)")
    else:
        items = load_musique(args.data, args.limit)
    if not items:
        print("no items loaded", file=sys.stderr)
        return 1

    want_sg = args.which in ("both", "simgraph")
    want_hr = args.which in ("both", "hipporag")
    sg_recalls: list[float] = []
    hr_recalls: list[float] = []
    paired: list[tuple[float, float]] = []  # (sg, hr) on items where BOTH ran — for a paired delta
    t0 = time.time()

    for n, it in enumerate(items):
        sg_r = recall_at_k(build_simgraph(it["texts"]), it, args.k) if want_sg else None
        hr_r = None
        if want_hr:
            try:
                hr_r = recall_at_k(build_hipporag(it["texts"], args, n), it, args.k)
            except Exception as e:  # noqa: BLE001 — a per-item HippoRAG failure shouldn't abort the run
                print(f"  [item {n}] HippoRAG failed: {e}", file=sys.stderr)
        if sg_r is not None:
            sg_recalls.append(sg_r)
        if hr_r is not None:
            hr_recalls.append(hr_r)
        if sg_r is not None and hr_r is not None:
            paired.append((sg_r, hr_r))
        if args.verbose:
            print(f"  item {n:>4}  simgraph={sg_r}  hipporag={hr_r}")

    def mean(xs: list[float]) -> float:
        return round(statistics.fmean(xs), 4) if xs else float("nan")

    print(f"\n== SimGraph vs HippoRAG — MuSiQue recall@{args.k} ({len(items)} items, {time.time()-t0:.1f}s) ==")
    if want_sg:
        print(f"  SimGraph  mean recall@{args.k} = {mean(sg_recalls)}   (n={len(sg_recalls)})")
    if want_hr:
        print(f"  HippoRAG  mean recall@{args.k} = {mean(hr_recalls)}   (n={len(hr_recalls)})")
    if paired:
        delta = statistics.fmean(s - h for s, h in paired)
        print(f"  paired Δ(SimGraph − HippoRAG) = {round(delta, 4)}  over n={len(paired)}  "
              f"→ {'SimGraph suffices (drop the heavy engine)' if delta >= -0.02 else 'HippoRAG earns its cost'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
