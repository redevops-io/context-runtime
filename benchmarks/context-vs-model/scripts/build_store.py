#!/usr/bin/env python3
"""One-time: embed the FinanceBench corpus into a DuckDB store for redevops-rag.

    PYTHONPATH=<repo>:.. python scripts/build_store.py --store /path/financebench.duckdb

Embeds every corpus passage (document_id = filing) so retrieval can be scoped to a
subset of filings for the pollution axis. GPU-accelerated if the embedder finds one.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

if os.environ.get("DATASET") == "nutri":
    from harness import data_nutri as data
else:
    from harness import data
from harness.rag_store import build_store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--no-reranker", action="store_true",
                    help="skip loading the cross-encoder (smaller; disables the rerank arm)")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    root = data.default_root()
    corpus = data.load_corpus(root)
    print(f"corpus: {len(corpus.passages)} passages / {len(corpus.by_doc)} docs / "
          f"{len(corpus.companies)} companies", file=sys.stderr)
    t0 = time.time()

    def prog(done, total):
        if done % 4096 == 0 or done == total:
            el = time.time() - t0
            print(f"  embedded {done}/{total}  ({el:.0f}s, {done/max(el,1):.0f}/s)", file=sys.stderr)

    fb = build_store(corpus, args.store, use_reranker=not args.no_reranker,
                     batch=args.batch, progress=prog)
    print(f"done: {fb.count} chunks → {args.store}  ({time.time()-t0:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
