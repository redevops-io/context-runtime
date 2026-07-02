#!/usr/bin/env python3
"""Build the FinanceBench corpus: SEC filing PDFs → tables-preserved passages.

    ./download.sh && uv run --with pdfplumber deploy/financebench/build_corpus.py

Uses PdfTableExtractor (pdfplumber) so the numbers in financial statements survive as
pipe tables instead of being flattened — the "hard to ingest" half of the demo.
"""
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from context_runtime.ingest.pdf_tables import PdfTableExtractor  # noqa: E402
from context_runtime.ingest.pipeline import ingest_corpus  # noqa: E402
from context_runtime.ingest.quality import HeuristicQuality  # noqa: E402
from context_runtime.sources.local import LocalFolderSource  # noqa: E402

FB = os.path.join(ROOT, ".financebench")
t0 = time.time()
stats = ingest_corpus(
    LocalFolderSource(os.path.join(FB, "pdfs")),
    os.path.join(FB, "corpus"),
    extractor=PdfTableExtractor(), quality=HeuristicQuality(), chunk_chars=1200, verbose=True)
print("\n=== STATS ===", stats.as_dict())
print(f"elapsed {time.time() - t0:.0f}s")
