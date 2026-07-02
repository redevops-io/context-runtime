#!/usr/bin/env bash
# Download the FinanceBench demo data: the open-source Q&A + a subset of SEC filing PDFs.
# FinanceBench (Patronus AI, https://github.com/patronus-ai/financebench) — real 10-K/10-Q
# filings + expert Q&A; a recognized "RAG is hard here" benchmark (dense tables, numerical,
# multi-hop). Data lands in ../../.financebench/ (gitignored).
set -euo pipefail
DEST="$(cd "$(dirname "$0")/../.." && pwd)/.financebench"
RAW="https://raw.githubusercontent.com/patronus-ai/financebench/main"
mkdir -p "$DEST/pdfs"

echo "▸ Q&A + doc metadata"
curl -sL "$RAW/data/financebench_open_source.jsonl" -o "$DEST/qa.jsonl"
curl -sL "$RAW/data/financebench_document_information.jsonl" -o "$DEST/docs.jsonl"

# a 6-filing subset covering all three question types (metrics / domain / novel)
SUBSET=(AMD_2022_10K AMERICANEXPRESS_2022_10K BOEING_2022_10K PEPSICO_2022_10K 3M_2022_10K 3M_2023Q2_10Q)
echo "▸ ${#SUBSET[@]} SEC filing PDFs"
for doc in "${SUBSET[@]}"; do
  curl -sL "$RAW/pdfs/$doc.pdf" -o "$DEST/pdfs/$doc.pdf"
  head -c5 "$DEST/pdfs/$doc.pdf" | grep -q '%PDF' && echo "  ✓ $doc.pdf" || { echo "  ✗ $doc.pdf"; exit 1; }
done
echo "done → $DEST"
