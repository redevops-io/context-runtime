#!/usr/bin/env bash
# Reproducible FinanceBench data fetch for the context-vs-model benchmark.
#
# Pulls the open-source FinanceBench Q&A + doc metadata and EVERY 10-K/10-Q PDF the 150
# questions reference (84 filings), then builds the passage corpus. All data is public
# (Patronus AI, https://github.com/patronus-ai/financebench, CC-BY-4.0). We ship this
# downloader, not the data.
#
# Result → ./.financebench/ : qa.jsonl, docs.jsonl, pdfs/*.pdf, corpus/*.txt, corpus.manifest.jsonl
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$HERE/.financebench}"
RAW="https://raw.githubusercontent.com/patronus-ai/financebench/main"
mkdir -p "$DEST/pdfs" "$DEST/corpus"

echo "▸ Q&A + doc metadata → $DEST"
curl -sfL "$RAW/data/financebench_open_source.jsonl" -o "$DEST/qa.jsonl"
curl -sfL "$RAW/data/financebench_document_information.jsonl" -o "$DEST/docs.jsonl"

echo "▸ resolving the filings the 150 questions reference"
mapfile -t DOCS < <(python3 -c "
import json
docs=sorted({json.loads(l)['doc_name'] for l in open('$DEST/qa.jsonl')})
print('\n'.join(docs))
")
echo "  ${#DOCS[@]} distinct filings"

echo "▸ fetching PDFs (skipping any already present)"
miss=0
for doc in "${DOCS[@]}"; do
  out="$DEST/pdfs/$doc.pdf"
  [ -s "$out" ] && head -c5 "$out" | grep -q '%PDF' && { echo "  · $doc (cached)"; continue; }
  if curl -sfL "$RAW/pdfs/$doc.pdf" -o "$out" && head -c5 "$out" | grep -q '%PDF'; then
    echo "  ✓ $doc"
  else
    echo "  ✗ $doc (not public — will be skipped in the run)"; rm -f "$out"; miss=$((miss+1))
  fi
done
[ "$miss" -gt 0 ] && echo "  ($miss filings unavailable; the harness benchmarks only covered questions)"

echo "▸ building the passage corpus (pdfplumber)"
# reuse contextos' build_corpus.py if present, else the local copy
BUILDER="$HERE/scripts/build_corpus.py"
[ -f "$BUILDER" ] || BUILDER="$(cd "$HERE/../.." && pwd)/deploy/financebench/build_corpus.py"
uv run --with pdfplumber python3 "$BUILDER" "$DEST" 2>/dev/null \
  || python3 "$BUILDER" "$DEST" "$DEST"

echo "done → $DEST"
python3 -c "
import json
docs={json.loads(l)['source'].rsplit('.pdf',1)[0] for l in open('$DEST/corpus.manifest.jsonl')}
qs=[json.loads(l) for l in open('$DEST/qa.jsonl')]
cov=sum(1 for q in qs if q['doc_name'] in docs)
print(f'corpus docs: {len(docs)} | questions covered: {cov}/{len(qs)}')
"
