#!/usr/bin/env bash
# Download the MedicalBench demo data — the public PubMedQA expert-labeled set.
#
# PubMedQA (Jin et al., EMNLP 2019, https://github.com/pubmedqa/pubmedqa) — 1 000 expert-
# annotated biomedical questions over real PubMed abstract passages, each with gold contexts
# and a long-form answer. It's the medical analogue of the FinanceBench half of this demo:
# real domain documents + expert Q&A, and a vocabulary (discharge, statement, balance,
# chronic/acute, …) that deliberately collides with financial 10-K language — which is what
# makes the heterogeneous-shards / coverage-routing win visible in the Mixed model.
#
# Data lands in ../../.medical/ (gitignored). Public, MIT-licensed; we ship the downloader,
# not the data — same policy as deploy/financebench/download.sh.
set -euo pipefail
DEST="$(cd "$(dirname "$0")/../.." && pwd)/.medical"
RAW="https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data"
mkdir -p "$DEST"

echo "▸ PubMedQA expert-labeled Q&A (ori_pqal.json, 1000 items)"
curl -sfL "$RAW/ori_pqal.json" -o "$DEST/pqal.json"
python3 - "$DEST/pqal.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert isinstance(d, dict) and d, "unexpected pqal.json shape"
print(f"  ✓ {len(d)} labeled questions")
PY
echo "done → $DEST/pqal.json   (now run build_corpus.py to build the passage corpus)"
