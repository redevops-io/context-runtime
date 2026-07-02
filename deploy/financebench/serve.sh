#!/usr/bin/env bash
# Serve the FinanceBench corpus through a Context Runtime control plane on :8092 — the same
# port the LibreChat "Context Runtime (Python)" endpoint and the LibreQB panel use, so the
# chat and the Query Board both run over the financial filings.
#
#   export KIMI_API_KEY=... KIMI_BASE_URL=https://api.moonshot.ai/v1 KIMI_MODEL=kimi-k2.6
#   ./download.sh && ./build_corpus.py && ./serve.sh
#
# CR_EMBEDDINGS=1 wraps HybridRetriever in a HopRouter so all five methods (bm25/vector/
# hybrid/graph/community) run in the panel. English corpus → no CR_QUERY_LANGS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CORPUS="$ROOT/.financebench/corpus"
[ -d "$CORPUS" ] || { echo "corpus missing — run ./build_corpus.py first"; exit 1; }
fuser -k 8092/tcp 2>/dev/null || true; sleep 1

env PYTHONPATH="$ROOT" CR_CORPUS_DIR="$CORPUS" CR_EMBEDDINGS=1 \
  CR_UPSTREAM_BASE_URL="${KIMI_BASE_URL:-}" CR_UPSTREAM_API_KEY="${KIMI_API_KEY:-}" \
  CR_UPSTREAM_MODEL="${KIMI_MODEL:-}" CR_UPSTREAM_MAX_TOKENS=4096 CR_UPSTREAM_TEMPERATURE=1 \
  CONTEXT_RUNTIME_HOME=/tmp/financebench-home CONTEXT_RUNTIME_API_KEY="${CONTEXT_RUNTIME_API_KEY:-context-runtime}" \
  uv run --no-project --with fastapi --with uvicorn --with httpx --with pyyaml --with fastembed \
  uvicorn context_runtime.control_plane.app:app --host 0.0.0.0 --port 8092
