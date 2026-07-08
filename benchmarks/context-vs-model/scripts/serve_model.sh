#!/usr/bin/env bash
# Serve one GGUF model on CPU via llama.cpp for the benchmark.
#
#   scripts/serve_model.sh <gguf-first-shard> <served-name> [port] [ctx] [threads]
#
# Notes that matter for a FAIR run:
#  --reasoning off : disable thinking. These are reasoning models; on a hard question
#     with a long polluted context they otherwise spend the whole token budget in
#     <think> and never emit an answer (empty content). Off = straight to the answer,
#     comparable across models, and far cheaper on CPU. This is a SERVER default because
#     client-side enable_thinking=false is silently dropped by some stacks.
#  point -m at the FIRST shard of a multi-part GGUF; llama.cpp auto-loads the rest.
set -euo pipefail

GGUF="${1:?path to (first shard of) the .gguf}"
NAME="${2:?served-model-name}"
PORT="${3:-8080}"
CTX="${4:-8192}"
THREADS="${5:-96}"
BIN="${LLAMA_SERVER:-/cache/llama.cpp/build/bin/llama-server}"
LOG="${LOG:-/cache/logs/llama_${NAME}.log}"

pkill -f "bin/llama-server" 2>/dev/null || true      # specific pattern — won't match this shell
sleep 2
mkdir -p "$(dirname "$LOG")"
setsid nohup "$BIN" -m "$GGUF" --host 0.0.0.0 --port "$PORT" \
  --ctx-size "$CTX" --threads "$THREADS" --jinja --reasoning off \
  -a "$NAME" > "$LOG" 2>&1 < /dev/null &
echo "launched $NAME on :$PORT (log $LOG)"

for i in $(seq 1 200); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] \
    && { echo "READY after ~$((i*3))s"; exit 0; }
  sleep 3
done
echo "TIMED OUT waiting for health"; tail -8 "$LOG"; exit 1
