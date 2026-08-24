#!/usr/bin/env bash
# Deploy the geospatial / zoning-intelligence demo to the proxmox host (geo.redevops.io).
#
# Stages a self-contained build context — the real Context Runtime geospatial capability + the harvested
# parcel sample — rsyncs it to the host, and builds + runs geo-api on :8098. No datastore.
set -euo pipefail
HOST="${GEO_HOST:-192.168.40.105}"
HOST_DIR="${GEO_HOST_DIR:-/projects/contextos/geo-demo}"
SRC="$(cd "$(dirname "$0")" && pwd)"          # .../contextos/deploy/geo-demo
REPO="$(cd "$SRC/../.." && pwd)"              # contextos repo root (has context_runtime)

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
echo "== stage build context (geospatial capability + harvested data) =="
rsync -aH --exclude __pycache__ --exclude '*.pyc' "$REPO/context_runtime" "$STAGE/"
cp "$SRC/app.py" "$SRC/requirements.txt" "$SRC/Dockerfile" "$SRC/geo-demo.compose.yml" "$STAGE/"
rsync -aH "$SRC/data" "$SRC/static" "$STAGE/"

echo "== sync -> $HOST:$HOST_DIR =="
ssh "root@$HOST" "mkdir -p $HOST_DIR"
rsync -aH --delete --exclude '.git' "$STAGE"/ "root@$HOST:$HOST_DIR/"

ssh "root@$HOST" GEO_HOST_DIR="$HOST_DIR" 'bash -s' <<'REMOTE'
set -e
cd "$GEO_HOST_DIR"
echo "== build + up =="
GEO_EDGE_PORT=8098 docker compose -p geo-demo -f geo-demo.compose.yml up -d --build
for i in $(seq 1 30); do
  curl -s --max-time 6 http://127.0.0.1:8098/healthz 2>/dev/null | grep -q '"ok":true' && { echo "healthy"; break; }
  sleep 2
done
curl -s http://127.0.0.1:8098/healthz; echo
REMOTE

echo
echo "== ingress: add geo.redevops.io -> 192.168.40.105:8098 to /main/cloudflared/config.yml,"
echo "   add the DNS CNAME in redevops.io's Cloudflare zone, then verify https://geo.redevops.io/"
