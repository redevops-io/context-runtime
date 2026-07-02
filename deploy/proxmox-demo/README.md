# Public demo deploy — LibreQB × FinanceBench on Proxmox

Ships the crown-jewel demo to `chat.redevops.io`: the LibreChat fork (with the **Libre
Query Board** panel) in **no-login guest mode**, plus the Context Runtime control plane
over the FinanceBench corpus — all fronted by one Caddy so the browser talks to a single
origin.

```
Cloudflare edge → cloudflared (host, systemd) → caddy :8092 (host) ─┬─ /librechat/* /v1/* → contextruntime:8092
                                                                    └─ everything else    → librechat:3080
```

Same-origin means the panel's `/librechat/compare` fetch needs no CORS and no second
subdomain. Guests are per-browser (isolated chats, 1-week TTL) via `DEMO_MODE` in the fork.

## Files

| File | What |
|---|---|
| `chat-demo.compose.yml` | caddy + contextruntime + librechat + mongo (isolated network) |
| `Caddyfile` | one-origin routing |
| `Dockerfile.contextruntime` | control-plane image (Python) |
| `librechat.yaml` | one FinanceBench endpoint → `contextruntime:8092/v1` |
| `cloudflared-ingress.snippet.yml` | the `chat.redevops.io` tunnel rule |
| `.env.example` | secrets template (render from Vault) |

## Runbook (you run these — they need your DNS / Vault / host access)

**1. DNS (manual, cross-account).** In **redevops.io's own** Cloudflare zone, add a proxied
CNAME `chat.redevops.io → <tunnel-id>.cfargotunnel.com` (same tunnel id as demo.redevops.io).
This can't be scripted here (redevops.io is a different CF account than the tunnel).

**2. cloudflared ingress.** Add the rule in `cloudflared-ingress.snippet.yml` to the
templated ingress in `ffmpeg-mcp-aws/ansible/local/deploy-cloudflared.yml` (next to the
demo.redevops.io rule), then re-run `deploy-cloudflared.yml`.

**3. Corpus + repo onto the host.** Get the FinanceBench data + the contextos repo to the
host (e.g. under `/projects/contextos`):
```bash
cd deploy/financebench && ./download.sh && uv run --with pdfplumber build_corpus.py   # → .financebench/corpus
# then rsync the contextos repo (incl. .financebench/corpus + deploy/proxmox-demo) to the host
```

**4. LibreQB fork image.** Build the fork (LibreQB branch → the panel + DEMO_MODE + the
auto-detecting compare URL are baked in) and tag it `librechat-api:latest` on the host:
```bash
docker build -t librechat-api:latest /path/to/LibreChat   # LibreQB branch checked out
```

**5. Secrets.** Render `.env` from Vault (see `.env.example`): `JWT_*`, `CREDS_KEY/IV`,
`KIMI_*`. Keep `DEMO_MODE=true`, `ALLOW_REGISTRATION=false`.

**6. Up.**
```bash
cd /projects/contextos/deploy/proxmox-demo
docker compose --env-file .env -f chat-demo.compose.yml up -d --build
```

**7. Warm + verify.** The control plane embeds ~5k passages on the first `/compare`
(a couple of minutes); warm it, then open `https://chat.redevops.io`:
```bash
curl -s localhost:8092/librechat/compare -H 'content-type: application/json' \
  -d '{"request":"operating margin revenue","k":2}' -o /dev/null -w '%{http_code}\n'
```

## Notes

- English corpus → do **not** set `CR_QUERY_LANGS`.
- The control plane image installs `fastembed` (CPU MiniLM, no torch); ~a few GB RAM total
  with LibreChat + mongo. Fits alongside the agentic-os stack.
- To also expose the medical (Russian) demo, run a second control plane with that corpus +
  `CR_QUERY_LANGS=ru CR_QUERY_XLATE_MODEL=moonshot-v1-8k` and a second endpoint.
- This stack is intentionally self-contained (own network + mongo) so it doesn't touch the
  agentic-os control plane; it only shares the host + the cloudflared tunnel.
