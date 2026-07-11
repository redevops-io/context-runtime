# Public demo deploy — LibreQB × Context Runtime on Proxmox

Ships the crown-jewel demo to `chat.redevops.io`: the LibreChat fork (with the **Libre
Query Board** panel) in **no-login guest mode**, plus the Context Runtime control plane
serving **three corpus-scoped models** from one process (multi-tenant, `CR_TENANTS`) — all
fronted by one Caddy so the browser talks to a single origin.

| Model (dropdown) | Corpus | Public source |
|---|---|---|
| Context Runtime · FinanceBench | real SEC 10-K pages (`/corpus`) | Patronus AI FinanceBench |
| Context Runtime · MedicalBench | biomedical abstracts (`/medical`) | PubMedQA |
| Context Runtime · Mixed | a heterogeneous shard mix of both, **coverage-routed** | — |

The Mixed model is the whitepaper's heterogeneous-source win, live: a finance question is
answered only from finance docs and a medical question only from medical docs (cross-domain
noise → 0). The chat shim (`/v1`) and the Query Board (`/librechat/compare`) both route by
the selected model id, so the transparency panel reflects the corpus you picked.

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
| `librechat.yaml` | one endpoint → `contextruntime:8092/v1`, three corpus modelSpecs |
| `cloudflared-ingress.snippet.yml` | the `chat.redevops.io` tunnel rule |
| `.env.example` | secrets template (render from Vault) |

## Runbook (you run these — they need your DNS / Vault / host access)

**1. DNS (manual, cross-account).** In **redevops.io's own** Cloudflare zone, add a proxied
CNAME `chat.redevops.io → <tunnel-id>.cfargotunnel.com` (same tunnel id as demo.redevops.io).
This can't be scripted here (redevops.io is a different CF account than the tunnel).

**2. cloudflared ingress.** Add the rule in `cloudflared-ingress.snippet.yml` to the
templated ingress in `ffmpeg-mcp-aws/ansible/local/deploy-cloudflared.yml` (next to the
demo.redevops.io rule), then re-run `deploy-cloudflared.yml`.

**3. Corpora onto the host.** Build BOTH corpora on the control node, then get them + the
contextos repo to the host (e.g. under `/projects/contextos`). The ansible `caddy` role does
the rsync; to do it by hand:
```bash
cd deploy/financebench && ./download.sh && uv run --with pdfplumber build_corpus.py   # → .financebench/corpus
cd ../medical        && ./download.sh && python3 build_corpus.py                       # → .medical/corpus (PubMedQA)
# then rsync .financebench/corpus + .medical/corpus + deploy/proxmox-demo to the host
```
The ansible path builds both automatically (`build-push.yml`) and mounts them RO at `/corpus`
and `/medical`; `CR_TENANTS` (in the rendered `.env`) wires the three models.

**4. LibreQB fork image.** Build the fork (LibreQB branch → the panel + DEMO_MODE + the
auto-detecting compare URL are baked in) and tag it `librechat-api:latest` on the host:
```bash
docker build -t librechat-api:latest /path/to/LibreChat   # LibreQB branch checked out
```

**5. Secrets.** Render `.env` from Vault (see `.env.example`): `JWT_*`, `CREDS_KEY/IV`,
`KIMI_*`. Keep `DEMO_MODE=true`, `ALLOW_REGISTRATION=false`.

**5b. AppArmor profile (this host only).** The host runs apparmor_parser 4.1, whose
default ABI breaks AF_UNIX socket creation under Docker's stock `docker-default`
(Python `socket.socketpair()` → `PermissionError [Errno 13]`; mongod → `open: Permission
denied`). The compose runs every service under `docker-contextos` instead — stock
docker-default confinement pinned to `abi <abi/3.0>,`. Install + load it once (survives
reboot via `apparmor.service`):
```bash
install -m0644 apparmor-docker-contextos.profile /etc/apparmor.d/docker-contextos
apparmor_parser -r -W /etc/apparmor.d/docker-contextos
```

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
- The three models are served by ONE control plane via `CR_TENANTS` (no second container).
  Each corpus is its own tenant with its own learned policy + Query Board. Add/remove a model
  by editing `CR_TENANTS` (`id=/path` for a single corpus, `id=shards(a:/p1,b:/p2)` for a
  coverage-routed mix) — no code change.
- Mixed-model routing cleanliness is tunable via `CR_ROUTE_MARGIN` (default `0.15`, coverage
  router). Lower = stricter single-domain routing if a borderline query leaks.
- **Per-tenant cross-language** (`CR_TENANT_LANGS`, opt-in): lists the CORPUS language(s) to
  translate an incoming query INTO before retrieval, per model. These corpora are English, so
  `context-runtime-finance=en | context-runtime-medical=en | context-runtime-mixed=en` lets a
  Spanish-speaking (e.g. Miami) customer query them. Empty ⇒ off. Each tenant can bridge a
  different language set (falls back to the global `CR_QUERY_LANGS`). Adds one cached translation
  call per new query.
- This stack is intentionally self-contained (own network + mongo) so it doesn't touch the
  agentic-os control plane; it only shares the host + the cloudflared tunnel.
