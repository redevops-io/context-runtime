# Release 0.4.1: control-plane authentication and ingestion, MCP stdio environment

Security release. It changes defaults, so read the migration section before upgrading.

## Who should upgrade

Everyone running `>= 0.1.0, <= 0.4.0`, in particular anyone who runs the control plane (`context_runtime.control_plane.app`, the image built by the
`Dockerfile`) on a network others can reach, and anyone who starts stdio MCP servers through `MCPClient.stdio` from a process that holds credentials in
its environment.

## What changed

### Control plane

- **Authentication fails closed.** With neither `CONTEXT_RUNTIME_API_KEY` nor `AGENTIC_OS_API_KEY` set, every route except `GET /health` answers `503`. With a key
  set, a missing or wrong credential is `401`. Before, the mutating routes (`/dispatch`, `/approvals/{id}/{decision}`, `/up`, `/down`, `/agent/run`,
  `/agent/outcome`, `/librechat/ingest`, ...) accepted anyone when no key was set.
- **Read routes need a credential.** `/approvals`, `/jobs/{id}`, `/status`, `/api/fleet`, `/notify/status`, `/agent/policy/{module}`, `/modules`, the
  dashboards (`/`, `/overview`, `/m/*`), the `/vibex/*` reads, `/librechat/retrieve|compare|explain|image|policy` and `/v1/*` were open with or without a
  key. They now require one.
- **Access classes.** Every route is `public`, `reader` or `admin` (table in [SECURITY.md](../SECURITY.md)). `GET /health` is the only public route and answers
  `{"status": "ok"}`. A route that is not classified is refused, and a test compares the live route table with the expected classes.
- **Tenant keys.** `CR_TENANT_KEYS` keeps its meaning (a key bound to tenants, `*` for all). A tenant key now works on the `reader` routes only. The control-plane
  key works on all routes; to choose a non-default tenant it must also be listed in `CR_TENANT_KEYS` (for example `ADMIN_KEY=*`), as before.
  `CR_ALLOW_ANONYMOUS_TENANTS=1` lets every authenticated caller use every tenant and does not admit callers without a credential.
- **Ingestion is limited to a corpus root.** `POST /librechat/ingest` indexes only inside `CONTEXT_RUNTIME_CORPUS_ROOT`, by identifier
  (`{"corpus": "name"}`) or by a path relative to the root. No root configured: ingestion is disabled (`503`). Absolute paths, `..`, hidden names, backslashes,
  `%` and control characters are `400`; a directory that resolves outside the root, and a corpus that contains a symlink to somewhere outside itself, are `404`.
- **Constant-time comparison.** The control-plane key and tenant keys are compared with `hmac.compare_digest`.
- **Documentation routes.** `/docs`, `/redoc` and `/openapi.json` are not served.
- **Container image.** The `Dockerfile` installs `agent-harness` at the commit `pyproject.toml` pins and checks the archive's sha256; before, the image of `0.4.0` failed at
  start with `ModuleNotFoundError: agent_harness`. `deploy/proxmox-demo/Dockerfile.contextruntime` has the same fix.
- **Deployment templates.** `deploy/librechat` and `deploy/financebench` no longer default to a known key. `deploy/proxmox-demo` requires `CONTEXT_RUNTIME_API_KEY`
  and `CR_PUBLIC_READ_KEY`; its proxy is an allowlist: it forwards only the reader routes (below), adds the read-only tenant key, and answers every other
  `/librechat/*` and `/v1/*` path with 404.

### MCP client

- `MCPClient.stdio` starts the server with an allowlisted environment, not a copy of the parent's (merged on `main` before this release: commit `d0440b3`, pull
  request #67, and its guard test, pull request #68). `inherit_env=True` or `CONTEXT_RUNTIME_MCP_INHERIT_ENV=1` restores the previous behaviour and logs a
  warning. The allowlist is documented in [SECURITY.md](../SECURITY.md).

### Also in this release (already on `main`, no behaviour change)

- Version declared as `0.4.0` in `pyproject.toml` and `context_runtime.__version__` (the `v0.4.0` tag itself still declares `0.2.0`), now `0.4.1`.
- Licence wording in `README.md` and a comment and docstring (no code change).

## Migration

1. **Set a control-plane key** before starting: `CONTEXT_RUNTIME_API_KEY=<random value>` (for example `openssl rand -hex 24`). Without it the service starts and
   answers `503` everywhere but `/health`.
2. **Send the key** from every client: `X-API-Key: <key>` or `Authorization: Bearer <key>`. This includes dashboards and scripts that read `/approvals`, `/status`,
   `/api/fleet` and the like. Browsers cannot add a header on their own: use a proxy or client that adds it.
3. **LibreChat and other OpenAI-compatible clients** set the key as their endpoint `apiKey` (they send it as a Bearer token). If you serve several corpus tenants,
   give the client a tenant key bound to them in `CR_TENANT_KEYS` instead of the control-plane key, for example `CR_TENANT_KEYS="<read key>=*"`.
4. **Public read-only front ends** (a browser panel that calls `/librechat/compare`): put a proxy in front of the control plane that forwards only the reader routes and
   adds a tenant key bound to the public corpora (`CR_TENANT_KEYS="<read key>=*"`); do not ship a key in page code and never give the proxy the control-plane key.
   `deploy/proxmox-demo/Caddyfile` is an example (an allowlist; anything not listed is 404 at the proxy). What the demo needs:

   | Demo feature | Route | Class | Credential the proxy adds |
   |---|---|---|---|
   | LibreChat model list and chat (LibreChat sends the key itself, server to server) | `GET /v1/models`, `POST /v1/chat/completions` | reader | the read key (LibreChat's endpoint `apiKey`) |
   | Query Board panel (browser, same origin) | `POST /librechat/compare` | reader | the read key |
   | EXPLAIN page | `POST /librechat/explain` | reader | the read key |
   | Retrieval and policy views, image thumbnails | `POST /librechat/retrieve`, `GET /librechat/policy`, `GET /librechat/image` | reader | the read key |
   | Learning from judge scores, explicit feedback, the self-contained chat loop | `/librechat/judge`, `/librechat/feedback`, `/librechat/chat` | admin | none: not exposed on the public demo. The shim already learns from implicit signals in `/v1/chat/completions`, so the demo's learning continues. |
   | Corpus ingestion | `/librechat/ingest` | admin | none: never exposed through the proxy; run it directly against the control plane with the administrative key |
   | Fleet dashboards, approvals, `/vibex/*` | admin routes | admin | none: not exposed on the public demo (they are not part of it) |
5. **Ingestion**: set `CONTEXT_RUNTIME_CORPUS_ROOT` to the directory that holds your corpora and call `/librechat/ingest` with `{"corpus": "<directory name>"}`.
   Clients that sent an absolute path now send the path relative to the root. Corpora configured with `CR_CORPUS_DIR`, `CR_TENANTS` or `CR_SHARDS` are read at
   startup from the paths you configure and are not affected.
6. **Rebuild the image** (`docker build .`). If you built your own image from the `0.4.0` `Dockerfile`, add the `agent-harness` install lines from the new one.
7. **Health checks** that parsed `modules` from `/health` should check the status code only.
8. **Stdio MCP servers** that relied on an inherited variable need it in `MCPClient.stdio(cmd, env={...})` (or `inherit_env=True`).
9. Keep the service on a trusted network or `127.0.0.1` as well.

```bash
export CONTEXT_RUNTIME_API_KEY="<random value>"
export CONTEXT_RUNTIME_CORPUS_ROOT=/srv/corpora
curl -s http://127.0.0.1:8080/health
curl -s -H "X-API-Key: $CONTEXT_RUNTIME_API_KEY" http://127.0.0.1:8080/approvals
curl -s -H "X-API-Key: $CONTEXT_RUNTIME_API_KEY" -H 'content-type: application/json' \
  -d '{"corpus": "finance"}' http://127.0.0.1:8080/librechat/ingest
```

## Checking whether a deployment was exposed

On a deployment of `<= 0.4.0` that is reachable by others, a `200` answer to `GET /approvals` or to `POST /librechat/ingest` without an `X-API-Key` header means the
routes answered anonymous callers. Rotate any credential that appeared in approval text or job results, and review `GET /librechat/policy` and the indexed
corpora for content you did not add.

## What was verified

- Tested: the route table (every route has an access class; one public route); `503` with no key, `401` with a missing or wrong key and success with the key on every
  route; anonymous reads of the formerly open routes; unauthorised writes and approval decisions; tenant keys on the `reader` routes and refused on the `admin`
  routes; ingestion outside the corpus root, traversal, absolute paths, encoded and Windows-style separators, and symlinked files and directories; the
  `Dockerfile` build and start on loopback (`/health` answers, a protected route is `503`, `401`, `200`); the MCP allowlist and `CONTEXT_RUNTIME_MCP_INHERIT_ENV=1`; the demo proxy: the control plane behind the unmodified `deploy/proxmox-demo/Caddyfile` (Caddy 2.11.7) on loopback, where a client without any credential gets the model list, retrieval, comparison, explanation, policy and chat, while ingestion, judging, feedback, the chat loop and every other control-plane route are refused at the proxy, and the control plane itself answers only `/health` to a caller without a credential.
- Not tested: a deployment behind your own proxy; the `deploy/proxmox-demo` LibreChat image (a stub stood in for it); hard links inside a corpus; a corpus directory that is modified while it is being ingested; Windows; the
  `deploy/proxmox-demo` compose stack as a whole; third-party MCP servers; timing differences of the old
  comparison (none was measured or claimed).

## What this release does not claim

- It is not an external audit or a third-party certification; the review that found these issues was AI-assisted and internal.
- It does not protect a deployment that exposes the credential, shares one key between untrusted clients, or places the service on an untrusted network without
  other controls.
- Existing indexed corpora, approvals and learned policies are not inspected or cleaned by the upgrade.

## Install

```bash
pip install "context-runtime @ git+https://github.com/redevops-io/context-runtime.git@v0.4.1"
python -c "import context_runtime; print(context_runtime.__version__)"
```
