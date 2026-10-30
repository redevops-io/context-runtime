# Security

Report suspected vulnerabilities privately; do not open a public issue.

- Email **redevops@redevops.io** (the ReDevOps organisation contact), stating that it is a security report.

Include the commit or tag and a minimal reproduction. The default offline path makes no network calls; real
model and retrieval bindings are opt-in extras.

## MCP stdio servers: environment

`MCPClient.stdio` starts the server with an allowlisted environment, not a copy of yours. By default the child receives only
`PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `LANG`, `LANGUAGE`, `LC_*`, `TZ`, `TERM`, `TMPDIR`, `TEMP`, `TMP`, the Windows
system variables (`SYSTEMROOT`, `SYSTEMDRIVE`, `WINDIR`, `COMSPEC`, `PATHEXT`), `PYTHONPATH`, `VIRTUAL_ENV`, `PYTHONUTF8`,
`PYTHONIOENCODING`, `SSL_CERT_FILE`, `SSL_CERT_DIR`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE` and `NO_PROXY`, plus the proxy
variables (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`) when their value has no embedded credentials. API keys, tokens,
passwords and cloud or database settings in your environment do **not** reach the server.

- Give a server what it needs explicitly: `MCPClient.stdio(cmd, env={"MY_SERVICE_TOKEN": token})`. The mapping wins over the allowlist.
- To restore the previous behaviour (the child inherits the whole parent environment) use `MCPClient.stdio(cmd, inherit_env=True)`, or set
  `CONTEXT_RUNTIME_MCP_INHERIT_ENV=1` for the process. A warning is logged each time the full environment is passed on. An explicit
  `inherit_env=False` overrides the variable.
- The bundled `deploy_scan` server (`mount_deploy_scan`) is handed its own settings explicitly (`SCAN_ALLOWLIST`, `SCAN_LIVE`, the `SCAN_ZAP_*`, `SCAN_NMAP_PORTS` and `SCAN_NUCLEI_TEMPLATES` options, and `DOCKER_HOST` and `DOCKER_CONTEXT` for live scans), and nothing else from your environment.

## Control plane: authentication, tenant keys and ingestion

Purpose: the control plane (`context_runtime.control_plane.app`, the image built by the `Dockerfile`) serves approvals, jobs, fleet
information, learned policy, retrieval over indexed corpora and an OpenAI-compatible chat route. It answers only callers that
present a configured credential.

- **Fails closed.** With neither `CONTEXT_RUNTIME_API_KEY` nor `AGENTIC_OS_API_KEY` set (an empty or whitespace-only value counts as unset),
  every route except `GET /health` answers `503`. With a key set, a missing or wrong credential is `401`. The credential is sent as
  `X-API-Key` or `Authorization: Bearer`. Keys are compared with `hmac.compare_digest`.
- **One class per route.** Each route is `public` (only `GET /health`, which returns `{"status": "ok"}`), `reader` or `admin`. A route
  that is not classified is refused, and `tests/test_control_plane_access.py` fails when the route table and its expected classes differ.

| Class | Credential | Routes |
|---|---|---|
| public | none | `GET /health` |
| reader | the control-plane key, or a tenant key from `CR_TENANT_KEYS` | `POST /librechat/retrieve`, `/librechat/compare`, `/librechat/explain`; `GET /librechat/image`, `/librechat/policy`; `GET /v1/models`; `POST /v1/chat/completions` |
| admin | the control-plane key | every other route: `/`, `/overview`, `/modules`, `/status`, `/api/fleet`, `/notify/status`, `/m/*`, `/up`, `/down`, `/approvals` (read and decide), `/dispatch`, `/jobs/*`, `/agent/*`, `/vibex/*`, `/librechat/ingest`, `/librechat/judge`, `/librechat/feedback`, `/librechat/chat` |

- **Tenant keys.** `CR_TENANT_KEYS="key1=tenantA,tenantB;key2=*"` binds a credential to corpus tenants. A tenant key is accepted on `reader` routes only,
  for the tenants it is bound to (`403` for others); it never opens an `admin` route. The control-plane key is accepted on `reader` routes too; for
  choosing a tenant it counts as any credential that is not listed in `CR_TENANT_KEYS` and so reaches the default tenant only, unless it is also
  listed there (for example `ADMIN_KEY=*`) or `CR_ALLOW_ANONYMOUS_TENANTS=1` lets every authenticated caller use every tenant. That flag never
  lets a caller without a credential in. With only tenant keys configured, `admin` routes answer `503`.
- **Ingestion.** `POST /librechat/ingest` indexes only a directory inside `CONTEXT_RUNTIME_CORPUS_ROOT`. Name it with `{"corpus": "<directory under the root>"}`
  (older clients may send `{"path": "<path relative to the root>"}`). With no root configured, ingestion is disabled (`503`). Absolute paths, `..`, hidden
  names, backslashes, `%` and control characters are refused (`400`), and so is a directory that resolves outside the root or a corpus that contains a
  symlink (file or directory) pointing outside itself (`404`, the same answer as for a missing corpus).
- **Documentation routes.** `/docs`, `/redoc` and `/openapi.json` are not served.
- **Browsers.** The dashboards cannot send a header on their own; put the control plane behind a proxy or client that adds the credential. A
  read-only public front end can sit behind a proxy that forwards only the `reader` routes and adds a tenant key (see `deploy/proxmox-demo/Caddyfile`; `tests/test_demo_proxy.py` runs it against the control plane with Caddy). The proxy must never hold the control-plane key.
- Bind the service to a trusted network or to `127.0.0.1` as well; the credential is not a substitute for network placement.
