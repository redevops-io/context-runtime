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

### Changing the allowlist

The list above is deliberately small, and it is also meant to evolve: a server or platform may legitimately need another variable.
An addition is a documented security review, not a prohibition. In the pull request, state (1) which variable and which server or
platform needs it, (2) why passing it through the explicit `env` mapping is not enough, (3) that its name and value cannot carry a
credential (a name containing KEY, TOKEN, SECRET, PASSWORD or CREDENTIAL does not qualify; a value that may embed credentials is
filtered the way the proxy variables are), and (4) the test that shows it reaches the child. Then update `SAFE_ENV_VARS` and
`REVIEWED_SAFE_ENV_VARS` in `tests/test_mcp_env_guard.py` together; that test exists so that the list never changes unnoticed.
