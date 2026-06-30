# Deploying the Context Runtime control plane

This replaces the retired `agentic-os` control-plane image. It serves the **same HTTP
API** the demo + cloudflared depend on (`/`, `/health`, `/status`, `/modules`,
`/m/<name>`, `/dispatch`, `/approvals`, `/agent/run`) on port **8080**, but the brain is
Context Runtime's `ModuleTenant` fleet — so `/status` reflects real tenants, and every
module shares one cost model that learns across the fleet.

## Run it

```bash
pip install -e ".[control-plane]"
uvicorn context_runtime.control_plane.app:app --host 0.0.0.0 --port 8080
# or: docker build -t context-runtime-cp . && docker run -p 8091:8080 context-runtime-cp
```

`CONTEXT_RUNTIME_API_KEY` gates the POST routes (falls back to `AGENTIC_OS_API_KEY` for
compatibility). `CONTEXT_RUNTIME_HOME` holds the approvals/audit log.

## Drop-in for `agentic-os-stack/integrated.compose.yml`

Replace the `control-plane:` service (which built from the now-gone
`/projects/agentic-os-src`) with this — same port mapping, network, volume, and
`depends_on`, only the build context + command change:

```yaml
  control-plane:
    build:
      context: /projects/context-runtime      # the context-runtime checkout
      dockerfile: Dockerfile
    command: uvicorn context_runtime.control_plane.app:app --host 0.0.0.0 --port 8080
    ports: ["8091:8080"]
    environment:
      - CONTEXT_RUNTIME_API_KEY=${CONTEXT_RUNTIME_API_KEY:-${AGENTIC_OS_API_KEY:-}}
      - CONTEXT_RUNTIME_HOME=/data
    volumes:
      - integrated_cp_data:/data
    networks: [agentic]
    restart: unless-stopped
    depends_on:
      - billing
      - control-tower
      - edge-sentinel
      - market-radar
      - growth-engine
      - social-autopilot
      - support
      - compliance
      - books
      - agentic-crm
      - lifecycle
      - agentic-privacy
      - growth-assistant
```

The per-module dashboards are still served by the module containers and proxied at
`/m/<name>` — unchanged. Only the control-plane image is re-based onto Context Runtime.

## The fleet registry

Modules are declared in [`../modules.yaml`](../modules.yaml) (name · repo · port ·
agents · approval_required). Each maps to a Context Runtime tenant via the
`integrations.modules` CATALOG (or a generic spec). `/status` is `deployed: true` for
every tenant the fleet has stood up (all of them, at startup).
