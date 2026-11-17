# Context Runtime control plane — the rebuildable replacement for the retired
# agentic-os control-plane image. Drop-in for agentic-os-stack/integrated.compose.yml:
# serves the SAME API (/, /health, /status, /modules, /m/<name>, /dispatch, /approvals)
# on :8080, but the brain is Context Runtime's ModuleTenant fleet.
FROM python:3.12-slim
WORKDIR /app

# Run from source (PYTHONPATH) so context_runtime/modules.yaml is always present —
# no packaging/data-file surprises for a long-running service.
COPY pyproject.toml README.md ./
COPY context_runtime ./context_runtime
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29" "httpx>=0.27" "pyyaml>=6.0"
# agent-harness is a base dependency of context_runtime (tools/base.py re-exports its tool contract), so the
# application does not import without it. Pinned to the same commit as pyproject.toml, from the commit's
# source archive (no git in this image) and verified against the archive's sha256.
# tests/test_container.py checks that this commit equals the one in pyproject.toml.
ARG AGENT_HARNESS_COMMIT=6b8ba3a596a5e2caf0fdf009f2af013517591662
ARG AGENT_HARNESS_SHA256=d67eadb33c161df00b70bad4f20c80f43d2f477031dce3259ee5d465c8844a47
RUN pip install --no-cache-dir "agent-harness @ https://github.com/redevops-io/agent-harness/archive/${AGENT_HARNESS_COMMIT}.tar.gz#sha256=${AGENT_HARNESS_SHA256}"
ENV PYTHONPATH=/app

EXPOSE 8080
# Authentication fails closed. CONTEXT_RUNTIME_API_KEY (falls back to AGENTIC_OS_API_KEY) is the
# control-plane key: with none set, every route except GET /health answers 503. CR_TENANT_KEYS binds
# tenant keys for the /librechat retrieval and /v1 routes. CONTEXT_RUNTIME_CORPUS_ROOT names the only
# directory tree /librechat/ingest may index (unset: ingestion is disabled).
# CONTEXT_RUNTIME_HOME (falls back to AGENTIC_OS_HOME) holds the approvals log.
CMD ["uvicorn", "context_runtime.control_plane.app:app", "--host", "0.0.0.0", "--port", "8080"]
