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
ENV PYTHONPATH=/app

EXPOSE 8080
# CONTEXT_RUNTIME_API_KEY gates POST routes (falls back to AGENTIC_OS_API_KEY);
# CONTEXT_RUNTIME_HOME (falls back to AGENTIC_OS_HOME) holds the approvals log.
CMD ["uvicorn", "context_runtime.control_plane.app:app", "--host", "0.0.0.0", "--port", "8080"]
