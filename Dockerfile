# Context Runtime control plane — the rebuildable replacement for the retired
# agentic-os control-plane image. Drop-in for agentic-os-stack/integrated.compose.yml:
# serves the SAME API (/, /health, /status, /modules, /m/<name>, /dispatch, /approvals)
# on :8080, but the brain is Context Runtime's ModuleTenant fleet.
FROM python:3.12-slim
WORKDIR /app

COPY pyproject.toml README.md ./
COPY context_runtime ./context_runtime
RUN pip install --no-cache-dir ".[control-plane]"

EXPOSE 8080
# CONTEXT_RUNTIME_API_KEY gates POST routes; CONTEXT_RUNTIME_HOME holds the approvals log.
CMD ["uvicorn", "context_runtime.control_plane.app:app", "--host", "0.0.0.0", "--port", "8080"]
