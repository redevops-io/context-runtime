"""Trace exporters — get a finalized Trace out of the process (SPEC §5.12).

Context Runtime's v0.1 trace was in-process only. These exporters ship it to where it can be
inspected, replayed, and turned into training data: a local JSONL file, **Langfuse**
(self-hostable trace/cost/eval UI), or **OpenTelemetry** via OpenLLMetry semantic
conventions. Langfuse/OTel deps are lazy-imported so the core stays dependency-free.

    rt = ContextRuntime(..., exporter=LangfuseExporter())
"""
from __future__ import annotations

import logging
from pathlib import Path

from .. import jsonio
from ..platform_services import LockTimeout, locked_append
from ..types import Trace

_log = logging.getLogger("context_runtime.observability")

#: How long an export may wait for the append lock. Traces are diagnostic and the exporter runs on the request
#: path, so contention must not stall the application for long.
EXPORT_LOCK_TIMEOUT = 1.0


class JsonlExporter:
    """Append each trace as one JSON line — zero deps, always available. Safe for concurrent threads and
    processes: each line is appended under an inter-process lock (a bare append can lose lines on Windows)."""

    def __init__(self, path: str = ".context_runtime/traces.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._warned = False

    def export(self, trace: Trace) -> None:
        """Append one line. If the lock is not available within ``EXPORT_LOCK_TIMEOUT`` (another process holds it,
        or this thread already does), the trace is dropped and ONE warning is logged for this exporter; any other
        I/O error is logged once and raised, as a failed write always was."""
        line = jsonio.dumps(trace) + "\n"        # explicit newline; written in one call, not fsync-ed (diagnostic)
        try:
            locked_append(self.path, line, durable=False, timeout=EXPORT_LOCK_TIMEOUT)
        except LockTimeout:
            self._warn_once(f"trace dropped: could not take the append lock for {self.path} within "
                            f"{EXPORT_LOCK_TIMEOUT:g}s (further drops are not logged)")
        except OSError as e:
            self._warn_once(f"trace export to {self.path} failed ({type(e).__name__}: {e}); further failures are "
                            "not logged")
            raise

    def _warn_once(self, message: str) -> None:
        if not self._warned:
            self._warned = True
            _log.warning(message)


class MultiExporter:
    """Fan a trace out to several exporters; one failing never blocks the others."""

    def __init__(self, *exporters):
        self.exporters = [e for e in exporters if e is not None]

    def export(self, trace: Trace) -> None:
        for e in self.exporters:
            try:
                e.export(trace)
            except Exception:
                pass


class LangfuseExporter:
    """Ship traces to Langfuse (self-hostable). Needs the ``[langfuse]`` extra.

    Maps the Context Runtime Trace → one Langfuse trace with a nested span per Context Runtime
    span, carrying tokens/cost/model attrs so Langfuse's cost + replay views light up.
    """

    def __init__(self, public_key: str | None = None, secret_key: str | None = None,
                 host: str | None = None):
        self._kw = {k: v for k, v in
                    {"public_key": public_key, "secret_key": secret_key, "host": host}.items() if v}
        self._client = None

    def _get(self):
        if self._client is None:
            try:
                from langfuse import Langfuse  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("LangfuseExporter needs: pip install langfuse") from e
            self._client = Langfuse(**self._kw)
        return self._client

    def export(self, trace: Trace) -> None:
        lf = self._get()
        t = lf.trace(id=trace.id, name="context_runtime.run", input=trace.goal_text,
                     metadata={"plan_id": trace.plan_id, "cache": trace.cache,
                               "cost_usd": trace.actual_cost_usd, "tokens": trace.actual_tokens})
        for s in trace.spans:
            t.span(name=s.name, start_time=s.start, end_time=s.end,
                   metadata={"kind": s.kind, **s.attrs})
        if hasattr(lf, "flush"):
            lf.flush()


class OpenLLMetryExporter:
    """Emit OTel spans via OpenLLMetry semantic conventions. Needs the ``[otel]`` extra."""

    def __init__(self, service_name: str = "context_runtime"):
        self.service_name = service_name
        self._tracer = None

    def _get(self):
        if self._tracer is None:
            try:
                from opentelemetry import trace as ot  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("OpenLLMetryExporter needs: pip install opentelemetry-sdk") from e
            self._tracer = ot.get_tracer(self.service_name)
        return self._tracer

    def export(self, trace: Trace) -> None:
        tracer = self._get()
        with tracer.start_as_current_span("context_runtime.run") as root:
            root.set_attribute("context_runtime.plan_id", trace.plan_id)
            root.set_attribute("gen_ai.usage.total_tokens", trace.actual_tokens)
            root.set_attribute("context_runtime.cost_usd", trace.actual_cost_usd)
            for s in trace.spans:
                with tracer.start_as_current_span(s.name) as span:
                    span.set_attribute("context_runtime.kind", s.kind)
                    for k, v in s.attrs.items():
                        if isinstance(v, (str, int, float, bool)):
                            span.set_attribute(f"context_runtime.{k}", v)
