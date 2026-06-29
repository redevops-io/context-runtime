"""Trace exporters — get a finalized Trace out of the process (SPEC §5.12).

ContextOS's v0.1 trace was in-process only. These exporters ship it to where it can be
inspected, replayed, and turned into training data: a local JSONL file, **Langfuse**
(self-hostable trace/cost/eval UI), or **OpenTelemetry** via OpenLLMetry semantic
conventions. Langfuse/OTel deps are lazy-imported so the core stays dependency-free.

    rt = ContextRuntime(..., exporter=LangfuseExporter())
"""
from __future__ import annotations

from pathlib import Path

from .. import jsonio
from ..types import Trace


class JsonlExporter:
    """Append each trace as one JSON line — zero deps, always available."""

    def __init__(self, path: str = ".contextos/traces.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, trace: Trace) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(jsonio.dumps(trace) + "\n")


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

    Maps the ContextOS Trace → one Langfuse trace with a nested span per ContextOS
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
        t = lf.trace(id=trace.id, name="contextos.run", input=trace.goal_text,
                     metadata={"plan_id": trace.plan_id, "cache": trace.cache,
                               "cost_usd": trace.actual_cost_usd, "tokens": trace.actual_tokens})
        for s in trace.spans:
            t.span(name=s.name, start_time=s.start, end_time=s.end,
                   metadata={"kind": s.kind, **s.attrs})
        if hasattr(lf, "flush"):
            lf.flush()


class OpenLLMetryExporter:
    """Emit OTel spans via OpenLLMetry semantic conventions. Needs the ``[otel]`` extra."""

    def __init__(self, service_name: str = "contextos"):
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
        with tracer.start_as_current_span("contextos.run") as root:
            root.set_attribute("contextos.plan_id", trace.plan_id)
            root.set_attribute("gen_ai.usage.total_tokens", trace.actual_tokens)
            root.set_attribute("contextos.cost_usd", trace.actual_cost_usd)
            for s in trace.spans:
                with tracer.start_as_current_span(s.name) as span:
                    span.set_attribute("contextos.kind", s.kind)
                    for k, v in s.attrs.items():
                        if isinstance(v, (str, int, float, bool)):
                            span.set_attribute(f"contextos.{k}", v)
