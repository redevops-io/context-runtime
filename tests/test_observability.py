"""Trace exporters: JSONL works offline; runtime wires an exporter; lazy deps don't crash import."""
from __future__ import annotations

import json

from context_runtime import ContextRuntime
from context_runtime.observability.exporters import JsonlExporter, MultiExporter
from context_runtime.plugins import base


def test_jsonl_exporter_writes_a_line(tmp_path):
    exp = JsonlExporter(str(tmp_path / "traces.jsonl"))
    assert isinstance(exp, base.TraceExporter)
    rt = ContextRuntime.default([{"chunk_id": "a::0", "filename": "a.md",
                                  "text": "alpha beta deploy failed", "created_at": None}],
                                exporter=exp)
    rt.run("why did deploy fail")
    lines = (tmp_path / "traces.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["plan_id"] and rec["spans"]


def test_multi_exporter_tolerates_a_failing_sink(tmp_path):
    class Boom:
        def export(self, trace):
            raise RuntimeError("nope")
    good = JsonlExporter(str(tmp_path / "t.jsonl"))
    multi = MultiExporter(Boom(), good)
    rt = ContextRuntime.default([{"chunk_id": "a::0", "filename": "a.md", "text": "x y z", "created_at": None}],
                                exporter=multi)
    rt.run("x y z")   # must not raise despite Boom
    assert (tmp_path / "t.jsonl").exists()


def test_langfuse_and_otel_import_without_their_deps():
    # constructing the exporter must not require the optional dep (lazy import on export)
    from context_runtime.observability.exporters import LangfuseExporter, OpenLLMetryExporter
    LangfuseExporter(); OpenLLMetryExporter()


def test_jsonl_exporter_appends_across_runs(tmp_path):
    path = tmp_path / "t.jsonl"
    exp = JsonlExporter(str(path))
    rt = ContextRuntime.default([{"chunk_id": "a::0", "filename": "a.md",
                                  "text": "alpha beta deploy failed", "created_at": None}], exporter=exp)
    rt.run("why did deploy fail")
    rt.run("what is alpha")
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2                          # append mode accumulates (training log), not truncates
    for ln in lines:
        rec = json.loads(ln)
        assert rec["plan_id"] and rec["spans"]      # every accumulated line is a complete, valid trace
