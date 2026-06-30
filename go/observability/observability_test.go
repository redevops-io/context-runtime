package observability

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/redevops-io/context-runtime/crtypes"
)

func TestJsonlExporterWritesExactlyOneLine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "traces.jsonl")
	exporter, err := NewJsonlExporter(path)
	if err != nil {
		t.Fatalf("new exporter: %v", err)
	}

	trace := NewTraceBuilder("plan_abc123", "why did deploy fail").Finalize()
	if err := exporter.Export(trace); err != nil {
		t.Fatalf("export: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read jsonl: %v", err)
	}
	lines := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected exactly one line, got %d: %q", len(lines), data)
	}
	if lines[0] == "" {
		t.Fatalf("expected a JSON record, got blank line")
	}
}

func TestMultiExporterToleratesPanickingSink(t *testing.T) {
	good := &recordingExporter{}
	multi := NewMultiExporter(panickingExporter{}, good)

	trace := crtypes.Trace{PlanID: "plan_abc123", GoalText: "goal", ID: "trace_xyz", SpecVersion: crtypes.SpecVersion}
	if err := multi.Export(trace); err != nil {
		t.Fatalf("multi export returned error: %v", err)
	}

	if got := len(good.traces); got != 1 {
		t.Fatalf("expected non-panicking sink to receive trace once, got %d", got)
	}
	if good.traces[0].ID != trace.ID {
		t.Fatalf("received wrong trace: %+v", good.traces[0])
	}
}

type recordingExporter struct {
	traces []crtypes.Trace
}

func (e *recordingExporter) Export(trace crtypes.Trace) error {
	e.traces = append(e.traces, trace)
	return nil
}

type panickingExporter struct{}

func (panickingExporter) Export(crtypes.Trace) error {
	panic("boom")
}
