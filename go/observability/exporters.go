package observability

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"

	"github.com/redevops-io/context-runtime/crtypes"
)

// Exporter receives a finalized trace.
type Exporter interface {
	Export(trace crtypes.Trace) error
}

// TraceExporter is a no-op exporter stub for future OTel/Langfuse integrations.
type TraceExporter struct{}

// Export intentionally drops the trace and succeeds.
func (TraceExporter) Export(crtypes.Trace) error { return nil }

// JsonlExporter appends each trace as one JSON object per line.
type JsonlExporter struct {
	Path string
	mu   sync.Mutex
}

// NewJsonlExporter creates a JSONL exporter, making the parent directory if needed.
func NewJsonlExporter(path string) (*JsonlExporter, error) {
	if path == "" {
		path = ".context_runtime/traces.jsonl"
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}
	return &JsonlExporter{Path: path}, nil
}

// Export appends trace as a single newline-terminated JSON record.
func (e *JsonlExporter) Export(trace crtypes.Trace) error {
	if e == nil {
		return nil
	}
	path := e.Path
	if path == "" {
		path = ".context_runtime/traces.jsonl"
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}

	data, err := json.Marshal(trace)
	if err != nil {
		return err
	}
	data = append(data, '\n')

	e.mu.Lock()
	defer e.mu.Unlock()
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(data)
	return err
}

// MultiExporter fans a trace out to multiple sinks. A sink returning an error or
// panicking is isolated so later sinks still receive the trace.
type MultiExporter struct {
	Exporters []Exporter
}

// NewMultiExporter creates a fan-out exporter, ignoring nil sinks.
func NewMultiExporter(exporters ...Exporter) *MultiExporter {
	m := &MultiExporter{}
	for _, exporter := range exporters {
		if exporter != nil {
			m.Exporters = append(m.Exporters, exporter)
		}
	}
	return m
}

// Export sends trace to every sink and returns the first non-panic error, if any.
func (m *MultiExporter) Export(trace crtypes.Trace) error {
	if m == nil {
		return nil
	}
	var firstErr error
	for _, exporter := range m.Exporters {
		if exporter == nil {
			continue
		}
		if err := safeExport(exporter, trace); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

func safeExport(exporter Exporter, trace crtypes.Trace) (err error) {
	defer func() {
		if recover() != nil {
			err = nil
		}
	}()
	return exporter.Export(trace)
}
