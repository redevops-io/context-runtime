package observability

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/redevops-io/context-runtime/crtypes"
)

// Now returns the current UTC time in RFC3339 form for trace span timestamps.
func Now() string {
	return time.Now().UTC().Format(time.RFC3339Nano)
}

// SpanEnd is returned by TraceBuilder.StartSpan. It can be called directly or via End.
type SpanEnd func()

// End closes the span. It is safe to call more than once.
func (e SpanEnd) End() {
	if e != nil {
		e()
	}
}

// TraceBuilder accumulates span observations and actual costs for a single run.
type TraceBuilder struct {
	mu        sync.Mutex
	planID    string
	goalText  string
	traceID   string
	startedAt time.Time

	spans     []crtypes.Span
	costUSD   float64
	tokens    int
	citations []string
	verified  *bool
	cache     crtypes.Cache
}

// NewTraceBuilder creates a builder for a trace associated with planID and goalText.
func NewTraceBuilder(planID, goalText string) *TraceBuilder {
	return &TraceBuilder{
		planID:    planID,
		goalText:  goalText,
		traceID:   newID("trace"),
		startedAt: time.Now().UTC(),
		spans:     []crtypes.Span{},
		citations: []string{},
		cache:     crtypes.CacheMiss,
	}
}

// StartSpan starts a span at the current time and returns an idempotent end function.
// The kind argument accepts crtypes.SpanKind or a string for convenient callers.
func (b *TraceBuilder) StartSpan(name string, kind any, attrs map[string]any) SpanEnd {
	if b == nil {
		return SpanEnd(func() {})
	}

	span := crtypes.Span{
		Name:  name,
		Kind:  asSpanKind(kind),
		Start: Now(),
		End:   "",
		Attrs: copyAttrs(attrs),
		ID:    newID("span"),
	}

	b.mu.Lock()
	idx := len(b.spans)
	b.spans = append(b.spans, span)
	b.mu.Unlock()

	var once sync.Once
	return SpanEnd(func() {
		once.Do(func() {
			b.mu.Lock()
			defer b.mu.Unlock()
			if idx >= 0 && idx < len(b.spans) && b.spans[idx].End == "" {
				b.spans[idx].End = Now()
			}
		})
	})
}

// Span appends a fully timed span. It mirrors the Python TraceBuilder.span helper.
func (b *TraceBuilder) Span(name string, kind any, attrs map[string]any, start, end string) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.spans = append(b.spans, crtypes.Span{
		Name:  name,
		Kind:  asSpanKind(kind),
		Start: start,
		End:   end,
		Attrs: copyAttrs(attrs),
		ID:    newID("span"),
	})
}

// AddCost rolls token and dollar usage into the trace actuals.
func (b *TraceBuilder) AddCost(usd float64, tokens int) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.costUSD += usd
	b.tokens += tokens
}

// SetCitations records the citations emitted by the run.
func (b *TraceBuilder) SetCitations(citations []string) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.citations = append([]string(nil), citations...)
}

// SetVerified records verification status. Pass bool, *bool, or nil to clear it.
func (b *TraceBuilder) SetVerified(ok any) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	switch v := ok.(type) {
	case nil:
		b.verified = nil
	case bool:
		vv := v
		b.verified = &vv
	case *bool:
		if v == nil {
			b.verified = nil
		} else {
			vv := *v
			b.verified = &vv
		}
	}
}

// SetCache records whether the run used the plan cache.
func (b *TraceBuilder) SetCache(cache crtypes.Cache) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.cache = cache
}

// Finalize returns an immutable snapshot of the accumulated trace. Open spans are
// closed at finalize time so callers that only start a span still get valid JSON.
func (b *TraceBuilder) Finalize() crtypes.Trace {
	if b == nil {
		return crtypes.Trace{SpecVersion: crtypes.SpecVersion, Cache: crtypes.CacheMiss, Spans: []crtypes.Span{}, Citations: []string{}, Extra: map[string]any{}}
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	now := time.Now().UTC()
	nowText := now.Format(time.RFC3339Nano)
	spans := make([]crtypes.Span, len(b.spans))
	for i, span := range b.spans {
		if span.End == "" {
			span.End = nowText
		}
		span.Attrs = copyAttrs(span.Attrs)
		spans[i] = span
	}

	var verified *bool
	if b.verified != nil {
		v := *b.verified
		verified = &v
	}

	return crtypes.Trace{
		PlanID:               b.planID,
		GoalText:             b.goalText,
		Spans:                spans,
		ActualCostUSD:        round(b.costUSD, 6),
		ActualLatencySeconds: round(now.Sub(b.startedAt).Seconds(), 4),
		ActualTokens:         b.tokens,
		Citations:            append([]string(nil), b.citations...),
		VerificationPassed:   verified,
		Cache:                b.cache,
		ID:                   b.traceID,
		SpecVersion:          crtypes.SpecVersion,
		Extra:                map[string]any{},
	}
}

// SaveTrace writes a finalized trace as pretty JSON into dirPath/<trace.id>.json.
func SaveTrace(trace crtypes.Trace, dirPath string) (string, error) {
	if dirPath == "" {
		dirPath = "."
	}
	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		return "", err
	}
	id := trace.ID
	if id == "" {
		id = newID("trace")
		trace.ID = id
	}
	path := filepath.Join(dirPath, id+".json")
	data, err := json.MarshalIndent(trace, "", "  ")
	if err != nil {
		return "", err
	}
	data = append(data, '\n')
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return "", err
	}
	return path, nil
}

func asSpanKind(kind any) crtypes.SpanKind {
	switch v := kind.(type) {
	case crtypes.SpanKind:
		return v
	case string:
		return crtypes.SpanKind(v)
	case fmt.Stringer:
		return crtypes.SpanKind(v.String())
	default:
		return crtypes.SpanKind(fmt.Sprint(v))
	}
}

func copyAttrs(attrs map[string]any) map[string]any {
	if attrs == nil {
		return map[string]any{}
	}
	out := make(map[string]any, len(attrs))
	for k, v := range attrs {
		out[k] = v
	}
	return out
}

func round(v float64, places int) float64 {
	factor := math.Pow10(places)
	return math.Round(v*factor) / factor
}

func newID(kind string) string {
	var b [6]byte
	if _, err := rand.Read(b[:]); err == nil {
		return kind + "_" + hex.EncodeToString(b[:])
	}
	return fmt.Sprintf("%s_%012x", kind, time.Now().UnixNano()&0xffffffffffff)
}
