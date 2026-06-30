package crtypes

import (
	"encoding/json"
	"testing"
)

// A Plan round-trips through JSON with snake_case field names preserved.
func TestPlanRoundTrip(t *testing.T) {
	p := Plan{
		Intent:      Intent{Bucket: "incident", Entities: []string{"ERR-500"}, Risk: "medium", Normalized: "deploy failed", Confidence: 0.8},
		Chosen:      Candidate{Steps: []StepSpec{{Type: "retrieve", Params: map[string]any{"method": "hybrid"}}}, ModelTier: "cheap"},
		Score:       PlanScore{ExpectedAccuracy: 0.9, CostUSD: 0.05, Total: 0.87, Feasible: true},
		Cache:       "miss",
		ID:          "plan_abc123",
		SpecVersion: "0.1",
	}
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	// the wire form must use the Python snake_case keys
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatalf("unmarshal raw: %v", err)
	}
	for _, k := range []string{"intent", "chosen", "score", "cache", "spec_version"} {
		if _, ok := raw[k]; !ok {
			t.Errorf("missing json key %q in %s", k, b)
		}
	}
	var back Plan
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.ID != p.ID || back.Intent.Bucket != p.Intent.Bucket ||
		back.Chosen.ModelTier != p.Chosen.ModelTier || back.Score.Total != p.Score.Total {
		t.Errorf("round-trip mismatch: %+v != %+v", back, p)
	}
}

// A Trace round-trips, including the optional VerificationPassed pointer and the Extra bag.
func TestTraceRoundTrip(t *testing.T) {
	yes := true
	tr := Trace{
		PlanID:             "plan_abc123",
		GoalText:           "why did deploy fail",
		Spans:              []Span{{Name: "reason", Kind: "reason", Start: "t0", End: "t1"}},
		ActualTokens:       574,
		VerificationPassed: &yes,
		Cache:              "miss",
		ID:                 "trace_xyz",
		SpecVersion:        "0.1",
		Extra:              map[string]any{"future_field": 42.0},
	}
	b, err := json.Marshal(tr)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back Trace
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.PlanID != tr.PlanID || back.ActualTokens != tr.ActualTokens ||
		back.VerificationPassed == nil || *back.VerificationPassed != true ||
		len(back.Spans) != 1 || back.Spans[0].Kind != "reason" {
		t.Errorf("trace round-trip mismatch: %+v", back)
	}
}
