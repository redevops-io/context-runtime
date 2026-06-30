package constraints

import (
	"testing"

	"github.com/redevops-io/context-runtime/crtypes"
)

func TestFeasibleRestrictedRequiresLocalTier(t *testing.T) {
	candidate := crtypes.Candidate{ModelTier: "cheap"}
	score := crtypes.PlanScore{CostUSD: 0.01}
	constraints := crtypes.Constraints{Sensitivity: crtypes.SensitivityRestricted}

	ok, reason := Feasible(candidate, score, constraints)
	if ok {
		t.Fatalf("Feasible() = true, want false")
	}
	want := "restricted data cannot use tier 'cheap'"
	if reason != want {
		t.Fatalf("reason = %q, want %q", reason, want)
	}
}

func TestFeasibleCostOverMax(t *testing.T) {
	maxCost := 0.05
	candidate := crtypes.Candidate{ModelTier: "local"}
	score := crtypes.PlanScore{CostUSD: 0.06}
	constraints := crtypes.Constraints{MaxCostUSD: &maxCost}

	ok, reason := Feasible(candidate, score, constraints)
	if ok {
		t.Fatalf("Feasible() = true, want false")
	}
	want := "cost $0.06 > max $0.05"
	if reason != want {
		t.Fatalf("reason = %q, want %q", reason, want)
	}
}

func TestFeasibleHappyPath(t *testing.T) {
	maxCost := 0.10
	maxLatency := 5.0
	candidate := crtypes.Candidate{
		ModelTier: "local",
		Steps: []crtypes.StepSpec{
			{Type: crtypes.StepTypeRetrieve},
			{Type: crtypes.StepTypeVerify},
		},
	}
	score := crtypes.PlanScore{CostUSD: 0.05, LatencySeconds: 2.0}
	constraints := crtypes.Constraints{
		MaxCostUSD:          &maxCost,
		MaxLatencySeconds:   &maxLatency,
		RequireCitations:    true,
		RequireVerification: true,
		Sensitivity:         crtypes.SensitivityRestricted,
	}

	ok, reason := Feasible(candidate, score, constraints)
	if !ok {
		t.Fatalf("Feasible() = false, want true; reason: %q", reason)
	}
	if reason != "" {
		t.Fatalf("reason = %q, want empty", reason)
	}
}
