package constraints

import (
	"fmt"

	"github.com/redevops-io/context-runtime/crtypes"
)

// Feasible returns whether candidate satisfies all hard constraints and, if not,
// a human-readable reason for the first violated constraint.
func Feasible(candidate crtypes.Candidate, score crtypes.PlanScore, c crtypes.Constraints) (bool, string) {
	if c.MaxCostUSD != nil && score.CostUSD > *c.MaxCostUSD {
		return false, fmt.Sprintf("cost $%.2f > max $%.2f", score.CostUSD, *c.MaxCostUSD)
	}
	if c.MaxLatencySeconds != nil && score.LatencySeconds > *c.MaxLatencySeconds {
		return false, fmt.Sprintf("latency %.0fs > max %.0fs", score.LatencySeconds, *c.MaxLatencySeconds)
	}
	if c.RequireCitations && !hasVerifyStep(candidate) {
		return false, "require_citations but no verify step"
	}
	if c.RequireVerification && !hasVerifyStep(candidate) {
		return false, "require_verification but no verify step"
	}
	if c.Sensitivity == crtypes.SensitivityRestricted && candidate.ModelTier != "local" {
		return false, fmt.Sprintf("restricted data cannot use tier '%s'", candidate.ModelTier)
	}
	return true, ""
}

func hasVerifyStep(candidate crtypes.Candidate) bool {
	for _, step := range candidate.Steps {
		if step.Type == crtypes.StepTypeVerify {
			return true
		}
	}
	return false
}
