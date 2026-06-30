// Package verification contains deterministic result verification helpers.
package verification

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/redevops-io/context-runtime/crtypes"
)

var citationPattern = regexp.MustCompile(`\[(\d+)\]`)

// BuiltContext is the assembled context that a model result was generated from.
//
// It mirrors the Python BuiltContext shape while reusing the Go crtypes boundary
// types for plans, execution graphs, and retrieval hits.
type BuiltContext struct {
	Plan          crtypes.Plan           `json:"plan"`
	Graph         crtypes.ExecutionGraph `json:"graph"`
	Hits          []crtypes.Hit          `json:"hits"`
	AssembledText string                 `json:"assembled_text"`
	TokenBudget   map[string]int         `json:"token_budget"`
}

// Verdict is the outcome of a verifier plugin.
type Verdict struct {
	Passed     bool     `json:"passed"`
	Confidence float64  `json:"confidence"`
	Findings   []string `json:"findings"`
}

// CitationVerifier checks that every [n] citation in an answer refers to an
// assembled context block.
type CitationVerifier struct{}

// Verify returns a deterministic citation/grounding verdict for result.
//
// The plan parameter is part of the verifier contract; this v0.1 verifier only
// needs the number of hits in ctx.
func (CitationVerifier) Verify(result crtypes.ModelResult, _ crtypes.Plan, ctx BuiltContext) Verdict {
	cited := make(map[int]struct{})
	for _, match := range citationPattern.FindAllStringSubmatch(result.Text, -1) {
		if len(match) < 2 {
			continue
		}
		citation, err := strconv.Atoi(match[1])
		if err != nil {
			// The regular expression only admits digits. If the number cannot fit in
			// an int, it cannot refer to any real in-memory context block.
			citation = len(ctx.Hits) + 1
		}
		cited[citation] = struct{}{}
	}

	if len(cited) == 0 {
		return Verdict{
			Passed:     false,
			Confidence: 0.2,
			Findings:   []string{"answer cites no sources"},
		}
	}

	dangling := make([]int, 0)
	for citation := range cited {
		if citation < 1 || citation > len(ctx.Hits) {
			dangling = append(dangling, citation)
		}
	}
	sort.Ints(dangling)

	if len(dangling) > 0 {
		return Verdict{
			Passed:     false,
			Confidence: 0.3,
			Findings: []string{
				fmt.Sprintf("citations refer to non-existent blocks: %s", formatIntList(dangling)),
			},
		}
	}

	return Verdict{Passed: true, Confidence: 0.9}
}

func formatIntList(values []int) string {
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = strconv.Itoa(value)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}
