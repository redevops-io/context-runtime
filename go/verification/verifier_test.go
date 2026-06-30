package verification

import (
	"reflect"
	"testing"

	"github.com/redevops-io/context-runtime/crtypes"
)

func TestCitationVerifierPassesWhenAllCitationsAreValid(t *testing.T) {
	verifier := CitationVerifier{}
	ctx := contextWithHits(2)

	verdict := verifier.Verify(
		crtypes.ModelResult{Text: "The answer uses the first block [1] and the second block [2]."},
		crtypes.Plan{},
		ctx,
	)

	if !verdict.Passed {
		t.Fatalf("expected valid citations to pass, got %+v", verdict)
	}
	if verdict.Confidence != 0.9 {
		t.Fatalf("expected confidence 0.9, got %v", verdict.Confidence)
	}
	if len(verdict.Findings) != 0 {
		t.Fatalf("expected no findings, got %v", verdict.Findings)
	}
}

func TestCitationVerifierFailsOnDanglingCitation(t *testing.T) {
	verifier := CitationVerifier{}
	ctx := contextWithHits(1)

	verdict := verifier.Verify(
		crtypes.ModelResult{Text: "This cites an existing block [1] and a missing block [2]."},
		crtypes.Plan{},
		ctx,
	)

	if verdict.Passed {
		t.Fatalf("expected dangling citation to fail, got %+v", verdict)
	}
	if verdict.Confidence != 0.3 {
		t.Fatalf("expected confidence 0.3, got %v", verdict.Confidence)
	}
	expectedFindings := []string{"citations refer to non-existent blocks: [2]"}
	if !reflect.DeepEqual(verdict.Findings, expectedFindings) {
		t.Fatalf("unexpected findings: got %v want %v", verdict.Findings, expectedFindings)
	}
}

func TestCitationVerifierFailsWhenThereAreNoCitations(t *testing.T) {
	verifier := CitationVerifier{}
	ctx := contextWithHits(2)

	verdict := verifier.Verify(
		crtypes.ModelResult{Text: "This answer does not cite any source."},
		crtypes.Plan{},
		ctx,
	)

	if verdict.Passed {
		t.Fatalf("expected uncited answer to fail, got %+v", verdict)
	}
	if verdict.Confidence != 0.2 {
		t.Fatalf("expected confidence 0.2, got %v", verdict.Confidence)
	}
	expectedFindings := []string{"answer cites no sources"}
	if !reflect.DeepEqual(verdict.Findings, expectedFindings) {
		t.Fatalf("unexpected findings: got %v want %v", verdict.Findings, expectedFindings)
	}
}

func contextWithHits(n int) BuiltContext {
	hits := make([]crtypes.Hit, n)
	for i := range hits {
		hits[i] = crtypes.Hit{ChunkID: "chunk"}
	}
	return BuiltContext{Hits: hits}
}
