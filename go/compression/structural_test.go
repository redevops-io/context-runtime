package compression

import (
	"strings"
	"testing"

	"github.com/redevops-io/context-runtime/crtypes"
)

func TestClipPreservesHeadTailAroundElisionMarker(t *testing.T) {
	input := "abcdefghijklmnopqrstuvwxyz"
	got := Clip(input, 10)

	if got == input {
		t.Fatalf("Clip returned the unmodified over-length input")
	}
	if !strings.Contains(got, "... [clipped 16 chars] ...") {
		t.Fatalf("Clip output missing elision marker: %q", got)
	}
	if !strings.HasPrefix(got, "abcdef\n") {
		t.Fatalf("Clip did not preserve expected head: %q", got)
	}
	if !strings.HasSuffix(got, "\nwxyz") {
		t.Fatalf("Clip did not preserve expected tail: %q", got)
	}
}

func TestStructuralCompressorAssembleNumbersHits(t *testing.T) {
	hits := []crtypes.Hit{
		{ChunkID: "chunk-1", Filename: "one.md", Text: "alpha", Score: 1.0},
		{ChunkID: "chunk-2", Filename: "two.md", Text: "bravo", Score: 0.9},
	}

	got := (StructuralCompressor{}).Assemble(hits, 200)

	first := "[1] one.md: alpha"
	second := "[2] two.md: bravo"
	if !strings.Contains(got.Text, first) {
		t.Fatalf("assembled context missing first numbered hit %q in %q", first, got.Text)
	}
	if !strings.Contains(got.Text, second) {
		t.Fatalf("assembled context missing second numbered hit %q in %q", second, got.Text)
	}
	if strings.Index(got.Text, first) > strings.Index(got.Text, second) {
		t.Fatalf("assembled context has hits out of order: %q", got.Text)
	}
}
