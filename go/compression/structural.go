package compression

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/redevops-io/context-runtime/crtypes"
)

const charsPerToken = 4

// Compressed is the result of a structural compression pass.
type Compressed struct {
	Text         string   `json:"text"`
	Tokens       int      `json:"tokens"`
	DerivedFrom  []string `json:"derived_from"`
	Omitted      []string `json:"omitted"`
	RefreshAfter *string  `json:"refresh_after"`
}

// Clip preserves the head and tail of text around an elision marker when text is
// longer than maxChars. Character counts are based on UTF-8 runes.
func Clip(text string, maxChars int) string {
	if text == "" {
		return ""
	}

	runes := []rune(text)
	if maxChars < 0 {
		maxChars = 0
	}
	if len(runes) <= maxChars {
		return text
	}

	head := maxChars * 2 / 3
	tail := maxChars - head
	omitted := len(runes) - head - tail

	var tailText string
	if tail > 0 {
		tailText = string(runes[len(runes)-tail:])
	}

	return fmt.Sprintf("%s\n... [clipped %d chars] ...\n%s", string(runes[:head]), omitted, tailText)
}

// StructuralCompressor performs offline, provenance-preserving compression.
type StructuralCompressor struct{}

// Compress clips a single text blob to approximately targetTokens.
func (StructuralCompressor) Compress(text string, targetTokens int) Compressed {
	maxChars := maxInt(200, targetTokens*charsPerToken)
	clipped := Clip(text, maxChars)
	omitted := []string(nil)
	if utf8.RuneCountInString(clipped) < utf8.RuneCountInString(text) {
		omitted = []string{"clipped-middle"}
	}

	return Compressed{
		Text:    clipped,
		Tokens:  maxInt(1, utf8.RuneCountInString(clipped)/charsPerToken),
		Omitted: omitted,
	}
}

// Assemble packs ranked hits into a citation-numbered context within a token
// budget. Included hits are cited as [1], [2], ... using their original rank.
func (StructuralCompressor) Assemble(hits []crtypes.Hit, targetTokens int) Compressed {
	budgetChars := maxInt(400, targetTokens*charsPerToken)
	parts := make([]string, 0, len(hits))
	derived := make([]string, 0, len(hits))
	omitted := []string(nil)
	used := 0

	for i, h := range hits {
		block := fmt.Sprintf("[%d] %s: %s", i+1, h.Filename, h.Text)
		blockLen := utf8.RuneCountInString(block)
		if used+blockLen > budgetChars && len(parts) > 0 {
			omitted = append(omitted, h.ChunkID)
			continue
		}
		if used+blockLen > budgetChars {
			block = Clip(block, budgetChars-used)
			blockLen = utf8.RuneCountInString(block)
		}
		parts = append(parts, block)
		derived = append(derived, h.ChunkID)
		used += blockLen
	}

	text := strings.Join(parts, "\n\n")
	return Compressed{
		Text:        text,
		Tokens:      maxInt(1, utf8.RuneCountInString(text)/charsPerToken),
		DerivedFrom: derived,
		Omitted:     omitted,
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
