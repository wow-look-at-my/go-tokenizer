package tokenizer

import (
	"github.com/dlclark/regexp2"
)

// PreTokenizer splits text into chunks before BPE encoding
type PreTokenizer struct {
	pattern       *regexp2.Regexp
	specialTokens map[string]int
}

// NewPreTokenizer creates a pre-tokenizer with the given regex pattern
func NewPreTokenizer(pattern string, specialTokens map[string]int) (*PreTokenizer, error) {
	re, err := regexp2.Compile(pattern, regexp2.None)
	if err != nil {
		return nil, err
	}

	return &PreTokenizer{
		pattern:       re,
		specialTokens: specialTokens,
	}, nil
}

// Tokenize splits text into pre-tokens
// Returns list of (text, isSpecial) pairs
func (p *PreTokenizer) Tokenize(text string) ([]PreToken, error) {
	if len(p.specialTokens) > 0 {
		return p.tokenizeWithSpecial(text)
	}
	return p.tokenizeRegex(text)
}

// PreToken represents a pre-tokenized chunk
type PreToken struct {
	Text      string
	IsSpecial bool
}

// tokenizeWithSpecial handles special tokens before regex splitting
func (p *PreTokenizer) tokenizeWithSpecial(text string) ([]PreToken, error) {
	var result []PreToken
	remaining := text

	for len(remaining) > 0 {
		// Find earliest special token
		earliestPos := -1
		var earliestToken string

		for token := range p.specialTokens {
			pos := indexOf(remaining, token)
			if pos != -1 && (earliestPos == -1 || pos < earliestPos) {
				earliestPos = pos
				earliestToken = token
			}
		}

		if earliestPos == -1 {
			// No more special tokens, process rest with regex
			tokens, err := p.tokenizeRegex(remaining)
			if err != nil {
				return nil, err
			}
			result = append(result, tokens...)
			break
		}

		// Process text before special token
		if earliestPos > 0 {
			tokens, err := p.tokenizeRegex(remaining[:earliestPos])
			if err != nil {
				return nil, err
			}
			result = append(result, tokens...)
		}

		// Add special token
		result = append(result, PreToken{Text: earliestToken, IsSpecial: true})
		remaining = remaining[earliestPos+len(earliestToken):]
	}

	return result, nil
}

// tokenizeRegex splits text using the regex pattern
func (p *PreTokenizer) tokenizeRegex(text string) ([]PreToken, error) {
	var result []PreToken

	match, err := p.pattern.FindStringMatch(text)
	if err != nil {
		return nil, err
	}

	for match != nil {
		result = append(result, PreToken{Text: match.String(), IsSpecial: false})
		match, err = p.pattern.FindNextMatch(match)
		if err != nil {
			return nil, err
		}
	}

	return result, nil
}

// indexOf returns the index of substr in s, or -1 if not found
func indexOf(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}
