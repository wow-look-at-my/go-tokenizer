package tokenizer

import (
	"unicode"
	"unicode/utf8"

	"github.com/dlclark/regexp2"
)

// PreTokenizer splits text into chunks before BPE encoding
type PreTokenizer struct {
	pattern       *regexp2.Regexp
	specialTokens map[string]int

	// scan splits text without the regex engine when the pattern is one this
	// package recognizes. regexp2 backtracks, and pre-tokenization was measured
	// as the majority of the time spent counting; a direct scanner removes it.
	// TestFastScanMatchesRegexp holds the two to identical output.
	scan func(text string, at int) int
}

// NewPreTokenizer creates a pre-tokenizer with the given regex pattern
func NewPreTokenizer(pattern string, specialTokens map[string]int) (*PreTokenizer, error) {
	re, err := regexp2.Compile(pattern, regexp2.None)
	if err != nil {
		return nil, err
	}

	p := &PreTokenizer{
		pattern:       re,
		specialTokens: specialTokens,
	}
	if pattern == cl100kPattern {
		p.scan = scanCl100k
	}
	return p, nil
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
	if p.scan != nil {
		return p.tokenizeScan(text), nil
	}

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

// tokenizeScan splits text with the direct scanner.
func (p *PreTokenizer) tokenizeScan(text string) []PreToken {
	// Most pre-tokens are a short word, so the count lands near a quarter of
	// the byte length; sizing for that avoids most of the regrowth.
	result := make([]PreToken, 0, len(text)/4+1)
	for at := 0; at < len(text); {
		end := p.scan(text, at)
		if end <= at {
			// A rune the pattern cannot start a match on would otherwise spin
			// here. Emitting it alone keeps the split lossless.
			_, size := utf8.DecodeRuneInString(text[at:])
			end = at + size
		}
		result = append(result, PreToken{Text: text[at:end]})
		at = end
	}
	return result
}

// scanCl100k returns the end offset of the pre-token starting at at, following
// the alternatives of cl100kPattern in the order the regex tries them.
func scanCl100k(text string, at int) int {
	r, size := utf8.DecodeRuneInString(text[at:])

	// (?i:'s|'t|'re|'ve|'m|'ll|'d)
	if r == '\'' {
		if end := scanContraction(text, at); end > at {
			return end
		}
	}

	// [^\r\n\p{L}\p{N}]?\p{L}+ -- a run of letters, which may take one leading
	// character that is neither a line break nor alphanumeric.
	if unicode.IsLetter(r) {
		return scanRun(text, at, unicode.IsLetter)
	}
	if r != '\r' && r != '\n' && !unicode.IsNumber(r) {
		if next, _ := utf8.DecodeRuneInString(text[at+size:]); unicode.IsLetter(next) {
			return scanRun(text, at+size, unicode.IsLetter)
		}
	}

	// \p{N}{1,3}
	if unicode.IsNumber(r) {
		end, taken := at, 0
		for end < len(text) && taken < 3 {
			d, dSize := utf8.DecodeRuneInString(text[end:])
			if !unicode.IsNumber(d) {
				break
			}
			end, taken = end+dSize, taken+1
		}
		return end
	}

	// ' ?[^\s\p{L}\p{N}]+[\r\n]*'
	symbolStart := at
	if r == ' ' {
		symbolStart = at + size
	}
	if end := scanRun(text, symbolStart, isSymbol); end > symbolStart {
		for end < len(text) && (text[end] == '\r' || text[end] == '\n') {
			end++
		}
		return end
	}

	if !unicode.IsSpace(r) {
		return at
	}

	// The remaining alternatives all consume whitespace. '\s*[\r\n]+' wins when
	// the run holds a line break, and ends after the last one.
	runEnd, lastBreakEnd := at, -1
	for runEnd < len(text) {
		w, wSize := utf8.DecodeRuneInString(text[runEnd:])
		if !unicode.IsSpace(w) {
			break
		}
		runEnd += wSize
		if w == '\r' || w == '\n' {
			lastBreakEnd = runEnd
		}
	}
	if lastBreakEnd > at {
		return lastBreakEnd
	}

	// '\s+(?!\S)' keeps the run only while nothing but whitespace follows, so a
	// run that runs into a word gives its last character to that word. A single
	// character cannot be given away, and '\s+' takes it instead.
	if runEnd == len(text) {
		return runEnd
	}
	if _, lastSize := utf8.DecodeLastRuneInString(text[at:runEnd]); runEnd-lastSize > at {
		return runEnd - lastSize
	}
	return runEnd
}

// scanContraction matches the apostrophe forms the pattern lists, case
// insensitively, and reports at when none of them fits.
func scanContraction(text string, at int) int {
	rest := text[at+1:]
	for _, suffix := range [...]string{"re", "ve", "ll", "s", "t", "m", "d"} {
		if len(rest) < len(suffix) {
			continue
		}
		if equalASCIIFold(rest[:len(suffix)], suffix) {
			return at + 1 + len(suffix)
		}
	}
	return at
}

// equalASCIIFold compares two ASCII strings ignoring case.
func equalASCIIFold(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		c := a[i]
		if 'A' <= c && c <= 'Z' {
			c += 'a' - 'A'
		}
		if c != b[i] {
			return false
		}
	}
	return true
}

// isSymbol reports whether r is matched by [^\s\p{L}\p{N}].
func isSymbol(r rune) bool {
	return !unicode.IsSpace(r) && !unicode.IsLetter(r) && !unicode.IsNumber(r)
}

// scanRun returns the end of the run of runes satisfying keep that begins at at.
func scanRun(text string, at int, keep func(rune) bool) int {
	end := at
	for end < len(text) {
		r, size := utf8.DecodeRuneInString(text[end:])
		if !keep(r) {
			break
		}
		end += size
	}
	return end
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
