package tokenizer

import (
	"slices"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/dlclark/regexp2"
)

// PreTokenizer splits text into chunks before BPE encoding
type PreTokenizer struct {
	pattern       *regexp2.Regexp
	specialTokens map[string]int

	// specialsByFirstByte lists the special tokens under their first byte, longest first.
	specialsByFirstByte map[byte][]string

	// splits replaces pattern with a HuggingFace Sequence of Split steps in Isolated mode.
	splits []*regexp2.Regexp

	// splitFast replaces splits for a pattern sequence this package recognizes.
	splitFast func(text string, emit func(string))

	// scan splits text without the regex engine when the pattern is one this
	// package recognizes. regexp2 backtracks, and pre-tokenization was measured
	// as the majority of the time spent counting; a direct scanner removes it.
	// TestFastScanMatchesRegexp holds both to identical output.
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
	p.indexSpecials()
	if pattern == cl100kPattern {
		p.scan = scanCl100k
	}
	return p, nil
}

// NewSplitPreTokenizer creates a pre-tokenizer that runs patterns as a
// HuggingFace Sequence of Split steps with the Isolated behavior: matches and
// the text between them both become pieces, and each pattern sees only the
// pieces the pattern before it produced.
func NewSplitPreTokenizer(patterns []string, specialTokens map[string]int) (*PreTokenizer, error) {
	p := &PreTokenizer{specialTokens: specialTokens}
	for _, pattern := range patterns {
		re, err := regexp2.Compile(pattern, regexp2.None)
		if err != nil {
			return nil, err
		}
		p.splits = append(p.splits, re)
	}
	if slices.Equal(patterns, deepseekSplits) {
		p.splitFast = splitDeepseek
	}
	p.indexSpecials()
	return p, nil
}

func (p *PreTokenizer) indexSpecials() {
	if len(p.specialTokens) == 0 {
		return
	}
	p.specialsByFirstByte = make(map[byte][]string)
	for token := range p.specialTokens {
		if token == "" {
			continue
		}
		p.specialsByFirstByte[token[0]] = append(p.specialsByFirstByte[token[0]], token)
	}
	for _, tokens := range p.specialsByFirstByte {
		sort.Slice(tokens, func(a, b int) bool {
			if len(tokens[a]) != len(tokens[b]) {
				return len(tokens[a]) > len(tokens[b])
			}
			return tokens[a] < tokens[b]
		})
	}
}

// Tokenize splits text into pre-tokens
// Returns list of (text, isSpecial) pairs
func (p *PreTokenizer) Tokenize(text string) ([]PreToken, error) {
	if len(p.specialTokens) > 0 {
		return p.tokenizeWithSpecial(text)
	}
	return p.tokenizeRegex(text)
}

// nextSpecial finds the leftmost special token in text, taking the longest
// one where several start at the same offset.
func (p *PreTokenizer) nextSpecial(text string) (int, string) {
	for i := 0; i < len(text); i++ {
		for _, token := range p.specialsByFirstByte[text[i]] {
			if strings.HasPrefix(text[i:], token) {
				return i, token
			}
		}
	}
	return -1, ""
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
		earliestPos, earliestToken := p.nextSpecial(remaining)

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
	if p.splitFast != nil {
		result := make([]PreToken, 0, len(text)/4+1)
		p.splitFast(text, func(piece string) { result = append(result, PreToken{Text: piece}) })
		return result, nil
	}
	if p.splits != nil {
		return p.tokenizeSplits(text)
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

// tokenizeSplits runs the Split steps in order over every piece.
func (p *PreTokenizer) tokenizeSplits(text string) ([]PreToken, error) {
	pieces := []string{text}
	for _, re := range p.splits {
		next := make([]string, 0, len(pieces))
		for _, piece := range pieces {
			var err error
			if next, err = splitIsolated(re, piece, next); err != nil {
				return nil, err
			}
		}
		pieces = next
	}
	result := make([]PreToken, len(pieces))
	for i, piece := range pieces {
		result[i] = PreToken{Text: piece}
	}
	return result, nil
}

// splitIsolated appends to out each match of re in s and each non-empty run of
// text between matches, in order.
func splitIsolated(re *regexp2.Regexp, s string, out []string) ([]string, error) {
	runes := []rune(s)
	last := 0
	match, err := re.FindRunesMatch(runes)
	for match != nil && err == nil {
		if match.Length > 0 {
			if match.Index > last {
				out = append(out, string(runes[last:match.Index]))
			}
			out = append(out, string(runes[match.Index:match.Index+match.Length]))
			last = match.Index + match.Length
		}
		match, err = re.FindNextMatch(match)
	}
	if err != nil {
		return nil, err
	}
	if last < len(runes) {
		out = append(out, string(runes[last:]))
	}
	return out, nil
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

// equalASCIIFold compares ASCII strings ignoring case.
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
