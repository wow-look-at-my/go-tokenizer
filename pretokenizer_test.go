package tokenizer

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/dlclark/regexp2"
	"github.com/stretchr/testify/require"
)

// splitWithRegexp reproduces the pre-tokenization the regex engine performs, so
// the direct scanner can be held to it.
func splitWithRegexp(t *testing.T, re *regexp2.Regexp, text string) []string {
	t.Helper()

	var out []string
	m, err := re.FindStringMatch(text)
	require.NoError(t, err)
	for m != nil {
		out = append(out, m.String())
		m, err = re.FindNextMatch(m)
		require.NoError(t, err)
	}
	return out
}

func splitWithScanner(text string) []string {
	var out []string
	p := &PreTokenizer{scan: scanCl100k}
	for _, pt := range p.tokenizeScan(text) {
		out = append(out, pt.Text)
	}
	return out
}

// scannerCases covers each alternative of the pattern and the boundaries
// between them: contractions, a leading symbol on a word, digit grouping,
// symbol runs that swallow trailing line breaks, whitespace that ends in a line
// break, and a whitespace run that must hand its last character to a word.
var scannerCases = []string{
	"",
	"a",
	"Hello World",
	"don't can't I'LL we'Ve he'd she's it't",
	"'s'x'",
	"12 123 1234 12345 007",
	"foo(bar) [baz] {qux};",
	"a\n\nb",
	"a  \n  b",
	"trailing   ",
	"   leading",
	"a \t\v\f b",
	"tabs\t\tand\r\nnewlines\r\n\r\n",
	"emoji 🚀 and 👨‍👩‍👧‍👦 families",
	"naïve café Здравствуй 日本語テキスト 한국어",
	"mixed123abc456 def",
	"snake_case camelCase SCREAMING_CASE kebab-case",
	"url https://example.com/a?b=1&c=2#d",
	"json {\"key\": [1, 2.5, null], \"nested\": {\"x\": true}}",
	"   ",
	"\n",
	"\r\n",
	"\n   \n",
	" .",
	". ",
	"a'",
	"'",
	"·middot ¡inverted",
	"1️⃣2️⃣ ½ ³ Ⅻ",
}

// TestFastScanMatchesRegexp is the contract that lets the scanner replace the
// regex engine: for every input, the two must split identically. A divergence
// would silently change token counts everywhere the cl100k pattern is used.
func TestFastScanMatchesRegexp(t *testing.T) {
	re, err := regexp2.Compile(cl100kPattern, regexp2.None)
	require.NoError(t, err)

	for _, text := range scannerCases {
		require.Equal(t, splitWithRegexp(t, re, text), splitWithScanner(text), "input %q", text)
	}
}

// TestFastScanMatchesRegexpOnCorpus runs the same contract over the measured
// corpus, which carries prose, source, structured data and non-Latin scripts at
// a size the hand-written cases cannot reach.
func TestFastScanMatchesRegexpOnCorpus(t *testing.T) {
	re, err := regexp2.Compile(cl100kPattern, regexp2.None)
	require.NoError(t, err)

	f, err := os.Open("testdata/anthropic-counts.jsonl")
	require.NoError(t, err)
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1<<20), 1<<20)
	texts := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var c countCase
		require.NoError(t, json.Unmarshal([]byte(line), &c))
		require.Equal(t, splitWithRegexp(t, re, c.Text), splitWithScanner(c.Text), "sample %s", c.ID)
		texts++
	}
	require.NoError(t, scanner.Err())
	require.NotZero(t, texts)
}

// TestFastScanIsLossless proves the split can be rejoined into the input, so no
// scanner path can drop or duplicate a byte.
func TestFastScanIsLossless(t *testing.T) {
	for _, text := range scannerCases {
		require.Equal(t, text, strings.Join(splitWithScanner(text), ""), "input %q", text)
	}
}
