package tokenizer

import (
	"testing"
	"unicode"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func preTokenTexts(t *testing.T, p *PreTokenizer, text string) []string {
	t.Helper()
	pts, err := p.Tokenize(text)
	require.NoError(t, err)
	out := make([]string, len(pts))
	for i, pt := range pts {
		out[i] = pt.Text
		if pt.IsSpecial {
			out[i] = "special:" + pt.Text
		}
	}
	return out
}

func TestSplitPreTokenizerIsolated(t *testing.T) {
	p, err := NewSplitPreTokenizer(deepseekSplits, nil)
	require.NoError(t, err)

	cases := map[string][]string{
		"Hello world": {"Hello", " world"},
		"12345":       {"123", "45"},
		// The digit split runs first, so the trailing spaces end their piece and stay whole instead of lending one.
		"a  1":       {"a", "  ", "1"},
		"你好world":    {"你好", "world"},
		"x.y":        {"x", ".y"},
		"don't stop": {"don", "'t", " stop"},
		"a\n\nb":     {"a", "\n\n", "b"},
		"":           {},
	}
	for in, want := range cases {
		assert.Equal(t, want, preTokenTexts(t, p, in), "input %q", in)
	}
}

// sameClasses reports whether the Go tables and the HuggingFace tables put
// every rune of s in the same regex classes.
func sameClasses(s string) bool {
	for _, r := range s {
		if unicode.IsLetter(r) != hfL.has(r) || unicode.IsMark(r) != hfM.has(r) ||
			unicode.IsNumber(r) != hfN.has(r) || unicode.IsPunct(r) != hfP.has(r) ||
			unicode.IsSymbol(r) != hfS.has(r) || unicode.IsSpace(r) != hfSpace.has(r) {
			return false
		}
	}
	return true
}

func TestDeepseekScanMatchesRegexp(t *testing.T) {
	p, err := NewSplitPreTokenizer(deepseekSplits, nil)
	require.NoError(t, err)
	require.NotNil(t, p.splitFast)

	compared := 0
	for _, row := range loadDeepseekGolden(t) {
		if !sameClasses(row.Text) {
			continue
		}
		compared++
		want, err := p.tokenizeSplits(row.Text)
		require.NoError(t, err)
		got, err := p.tokenizeRegex(row.Text)
		require.NoError(t, err)
		require.Equal(t, want, got, "row %s: %q", row.ID, row.Text)
	}
	assert.NotZero(t, compared)
}

func TestHFClassesDifferFromGo(t *testing.T) {
	// U+32BD4 is a letter in the Go tables but unassigned in HuggingFace's.
	assert.True(t, unicode.IsLetter('\U00032BD4'))
	assert.False(t, hfL.has('\U00032BD4'))
	assert.True(t, hfL.has('a'))
	assert.True(t, hfL.has('\U00020000'))
	assert.True(t, hfSpace.has('　'))
	assert.False(t, hfSpace.has('​'))
}

func TestParseRangesRejectsBadInput(t *testing.T) {
	assert.Panics(t, func() { parseRanges("L", []string{"zz"}) })
	assert.Panics(t, func() { parseRanges("L", []string{"20-10"}) })
	table := parseRanges("L", []string{"41-5a", "ff00-10010", "20000"})
	assert.True(t, unicode.Is(table, 'Q'))
	assert.True(t, unicode.Is(table, '\U00010005'))
	assert.True(t, unicode.Is(table, '￿'))
	assert.True(t, unicode.Is(table, '\U00020000'))
	assert.False(t, unicode.Is(table, '\U00020001'))
}

func TestSplitPreTokenizerBadPattern(t *testing.T) {
	_, err := NewSplitPreTokenizer([]string{"("}, nil)
	assert.Error(t, err)
}

func TestSpecialTokensLeftmostLongest(t *testing.T) {
	specials := map[string]int{"<｜": 1, "<｜User｜>": 2, "<think>": 3}
	p, err := NewSplitPreTokenizer(deepseekSplits, specials)
	require.NoError(t, err)

	assert.Equal(t, []string{"a", "special:<｜User｜>", "b", "special:<think>"}, preTokenTexts(t, p, "a<｜User｜>b<think>"))
	assert.Equal(t, []string{"special:<｜", "Use"}, preTokenTexts(t, p, "<｜Use"))
}

// byteLevelTestBPE builds a byte-level vocabulary over the bytes 'a' and 'b'
// with the merges a+a, then a+b.
func byteLevelTestBPE() *BPE {
	v := NewVocabulary()
	for id, tok := range []string{"a", "b", "aa", "ab"} {
		v.encoder[tok] = id
		v.decoder[id] = []byte(tok)
	}
	v.merges[mergePairKey(0, 0)] = 0
	v.merges[mergePairKey(0, 1)] = 1
	return &BPE{vocab: v, byteLevel: true}
}

func TestByteLevelBPEMergesLeftmostFirst(t *testing.T) {
	b := byteLevelTestBPE()
	// Equal merges resolve from the left: "aaa" is "aa" then "a".
	assert.Equal(t, []int{2, 0}, b.Encode([]byte("aaa")))
	// a+a outranks a+b, so "aab" keeps the b alone.
	assert.Equal(t, []int{2, 1}, b.Encode([]byte("aab")))
	assert.Equal(t, 2, b.CountTokens([]byte("aab")))
	assert.Equal(t, []int{3}, b.Encode([]byte("ab")))
	assert.Equal(t, "aab", string(b.Decode([]int{2, 1})))
}
