package tokenizer

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"strings"
	"testing"
)

func TestGemmaNew(t *testing.T) {
	tok, err := NewWithEncoding("gemma")
	require.Nil(t, err)

	assert.Equal(t, 262144, tok.VocabSize())

}

func TestGemmaTokenMappings(t *testing.T) {
	tok, err := NewWithEncoding("gemma")
	require.Nil(t, err)

	// Space should encode to ▁ token (236743) after normalization
	spaceTokens, _ := tok.Encode(" ")
	assert.False(t, len(spaceTokens) != 1 || spaceTokens[0] != 236743)

	// ▁ should also encode to token 236743
	underscoreTokens, _ := tok.Encode("▁")
	assert.False(t, len(underscoreTokens) != 1 || underscoreTokens[0] != 236743)

	// Decoding should convert ▁ back to space
	decoded, _ := tok.Decode([]int{236743})
	assert.Equal(t, " ", decoded)

}

func TestGemmaEncode(t *testing.T) {
	tok, err := NewWithEncoding("gemma")
	require.Nil(t, err)

	// Test basic encoding - "hello" is a single token
	tokens, err := tok.Encode("hello")
	require.Nil(t, err)

	assert.False(t, len(tokens) != 1 || tokens[0] != 23391)

}

func TestGemmaRoundTrip(t *testing.T) {
	tok, err := NewWithEncoding("gemma")
	require.Nil(t, err)

	testCases := []string{
		"hello",
		"world",
		"testing",
		"Hello World",
		"The quick brown fox jumps over the lazy dog.",
		"Unicode: 日本語 中文",
	}

	for _, tc := range testCases {
		tokens, err := tok.Encode(tc)
		assert.Nil(t, err)

		decoded, err := tok.Decode(tokens)
		assert.Nil(t, err)

		assert.Equal(t, tc, decoded)

	}
}

func TestGemmaSpecialTokens(t *testing.T) {
	tok, err := NewWithEncoding("gemma")
	require.Nil(t, err)

	specialTests := []struct {
		token string
		id    int
	}{
		{"<pad>", 0},
		{"<eos>", 1},
		{"<bos>", 2},
		{"<unk>", 3},
	}

	for _, st := range specialTests {
		tokens, err := tok.Encode(st.token)
		assert.Nil(t, err)

		assert.False(t, len(tokens) != 1 || tokens[0] != st.id)

	}
}

func BenchmarkGemmaEncode(b *testing.B) {
	tok, err := NewWithEncoding("gemma")
	require.Nil(b, err)

	text := "The▁quick▁brown▁fox▁jumps▁over▁the▁lazy▁dog."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(text)
	}
}

func benchGemmaCountTokens(b *testing.B, size int) {
	b.Helper()
	tok, err := NewWithEncoding("gemma")
	require.Nil(b, err)

	base := "The quick brown fox jumps over the lazy dog. "
	var sb strings.Builder
	for sb.Len() < size {
		sb.WriteString(base)
	}
	text := sb.String()[:size]

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = tok.CountTokens(text)
	}
}

func BenchmarkGemmaCountTokens_1KB(b *testing.B)   { benchGemmaCountTokens(b, 1_000) }
func BenchmarkGemmaCountTokens_10KB(b *testing.B)  { benchGemmaCountTokens(b, 10_000) }
func BenchmarkGemmaCountTokens_100KB(b *testing.B) { benchGemmaCountTokens(b, 100_000) }
