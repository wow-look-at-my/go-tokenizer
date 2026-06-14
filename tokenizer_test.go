package tokenizer

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"strings"
	"testing"
)

func TestNew(t *testing.T) {
	tok, err := New()
	require.Nil(t, err)

	assert.NotEqual(t, 0, tok.VocabSize())

}

func TestEncodeHello(t *testing.T) {
	tok, err := New()
	require.Nil(t, err)

	tokens, err := tok.Encode("hello")
	require.Nil(t, err)

	// cl100k_base: "hello" -> [15339]
	expected := []int{15339}
	assert.True(t, intSliceEqual(tokens, expected))

}

func TestEncodeHelloWorld(t *testing.T) {
	tok, err := New()
	require.Nil(t, err)

	tokens, err := tok.Encode("Hello World")
	require.Nil(t, err)

	// cl100k_base: "Hello World" -> [9906, 4435]
	expected := []int{9906, 4435}
	assert.True(t, intSliceEqual(tokens, expected))

}

func TestEncodeDecode(t *testing.T) {
	tok, err := New()
	require.Nil(t, err)

	testCases := []string{
		"hello",
		"Hello World",
		"The quick brown fox jumps over the lazy dog.",
		"1234567890",
		"Special chars: !@#$%^&*()",
		"Unicode: 日本語 中文 한국어",
		"Code: func main() { fmt.Println(\"Hello\") }",
		"",
	}

	for _, tc := range testCases {
		t.Run(tc, func(t *testing.T) {
			tokens, err := tok.Encode(tc)
			require.Nil(t, err)

			decoded, err := tok.Decode(tokens)
			require.Nil(t, err)

			assert.Equal(t, tc, decoded)

		})
	}
}

func TestCountTokens(t *testing.T) {
	tok, err := New()
	require.Nil(t, err)

	testCases := []struct {
		text     string
		expected int
	}{
		{"hello", 1},
		{"Hello World", 2},
		{"", 0},
	}

	for _, tc := range testCases {
		t.Run(tc.text, func(t *testing.T) {
			count, err := tok.CountTokens(tc.text)
			require.Nil(t, err)

			assert.Equal(t, tc.expected, count)

		})
	}
}

func TestCountTokensMatchesEncode(t *testing.T) {
	tok, err := New()
	require.Nil(t, err)

	testCases := []string{
		"hello",
		"Hello World",
		"The quick brown fox jumps over the lazy dog.",
		"1234567890",
		"Special chars: !@#$%^&*()",
		"Unicode: 日本語 中文 한국어",
		"Code: func main() { fmt.Println(\"Hello\") }",
		"A longer piece of text that might have more tokens and exercises the algorithm more thoroughly.",
	}

	for _, tc := range testCases {
		t.Run(tc, func(t *testing.T) {
			tokens, err := tok.Encode(tc)
			require.Nil(t, err)

			count, err := tok.CountTokens(tc)
			require.Nil(t, err)

			assert.Equal(t, len(tokens), count)

		})
	}
}

func TestSpecialTokens(t *testing.T) {
	tok, err := New()
	require.Nil(t, err)

	tokens, err := tok.Encode("<|endoftext|>")
	require.Nil(t, err)

	// <|endoftext|> should be token 100257
	expected := []int{100257}
	assert.True(t, intSliceEqual(tokens, expected))

}

func TestMixedSpecialTokens(t *testing.T) {
	tok, err := New()
	require.Nil(t, err)

	text := "Hello<|endoftext|>World"
	tokens, err := tok.Encode(text)
	require.Nil(t, err)

	// Should have 3 tokens: Hello, <|endoftext|>, World
	assert.Equal(t, 3, len(tokens))

	// Middle token should be the special token
	assert.Equal(t, 100257, tokens[1])

}

func TestEmptyInput(t *testing.T) {
	tok, err := New()
	require.Nil(t, err)

	tokens, err := tok.Encode("")
	require.Nil(t, err)

	assert.Equal(t, 0, len(tokens))

	count, err := tok.CountTokens("")
	require.Nil(t, err)

	assert.Equal(t, 0, count)

	decoded, err := tok.Decode(nil)
	require.Nil(t, err)

	assert.Equal(t, "", decoded)

}

func TestVocabSize(t *testing.T) {
	tok, err := New()
	require.Nil(t, err)

	// cl100k_base has 100256 base tokens + 5 special tokens = 100261
	size := tok.VocabSize()
	assert.GreaterOrEqual(t, size, 100256)

}

func TestLoadVocabulary(t *testing.T) {
	// Test loading vocabulary from a reader
	vocabData := `SGVsbG8= 0
V29ybGQ= 1
`
	vocab, err := LoadTiktoken(strings.NewReader(vocabData))
	require.Nil(t, err)

	assert.Equal(t, 2, vocab.Size())

	// "Hello" -> 0
	rank, ok := vocab.Encode([]byte("Hello"))
	assert.False(t, !ok || rank != 0)

	// 1 -> "World"
	bytes, ok := vocab.Decode(1)
	assert.False(t, !ok || string(bytes) != "World")

}

func BenchmarkEncode(b *testing.B) {
	tok, err := New()
	require.Nil(b, err)

	text := "The quick brown fox jumps over the lazy dog. " +
		"This is a longer piece of text that will exercise the tokenizer " +
		"with various words, numbers like 12345, and punctuation!"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(text)
	}
}

func BenchmarkCountTokens(b *testing.B) {
	tok, err := New()
	require.Nil(b, err)

	text := "The quick brown fox jumps over the lazy dog. " +
		"This is a longer piece of text that will exercise the tokenizer " +
		"with various words, numbers like 12345, and punctuation!"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = tok.CountTokens(text)
	}
}

func BenchmarkEncodeLongText(b *testing.B) {
	tok, err := New()
	require.Nil(b, err)

	// Create a longer text
	var sb strings.Builder
	for i := 0; i < 100; i++ {
		sb.WriteString("The quick brown fox jumps over the lazy dog. ")
	}
	text := sb.String()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(text)
	}
}

func intSliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
