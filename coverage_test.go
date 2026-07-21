package tokenizer

import (
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"os"
	"path/filepath"
	"testing"
)

func TestNewFromFile(t *testing.T) {
	// Create a temp tiktoken file
	tmpDir := t.TempDir()
	vocabPath := filepath.Join(tmpDir, "test.tiktoken")

	// Write a simple vocab: "hello" = 0, "world" = 1
	content := "aGVsbG8= 0\nd29ybGQ= 1\n"
	require.NoError(t, os.WriteFile(vocabPath, []byte(content), 0644))

	tok, err := NewFromFile(vocabPath)
	require.Nil(t, err)

	assert.Equal(t, 2, tok.VocabSize())

	// Test encoding
	tokens, err := tok.Encode("hello")
	require.Nil(t, err)

	assert.False(t, len(tokens) != 1 || tokens[0] != 0)

}

func TestNewFromFileWithOptions(t *testing.T) {
	tmpDir := t.TempDir()
	vocabPath := filepath.Join(tmpDir, "test.tiktoken")

	content := "aGVsbG8= 0\nd29ybGQ= 1\n"
	require.NoError(t, os.WriteFile(vocabPath, []byte(content), 0644))

	specialTokens := map[string]int{"<test>": 100}

	tok, err := NewFromFile(vocabPath,
		WithPattern(`.+`),
		WithSpecialTokens(specialTokens),
		WithCacheSize(100),
	)
	require.Nil(t, err)

	// Test special token
	tokens, err := tok.Encode("<test>")
	require.Nil(t, err)

	assert.False(t, len(tokens) != 1 || tokens[0] != 100)

}

func TestNewFromFileNotFound(t *testing.T) {
	_, err := NewFromFile("/nonexistent/path/vocab.tiktoken")
	assert.NotNil(t, err)

}

func TestNewWithEncodingUnknown(t *testing.T) {
	_, err := NewWithEncoding("unknown_encoding")
	assert.NotNil(t, err)

}

func TestCountTokensCl100k(t *testing.T) {
	tok, err := New()
	require.Nil(t, err)

	// Test count matches encode length for various inputs
	testCases := []string{
		"hello",
		"Hello World",
		"The quick brown fox",
		"12345",
		"Special: !@#$%",
	}

	for _, tc := range testCases {
		tokens, _ := tok.Encode(tc)
		count, err := tok.CountTokens(tc)
		assert.Nil(t, err)

		assert.Equal(t, len(tokens), count)

	}
}

func TestCountTokensGemma(t *testing.T) {
	tok, err := NewWithEncoding("gemma")
	require.Nil(t, err)

	testCases := []string{
		"hello",
		"Hello World",
		"The quick brown fox",
	}

	for _, tc := range testCases {
		tokens, _ := tok.Encode(tc)
		count, err := tok.CountTokens(tc)
		assert.Nil(t, err)

		assert.Equal(t, len(tokens), count)

	}
}

func TestPreTokenizerErrors(t *testing.T) {
	// Test with invalid regex pattern
	_, err := NewPreTokenizer("[invalid", nil)
	assert.NotNil(t, err)

}

func TestPreTokenizerNoSpecialTokens(t *testing.T) {
	pt, err := NewPreTokenizer(`.+`, nil)
	require.Nil(t, err)

	tokens, err := pt.Tokenize("hello world")
	require.Nil(t, err)

	assert.False(t, len(tokens) != 1 || tokens[0].Text != "hello world")

}

func TestPreTokenizerWithSpecialTokens(t *testing.T) {
	specialTokens := map[string]int{"<s>": 0, "</s>": 1}
	pt, err := NewPreTokenizer(`.+`, specialTokens)
	require.Nil(t, err)

	tokens, err := pt.Tokenize("<s>hello</s>")
	require.Nil(t, err)

	assert.Equal(t, 3, len(tokens))

	assert.False(t, !tokens[0].IsSpecial || tokens[0].Text != "<s>")

	assert.False(t, tokens[1].IsSpecial || tokens[1].Text != "hello")

	assert.False(t, !tokens[2].IsSpecial || tokens[2].Text != "</s>")

}

func TestLoadTiktokenErrors(t *testing.T) {
	// Test with invalid base64
	_, err := LoadTiktoken(stringReader("invalid!base64 0\n"))
	assert.NotNil(t, err)

	// Test with invalid rank
	_, err = LoadTiktoken(stringReader("aGVsbG8= notanumber\n"))
	assert.NotNil(t, err)

	// Test with wrong number of fields
	_, err = LoadTiktoken(stringReader("aGVsbG8=\n"))
	assert.NotNil(t, err)

}

func TestLoadBinaryErrors(t *testing.T) {
	// Test with invalid magic
	_, err := LoadBinary([]byte("XXXX0000000000000000"))
	assert.NotNil(t, err)

	// Test with unsupported version
	data := []byte("BPEV")
	data = append(data, 99, 0, 0, 0) // version 99
	data = append(data, 0, 0, 0, 0)  // vocab size
	data = append(data, 0, 0, 0, 0)  // num groups
	_, err = LoadBinary(data)
	assert.NotNil(t, err)

	// Test with too short data
	_, err = LoadBinary([]byte("BPE"))
	assert.NotNil(t, err)

}

func TestLoadBinaryV1(t *testing.T) {
	// Build a minimal v1 format binary
	// Header: BPEV + version 1 + vocab size 1 + num groups 1
	data := []byte("BPEV")
	data = append(data, 1, 0, 0, 0) // version 1
	data = append(data, 1, 0, 0, 0) // vocab size 1
	data = append(data, 1, 0, 0, 0) // num groups 1
	// Group: length 1, count 1
	data = append(data, 1, 0)       // token length 1
	data = append(data, 1, 0, 0, 0) // count 1
	// Token: "a" with id 0
	data = append(data, 'a')        // token
	data = append(data, 0, 0, 0, 0) // id 0
	// Merges: 0
	data = append(data, 0, 0, 0, 0)

	v, err := LoadBinary(data)
	require.Nil(t, err)

	assert.Equal(t, 1, v.Size())

	id, ok := v.Encode([]byte("a"))
	assert.False(t, !ok || id != 0)

}

func TestBPEEncodeEmpty(t *testing.T) {
	tok, _ := New()
	tokens, err := tok.Encode("")
	require.Nil(t, err)

	assert.Nil(t, tokens)

}

func TestBPEDecodeEmpty(t *testing.T) {
	tok, _ := New()
	text, err := tok.Decode(nil)
	require.Nil(t, err)

	assert.Equal(t, "", text)

}

func TestVocabularyMethods(t *testing.T) {
	v := NewVocabulary()

	// Test empty vocabulary
	assert.Equal(t, 0, v.Size())

	_, ok := v.Encode([]byte("test"))
	assert.False(t, ok)

	_, ok = v.Decode(0)
	assert.False(t, ok)

	// Test HasMerges
	assert.False(t, v.HasMerges())

	// Test MergePriority
	_, ok = v.MergePriority(0, 1)
	assert.False(t, ok)

}

func TestGemmaCountTokens(t *testing.T) {
	tok, err := NewWithEncoding("gemma")
	require.Nil(t, err)

	// Empty string
	count, err := tok.CountTokens("")
	require.Nil(t, err)

	assert.Equal(t, 0, count)

	// Single token
	count, _ = tok.CountTokens("hello")
	tokens, _ := tok.Encode("hello")
	assert.Equal(t, len(tokens), count)

}

// stringReader is a helper to create an io.Reader from a string
type stringReader string

func (s stringReader) Read(p []byte) (n int, err error) {
	return copy(p, s), nil
}

func TestCountTokensWithoutCache(t *testing.T) {
	// Test CountTokens on fresh tokenizer without prior Encode calls
	// This exercises the count path without cache hits

	t.Run("cl100k", func(t *testing.T) {
		tok, _ := New()
		// Use unique strings that won't be in cache
		count, err := tok.CountTokens("xyzzy12345abcde")
		require.Nil(t, err)

		assert.Greater(t, count, 0)

	})

	t.Run("gemma", func(t *testing.T) {
		tok, _ := NewWithEncoding("gemma")
		// Use unique strings that won't be in cache
		count, err := tok.CountTokens("xyzzy12345abcde unique text here")
		require.Nil(t, err)

		assert.Greater(t, count, 0)

	})
}

func TestCountTokensSpecialTokens(t *testing.T) {
	tok, _ := New()

	// Test counting with special tokens
	count, err := tok.CountTokens("Hello<|endoftext|>World")
	require.Nil(t, err)

	tokens, _ := tok.Encode("Hello<|endoftext|>World")
	assert.Equal(t, len(tokens), count)

}

func TestNewTokenizerCacheError(t *testing.T) {
	// Test with cache size of 0 should still work
	v := NewVocabulary()
	v.encoder["test"] = 0
	v.decoder[0] = []byte("test")

	_, err := newTokenizer(v, `.+`, nil, nil, nil, 1)
	assert.Nil(t, err)

}

func TestBPECountTokensDirectly(t *testing.T) {
	// Test the BPE CountTokens method directly
	tok, _ := New()

	// Access internal bpe through encode/count comparison
	text := "a]b]c]d]e]f" // unusual text to avoid cache
	tokens, _ := tok.Encode(text)
	count, _ := tok.CountTokens(text)

	assert.Equal(t, len(tokens), count)

}

func TestGemmaCountTokensDirectly(t *testing.T) {
	tok, _ := NewWithEncoding("gemma")

	// Test with text that exercises the merge-based counting
	text := "unusual!@#text$%^here&*()"
	tokens, _ := tok.Encode(text)

	// Create new tokenizer to avoid cache
	tok2, _ := NewWithEncoding("gemma")
	count, _ := tok2.CountTokens(text)

	assert.Equal(t, len(tokens), count)

}

func TestTokenizeRegexError(t *testing.T) {
	// Create pretokenizer and test edge cases
	pt, _ := NewPreTokenizer(`\w+`, nil)

	// Empty string
	tokens, err := pt.Tokenize("")
	assert.Nil(t, err)

	assert.Equal(t, 0, len(tokens))

}

func TestMultipleSpecialTokensInSequence(t *testing.T) {
	tok, _ := New()

	// Multiple special tokens in sequence
	text := "<|endoftext|><|endoftext|><|endoftext|>"
	tokens, err := tok.Encode(text)
	require.Nil(t, err)

	assert.Equal(t, 3, len(tokens))

}
