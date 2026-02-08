package tokenizer

import (
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
	if err := os.WriteFile(vocabPath, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write test vocab: %v", err)
	}

	tok, err := NewFromFile(vocabPath)
	if err != nil {
		t.Fatalf("NewFromFile error: %v", err)
	}

	if tok.VocabSize() != 2 {
		t.Errorf("VocabSize() = %d, want 2", tok.VocabSize())
	}

	// Test encoding
	tokens, err := tok.Encode("hello")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(tokens) != 1 || tokens[0] != 0 {
		t.Errorf("Encode(\"hello\") = %v, want [0]", tokens)
	}
}

func TestNewFromFileWithOptions(t *testing.T) {
	tmpDir := t.TempDir()
	vocabPath := filepath.Join(tmpDir, "test.tiktoken")

	content := "aGVsbG8= 0\nd29ybGQ= 1\n"
	if err := os.WriteFile(vocabPath, []byte(content), 0644); err != nil {
		t.Fatalf("Failed to write test vocab: %v", err)
	}

	specialTokens := map[string]int{"<test>": 100}

	tok, err := NewFromFile(vocabPath,
		WithPattern(`.+`),
		WithSpecialTokens(specialTokens),
		WithCacheSize(100),
	)
	if err != nil {
		t.Fatalf("NewFromFile error: %v", err)
	}

	// Test special token
	tokens, err := tok.Encode("<test>")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(tokens) != 1 || tokens[0] != 100 {
		t.Errorf("Encode(\"<test>\") = %v, want [100]", tokens)
	}
}

func TestNewFromFileNotFound(t *testing.T) {
	_, err := NewFromFile("/nonexistent/path/vocab.tiktoken")
	if err == nil {
		t.Error("Expected error for nonexistent file")
	}
}

func TestNewWithEncodingUnknown(t *testing.T) {
	_, err := NewWithEncoding("unknown_encoding")
	if err == nil {
		t.Error("Expected error for unknown encoding")
	}
}

func TestCountTokensCl100k(t *testing.T) {
	tok, err := New()
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

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
		if err != nil {
			t.Errorf("CountTokens(%q) error: %v", tc, err)
			continue
		}
		if count != len(tokens) {
			t.Errorf("CountTokens(%q) = %d, Encode returned %d tokens", tc, count, len(tokens))
		}
	}
}

func TestCountTokensGemma(t *testing.T) {
	tok, err := NewWithEncoding("gemma")
	if err != nil {
		t.Fatalf("NewWithEncoding error: %v", err)
	}

	testCases := []string{
		"hello",
		"Hello World",
		"The quick brown fox",
	}

	for _, tc := range testCases {
		tokens, _ := tok.Encode(tc)
		count, err := tok.CountTokens(tc)
		if err != nil {
			t.Errorf("CountTokens(%q) error: %v", tc, err)
			continue
		}
		if count != len(tokens) {
			t.Errorf("CountTokens(%q) = %d, Encode returned %d tokens", tc, count, len(tokens))
		}
	}
}

func TestPreTokenizerErrors(t *testing.T) {
	// Test with invalid regex pattern
	_, err := NewPreTokenizer("[invalid", nil)
	if err == nil {
		t.Error("Expected error for invalid regex pattern")
	}
}

func TestPreTokenizerNoSpecialTokens(t *testing.T) {
	pt, err := NewPreTokenizer(`.+`, nil)
	if err != nil {
		t.Fatalf("NewPreTokenizer error: %v", err)
	}

	tokens, err := pt.Tokenize("hello world")
	if err != nil {
		t.Fatalf("Tokenize error: %v", err)
	}

	if len(tokens) != 1 || tokens[0].Text != "hello world" {
		t.Errorf("Tokenize returned %v, want single token", tokens)
	}
}

func TestPreTokenizerWithSpecialTokens(t *testing.T) {
	specialTokens := map[string]int{"<s>": 0, "</s>": 1}
	pt, err := NewPreTokenizer(`.+`, specialTokens)
	if err != nil {
		t.Fatalf("NewPreTokenizer error: %v", err)
	}

	tokens, err := pt.Tokenize("<s>hello</s>")
	if err != nil {
		t.Fatalf("Tokenize error: %v", err)
	}

	if len(tokens) != 3 {
		t.Errorf("Expected 3 tokens, got %d: %v", len(tokens), tokens)
	}

	if !tokens[0].IsSpecial || tokens[0].Text != "<s>" {
		t.Errorf("First token should be special <s>, got %v", tokens[0])
	}
	if tokens[1].IsSpecial || tokens[1].Text != "hello" {
		t.Errorf("Second token should be regular 'hello', got %v", tokens[1])
	}
	if !tokens[2].IsSpecial || tokens[2].Text != "</s>" {
		t.Errorf("Third token should be special </s>, got %v", tokens[2])
	}
}

func TestLoadTiktokenErrors(t *testing.T) {
	// Test with invalid base64
	_, err := LoadTiktoken(stringReader("invalid!base64 0\n"))
	if err == nil {
		t.Error("Expected error for invalid base64")
	}

	// Test with invalid rank
	_, err = LoadTiktoken(stringReader("aGVsbG8= notanumber\n"))
	if err == nil {
		t.Error("Expected error for invalid rank")
	}

	// Test with wrong number of fields
	_, err = LoadTiktoken(stringReader("aGVsbG8=\n"))
	if err == nil {
		t.Error("Expected error for wrong number of fields")
	}
}

func TestLoadBinaryErrors(t *testing.T) {
	// Test with invalid magic
	_, err := LoadBinary([]byte("XXXX0000000000000000"))
	if err == nil {
		t.Error("Expected error for invalid magic")
	}

	// Test with unsupported version
	data := []byte("BPEV")
	data = append(data, 99, 0, 0, 0) // version 99
	data = append(data, 0, 0, 0, 0)  // vocab size
	data = append(data, 0, 0, 0, 0)  // num groups
	_, err = LoadBinary(data)
	if err == nil {
		t.Error("Expected error for unsupported version")
	}

	// Test with too short data
	_, err = LoadBinary([]byte("BPE"))
	if err == nil {
		t.Error("Expected error for too short data")
	}
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
	if err != nil {
		t.Fatalf("LoadBinary v1 error: %v", err)
	}
	if v.Size() != 1 {
		t.Errorf("Size() = %d, want 1", v.Size())
	}
	if id, ok := v.Encode([]byte("a")); !ok || id != 0 {
		t.Errorf("Encode('a') = %d, %v, want 0, true", id, ok)
	}
}

func TestBPEEncodeEmpty(t *testing.T) {
	tok, _ := New()
	tokens, err := tok.Encode("")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if tokens != nil {
		t.Errorf("Encode(\"\") = %v, want nil", tokens)
	}
}

func TestBPEDecodeEmpty(t *testing.T) {
	tok, _ := New()
	text, err := tok.Decode(nil)
	if err != nil {
		t.Fatalf("Decode error: %v", err)
	}
	if text != "" {
		t.Errorf("Decode(nil) = %q, want \"\"", text)
	}
}

func TestVocabularyMethods(t *testing.T) {
	v := NewVocabulary()

	// Test empty vocabulary
	if v.Size() != 0 {
		t.Errorf("Size() = %d, want 0", v.Size())
	}

	_, ok := v.Encode([]byte("test"))
	if ok {
		t.Error("Encode should return false for unknown token")
	}

	_, ok = v.Decode(0)
	if ok {
		t.Error("Decode should return false for unknown rank")
	}

	// Test HasMerges
	if v.HasMerges() {
		t.Error("HasMerges should return false for empty vocab")
	}

	// Test MergePriority
	_, ok = v.MergePriority(0, 1)
	if ok {
		t.Error("MergePriority should return false for no merges")
	}
}

func TestGemmaCountTokens(t *testing.T) {
	tok, err := NewWithEncoding("gemma")
	if err != nil {
		t.Fatalf("NewWithEncoding error: %v", err)
	}

	// Empty string
	count, err := tok.CountTokens("")
	if err != nil {
		t.Fatalf("CountTokens error: %v", err)
	}
	if count != 0 {
		t.Errorf("CountTokens(\"\") = %d, want 0", count)
	}

	// Single token
	count, _ = tok.CountTokens("hello")
	tokens, _ := tok.Encode("hello")
	if count != len(tokens) {
		t.Errorf("CountTokens mismatch: %d vs %d", count, len(tokens))
	}
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
		if err != nil {
			t.Fatalf("CountTokens error: %v", err)
		}
		if count <= 0 {
			t.Errorf("CountTokens returned %d, want > 0", count)
		}
	})

	t.Run("gemma", func(t *testing.T) {
		tok, _ := NewWithEncoding("gemma")
		// Use unique strings that won't be in cache
		count, err := tok.CountTokens("xyzzy12345abcde unique text here")
		if err != nil {
			t.Fatalf("CountTokens error: %v", err)
		}
		if count <= 0 {
			t.Errorf("CountTokens returned %d, want > 0", count)
		}
	})
}

func TestCountTokensSpecialTokens(t *testing.T) {
	tok, _ := New()

	// Test counting with special tokens
	count, err := tok.CountTokens("Hello<|endoftext|>World")
	if err != nil {
		t.Fatalf("CountTokens error: %v", err)
	}

	tokens, _ := tok.Encode("Hello<|endoftext|>World")
	if count != len(tokens) {
		t.Errorf("CountTokens = %d, Encode = %d tokens", count, len(tokens))
	}
}

func TestNewTokenizerCacheError(t *testing.T) {
	// Test with cache size of 0 should still work
	v := NewVocabulary()
	v.encoder["test"] = 0
	v.decoder[0] = []byte("test")

	_, err := newTokenizer(v, `.+`, nil, nil, nil, 1)
	if err != nil {
		t.Errorf("newTokenizer with small cache failed: %v", err)
	}
}

func TestBPECountTokensDirectly(t *testing.T) {
	// Test the BPE CountTokens method directly
	tok, _ := New()

	// Access internal bpe through encode/count comparison
	text := "a]b]c]d]e]f"  // unusual text to avoid cache
	tokens, _ := tok.Encode(text)
	count, _ := tok.CountTokens(text)

	if count != len(tokens) {
		t.Errorf("Count mismatch: CountTokens=%d, len(Encode)=%d", count, len(tokens))
	}
}

func TestGemmaCountTokensDirectly(t *testing.T) {
	tok, _ := NewWithEncoding("gemma")

	// Test with text that exercises the merge-based counting
	text := "unusual!@#text$%^here&*()"
	tokens, _ := tok.Encode(text)

	// Create new tokenizer to avoid cache
	tok2, _ := NewWithEncoding("gemma")
	count, _ := tok2.CountTokens(text)

	if count != len(tokens) {
		t.Errorf("Count mismatch: CountTokens=%d, len(Encode)=%d", count, len(tokens))
	}
}

func TestTokenizeRegexError(t *testing.T) {
	// Create pretokenizer and test edge cases
	pt, _ := NewPreTokenizer(`\w+`, nil)

	// Empty string
	tokens, err := pt.Tokenize("")
	if err != nil {
		t.Errorf("Tokenize(\"\") error: %v", err)
	}
	if len(tokens) != 0 {
		t.Errorf("Tokenize(\"\") = %v, want empty", tokens)
	}
}

func TestMultipleSpecialTokensInSequence(t *testing.T) {
	tok, _ := New()

	// Multiple special tokens in sequence
	text := "<|endoftext|><|endoftext|><|endoftext|>"
	tokens, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(tokens) != 3 {
		t.Errorf("Expected 3 tokens, got %d", len(tokens))
	}
}
