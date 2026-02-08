package tokenizer

import (
	"strings"
	"testing"
)

func TestNew(t *testing.T) {
	tok, err := New()
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

	if tok.VocabSize() == 0 {
		t.Error("VocabSize() returned 0")
	}
}

func TestEncodeHello(t *testing.T) {
	tok, err := New()
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

	tokens, err := tok.Encode("hello")
	if err != nil {
		t.Fatalf("Encode() error: %v", err)
	}

	// cl100k_base: "hello" -> [15339]
	expected := []int{15339}
	if !intSliceEqual(tokens, expected) {
		t.Errorf("Encode(\"hello\") = %v, want %v", tokens, expected)
	}
}

func TestEncodeHelloWorld(t *testing.T) {
	tok, err := New()
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

	tokens, err := tok.Encode("Hello World")
	if err != nil {
		t.Fatalf("Encode() error: %v", err)
	}

	// cl100k_base: "Hello World" -> [9906, 4435]
	expected := []int{9906, 4435}
	if !intSliceEqual(tokens, expected) {
		t.Errorf("Encode(\"Hello World\") = %v, want %v", tokens, expected)
	}
}

func TestEncodeDecode(t *testing.T) {
	tok, err := New()
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

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
			if err != nil {
				t.Fatalf("Encode() error: %v", err)
			}

			decoded, err := tok.Decode(tokens)
			if err != nil {
				t.Fatalf("Decode() error: %v", err)
			}

			if decoded != tc {
				t.Errorf("Round-trip failed: got %q, want %q", decoded, tc)
			}
		})
	}
}

func TestCountTokens(t *testing.T) {
	tok, err := New()
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

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
			if err != nil {
				t.Fatalf("CountTokens() error: %v", err)
			}

			if count != tc.expected {
				t.Errorf("CountTokens(%q) = %d, want %d", tc.text, count, tc.expected)
			}
		})
	}
}

func TestCountTokensMatchesEncode(t *testing.T) {
	tok, err := New()
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

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
			if err != nil {
				t.Fatalf("Encode() error: %v", err)
			}

			count, err := tok.CountTokens(tc)
			if err != nil {
				t.Fatalf("CountTokens() error: %v", err)
			}

			if count != len(tokens) {
				t.Errorf("CountTokens(%q) = %d, but Encode() returned %d tokens", tc, count, len(tokens))
			}
		})
	}
}

func TestSpecialTokens(t *testing.T) {
	tok, err := New()
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

	tokens, err := tok.Encode("<|endoftext|>")
	if err != nil {
		t.Fatalf("Encode() error: %v", err)
	}

	// <|endoftext|> should be token 100257
	expected := []int{100257}
	if !intSliceEqual(tokens, expected) {
		t.Errorf("Encode(\"<|endoftext|>\") = %v, want %v", tokens, expected)
	}
}

func TestMixedSpecialTokens(t *testing.T) {
	tok, err := New()
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

	text := "Hello<|endoftext|>World"
	tokens, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode() error: %v", err)
	}

	// Should have 3 tokens: Hello, <|endoftext|>, World
	if len(tokens) != 3 {
		t.Errorf("Encode(%q) returned %d tokens, want 3", text, len(tokens))
	}

	// Middle token should be the special token
	if tokens[1] != 100257 {
		t.Errorf("Middle token = %d, want 100257", tokens[1])
	}
}

func TestEmptyInput(t *testing.T) {
	tok, err := New()
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

	tokens, err := tok.Encode("")
	if err != nil {
		t.Fatalf("Encode() error: %v", err)
	}
	if len(tokens) != 0 {
		t.Errorf("Encode(\"\") = %v, want []", tokens)
	}

	count, err := tok.CountTokens("")
	if err != nil {
		t.Fatalf("CountTokens() error: %v", err)
	}
	if count != 0 {
		t.Errorf("CountTokens(\"\") = %d, want 0", count)
	}

	decoded, err := tok.Decode(nil)
	if err != nil {
		t.Fatalf("Decode() error: %v", err)
	}
	if decoded != "" {
		t.Errorf("Decode(nil) = %q, want \"\"", decoded)
	}
}

func TestVocabSize(t *testing.T) {
	tok, err := New()
	if err != nil {
		t.Fatalf("New() error: %v", err)
	}

	// cl100k_base has 100256 base tokens + 5 special tokens = 100261
	size := tok.VocabSize()
	if size < 100256 {
		t.Errorf("VocabSize() = %d, want >= 100256", size)
	}
}

func TestLoadVocabulary(t *testing.T) {
	// Test loading vocabulary from a reader
	vocabData := `SGVsbG8= 0
V29ybGQ= 1
`
	vocab, err := LoadTiktoken(strings.NewReader(vocabData))
	if err != nil {
		t.Fatalf("LoadTiktoken() error: %v", err)
	}

	if vocab.Size() != 2 {
		t.Errorf("vocab.Size() = %d, want 2", vocab.Size())
	}

	// "Hello" -> 0
	rank, ok := vocab.Encode([]byte("Hello"))
	if !ok || rank != 0 {
		t.Errorf("vocab.Encode(\"Hello\") = %d, %v, want 0, true", rank, ok)
	}

	// 1 -> "World"
	bytes, ok := vocab.Decode(1)
	if !ok || string(bytes) != "World" {
		t.Errorf("vocab.Decode(1) = %q, %v, want \"World\", true", bytes, ok)
	}
}

func BenchmarkEncode(b *testing.B) {
	tok, err := New()
	if err != nil {
		b.Fatalf("New() error: %v", err)
	}

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
	if err != nil {
		b.Fatalf("New() error: %v", err)
	}

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
	if err != nil {
		b.Fatalf("New() error: %v", err)
	}

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
