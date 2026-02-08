package tokenizer

import (
	"testing"
)

func TestGemmaNew(t *testing.T) {
	tok, err := NewWithEncoding("gemma")
	if err != nil {
		t.Fatalf("NewWithEncoding(gemma) error: %v", err)
	}

	if tok.VocabSize() != 262144 {
		t.Errorf("VocabSize() = %d, want 262144", tok.VocabSize())
	}
}

func TestGemmaTokenMappings(t *testing.T) {
	tok, err := NewWithEncoding("gemma")
	if err != nil {
		t.Fatalf("NewWithEncoding(gemma) error: %v", err)
	}

	// Space should encode to ▁ token (236743) after normalization
	spaceTokens, _ := tok.Encode(" ")
	if len(spaceTokens) != 1 || spaceTokens[0] != 236743 {
		t.Errorf("Encode(' ') = %v, want [236743]", spaceTokens)
	}

	// ▁ should also encode to token 236743
	underscoreTokens, _ := tok.Encode("▁")
	if len(underscoreTokens) != 1 || underscoreTokens[0] != 236743 {
		t.Errorf("Encode('▁') = %v, want [236743]", underscoreTokens)
	}

	// Decoding should convert ▁ back to space
	decoded, _ := tok.Decode([]int{236743})
	if decoded != " " {
		t.Errorf("Decode([236743]) = %q, want \" \"", decoded)
	}
}

func TestGemmaEncode(t *testing.T) {
	tok, err := NewWithEncoding("gemma")
	if err != nil {
		t.Fatalf("NewWithEncoding(gemma) error: %v", err)
	}

	// Test basic encoding - "hello" is a single token
	tokens, err := tok.Encode("hello")
	if err != nil {
		t.Fatalf("Encode error: %v", err)
	}
	if len(tokens) != 1 || tokens[0] != 23391 {
		t.Errorf("Encode(\"hello\") = %v, want [23391]", tokens)
	}
}

func TestGemmaRoundTrip(t *testing.T) {
	tok, err := NewWithEncoding("gemma")
	if err != nil {
		t.Fatalf("NewWithEncoding(gemma) error: %v", err)
	}

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
		if err != nil {
			t.Errorf("Encode(%q) error: %v", tc, err)
			continue
		}

		decoded, err := tok.Decode(tokens)
		if err != nil {
			t.Errorf("Decode error: %v", err)
			continue
		}

		if decoded != tc {
			t.Errorf("Round-trip failed: got %q, want %q", decoded, tc)
		}
	}
}

func TestGemmaSpecialTokens(t *testing.T) {
	tok, err := NewWithEncoding("gemma")
	if err != nil {
		t.Fatalf("NewWithEncoding(gemma) error: %v", err)
	}

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
		if err != nil {
			t.Errorf("Encode(%q) error: %v", st.token, err)
			continue
		}
		if len(tokens) != 1 || tokens[0] != st.id {
			t.Errorf("Encode(%q) = %v, want [%d]", st.token, tokens, st.id)
		}
	}
}

func BenchmarkGemmaEncode(b *testing.B) {
	tok, err := NewWithEncoding("gemma")
	if err != nil {
		b.Fatalf("NewWithEncoding error: %v", err)
	}

	text := "The▁quick▁brown▁fox▁jumps▁over▁the▁lazy▁dog."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(text)
	}
}
