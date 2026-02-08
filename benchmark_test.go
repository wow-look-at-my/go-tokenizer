package tokenizer

import (
	"fmt"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Test‑data helpers
// ---------------------------------------------------------------------------

// englishProse returns roughly n bytes of English prose.
func englishProse(n int) string {
	const sentence = "The quick brown fox jumps over the lazy dog. "
	reps := (n / len(sentence)) + 1
	return strings.Repeat(sentence, reps)[:n]
}

// codeSnippet returns roughly n bytes of Go source code.
func codeSnippet(n int) string {
	const block = `func main() {
	for i := 0; i < 100; i++ {
		fmt.Printf("iteration %d: value=%v\n", i, compute(i))
	}
}
`
	reps := (n / len(block)) + 1
	return strings.Repeat(block, reps)[:n]
}

// unicodeText returns roughly n bytes of CJK‑heavy Unicode text.
func unicodeText(n int) string {
	const block = "日本語のテスト文章です。中文测试文本。한국어 테스트 문장입니다. "
	reps := (n / len(block)) + 1
	s := strings.Repeat(block, reps)
	// Truncate at rune boundary
	r := []rune(s)
	for i := len(r); i > 0; i-- {
		if len(string(r[:i])) <= n {
			return string(r[:i])
		}
	}
	return ""
}

// numberText returns roughly n bytes of numeric/mixed‑digit text.
func numberText(n int) string {
	const block = "There are 12345 items costing $67.89 each, totaling 838,102.05 units. "
	reps := (n / len(block)) + 1
	return strings.Repeat(block, reps)[:n]
}

// specialTokenText returns text interspersed with special tokens.
func specialTokenText(n int) string {
	const block = "Hello world<|endoftext|>Another segment<|fim_prefix|>code here<|fim_suffix|>more text. "
	reps := (n / len(block)) + 1
	return strings.Repeat(block, reps)[:n]
}

// whitespaceHeavy returns text with lots of whitespace and newlines.
func whitespaceHeavy(n int) string {
	const block = "word   another\t\ttabbed\n\nnewlines\n   indented   trailing   \n"
	reps := (n / len(block)) + 1
	return strings.Repeat(block, reps)[:n]
}

// ---------------------------------------------------------------------------
// Encode benchmarks — by input size
// ---------------------------------------------------------------------------

func BenchmarkEncodeBySize(b *testing.B) {
	tok, err := New()
	if err != nil {
		b.Fatalf("New() error: %v", err)
	}

	sizes := []struct {
		name string
		size int
	}{
		{"10B", 10},
		{"100B", 100},
		{"1KB", 1_000},
		{"10KB", 10_000},
		{"100KB", 100_000},
	}

	for _, s := range sizes {
		text := englishProse(s.size)
		b.Run(s.name, func(b *testing.B) {
			b.SetBytes(int64(len(text)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = tok.Encode(text)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Encode benchmarks — by content type
// ---------------------------------------------------------------------------

func BenchmarkEncodeByContent(b *testing.B) {
	tok, err := New()
	if err != nil {
		b.Fatalf("New() error: %v", err)
	}

	const size = 1_000

	cases := []struct {
		name string
		text string
	}{
		{"English", englishProse(size)},
		{"Code", codeSnippet(size)},
		{"Unicode", unicodeText(size)},
		{"Numbers", numberText(size)},
		{"SpecialTokens", specialTokenText(size)},
		{"Whitespace", whitespaceHeavy(size)},
	}

	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			b.SetBytes(int64(len(c.text)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = tok.Encode(c.text)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// CountTokens benchmarks — by input size
// ---------------------------------------------------------------------------

func BenchmarkCountTokensBySize(b *testing.B) {
	tok, err := New()
	if err != nil {
		b.Fatalf("New() error: %v", err)
	}

	sizes := []struct {
		name string
		size int
	}{
		{"10B", 10},
		{"100B", 100},
		{"1KB", 1_000},
		{"10KB", 10_000},
		{"100KB", 100_000},
	}

	for _, s := range sizes {
		text := englishProse(s.size)
		b.Run(s.name, func(b *testing.B) {
			b.SetBytes(int64(len(text)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = tok.CountTokens(text)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Decode benchmarks — by token count
// ---------------------------------------------------------------------------

func BenchmarkDecodeBySize(b *testing.B) {
	tok, err := New()
	if err != nil {
		b.Fatalf("New() error: %v", err)
	}

	sizes := []struct {
		name string
		size int
	}{
		{"10B", 10},
		{"100B", 100},
		{"1KB", 1_000},
		{"10KB", 10_000},
		{"100KB", 100_000},
	}

	for _, s := range sizes {
		text := englishProse(s.size)
		tokens, err := tok.Encode(text)
		if err != nil {
			b.Fatalf("Encode() error: %v", err)
		}
		b.Run(fmt.Sprintf("%s_%dtokens", s.name, len(tokens)), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = tok.Decode(tokens)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Gemma encoding benchmarks
// ---------------------------------------------------------------------------

func BenchmarkGemmaEncodeBySize(b *testing.B) {
	tok, err := NewWithEncoding("gemma")
	if err != nil {
		b.Fatalf("NewWithEncoding(gemma) error: %v", err)
	}

	sizes := []struct {
		name string
		size int
	}{
		{"10B", 10},
		{"100B", 100},
		{"1KB", 1_000},
		{"10KB", 10_000},
	}

	for _, s := range sizes {
		text := englishProse(s.size)
		b.Run(s.name, func(b *testing.B) {
			b.SetBytes(int64(len(text)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = tok.Encode(text)
			}
		})
	}
}

func BenchmarkGemmaEncodeByContent(b *testing.B) {
	tok, err := NewWithEncoding("gemma")
	if err != nil {
		b.Fatalf("NewWithEncoding(gemma) error: %v", err)
	}

	const size = 500

	cases := []struct {
		name string
		text string
	}{
		{"English", englishProse(size)},
		{"Code", codeSnippet(size)},
		{"Unicode", unicodeText(size)},
	}

	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			b.SetBytes(int64(len(c.text)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = tok.Encode(c.text)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Encoding comparison — cl100k_base vs gemma on the same text
// ---------------------------------------------------------------------------

func BenchmarkEncodeCompareEncodings(b *testing.B) {
	cl100k, err := New()
	if err != nil {
		b.Fatalf("New() error: %v", err)
	}
	gemma, err := NewWithEncoding("gemma")
	if err != nil {
		b.Fatalf("NewWithEncoding(gemma) error: %v", err)
	}

	text := englishProse(1_000)

	b.Run("cl100k_base", func(b *testing.B) {
		b.SetBytes(int64(len(text)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = cl100k.Encode(text)
		}
	})
	b.Run("gemma", func(b *testing.B) {
		b.SetBytes(int64(len(text)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = gemma.Encode(text)
		}
	})
}

// ---------------------------------------------------------------------------
// Parallel encode — exercises cache lock contention
// ---------------------------------------------------------------------------

func BenchmarkEncodeParallel(b *testing.B) {
	tok, err := New()
	if err != nil {
		b.Fatalf("New() error: %v", err)
	}

	text := englishProse(1_000)

	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = tok.Encode(text)
		}
	})
}

func BenchmarkCountTokensParallel(b *testing.B) {
	tok, err := New()
	if err != nil {
		b.Fatalf("New() error: %v", err)
	}

	text := englishProse(1_000)

	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, _ = tok.CountTokens(text)
		}
	})
}

// ---------------------------------------------------------------------------
// Tokenizer initialization
// ---------------------------------------------------------------------------

func BenchmarkNewTokenizer(b *testing.B) {
	b.Run("cl100k_base", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = New()
		}
	})
	b.Run("gemma", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = NewWithEncoding("gemma")
		}
	})
}

// ---------------------------------------------------------------------------
// Cache warm vs cold
// ---------------------------------------------------------------------------

func BenchmarkEncodeCacheWarm(b *testing.B) {
	tok, err := New()
	if err != nil {
		b.Fatalf("New() error: %v", err)
	}

	text := englishProse(1_000)
	// Warm the cache
	_, _ = tok.Encode(text)

	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = tok.Encode(text)
	}
}

func BenchmarkEncodeCacheCold(b *testing.B) {
	text := englishProse(1_000)

	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		tok, err := New()
		if err != nil {
			b.Fatalf("New() error: %v", err)
		}
		b.StartTimer()
		_, _ = tok.Encode(text)
	}
}

// ---------------------------------------------------------------------------
// Round‑trip (Encode then Decode)
// ---------------------------------------------------------------------------

func BenchmarkRoundTrip(b *testing.B) {
	tok, err := New()
	if err != nil {
		b.Fatalf("New() error: %v", err)
	}

	text := englishProse(1_000)

	b.SetBytes(int64(len(text)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokens, _ := tok.Encode(text)
		_, _ = tok.Decode(tokens)
	}
}

// ---------------------------------------------------------------------------
// Encode vs CountTokens (measure overhead difference)
// ---------------------------------------------------------------------------

func BenchmarkEncodeVsCountTokens(b *testing.B) {
	tok, err := New()
	if err != nil {
		b.Fatalf("New() error: %v", err)
	}

	text := englishProse(1_000)

	b.Run("Encode", func(b *testing.B) {
		b.SetBytes(int64(len(text)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = tok.Encode(text)
		}
	})
	b.Run("CountTokens", func(b *testing.B) {
		b.SetBytes(int64(len(text)))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = tok.CountTokens(text)
		}
	})
}

// ---------------------------------------------------------------------------
// VocabSize (should be trivially fast — sanity check)
// ---------------------------------------------------------------------------

func BenchmarkVocabSize(b *testing.B) {
	tok, err := New()
	if err != nil {
		b.Fatalf("New() error: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.VocabSize()
	}
}
