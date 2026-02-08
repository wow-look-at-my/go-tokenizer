package tokenizer

import "strings"

// EncodingConfig holds configuration for a specific encoding
type EncodingConfig struct {
	Name          string
	Pattern       string
	SpecialTokens map[string]int
	IsBinary      bool                // true if vocab is in binary format
	Normalize     func(string) string // text normalization before encoding
	Denormalize   func(string) string // text denormalization after decoding
}

// cl100k_base pattern from OpenAI tiktoken
// This pattern handles contractions, words, numbers, punctuation, and whitespace
const cl100kPattern = `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

// cl100k_base special tokens
var cl100kSpecialTokens = map[string]int{
	"<|endoftext|>":   100257,
	"<|fim_prefix|>":  100258,
	"<|fim_middle|>":  100259,
	"<|fim_suffix|>":  100260,
	"<|endofprompt|>": 100276,
}

// p50k_base pattern
const p50kPattern = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`

// p50k_base special tokens
var p50kSpecialTokens = map[string]int{
	"<|endoftext|>": 50256,
}

// o200k_base pattern (for GPT-4o and later models)
const o200kPattern = `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

// o200k_base special tokens
var o200kSpecialTokens = map[string]int{
	"<|endoftext|>":   199999,
	"<|endofprompt|>": 200018,
}

// gemma pattern - match everything
const gemmaPattern = `.+`

// gemma special tokens
var gemmaSpecialTokens = map[string]int{
	"<pad>": 0,
	"<eos>": 1,
	"<bos>": 2,
	"<unk>": 3,
}

// gemmaNormalize replaces spaces with ▁ (SentencePiece style)
func gemmaNormalize(s string) string {
	return strings.ReplaceAll(s, " ", "▁")
}

// gemmaDenormalize replaces ▁ with spaces
func gemmaDenormalize(s string) string {
	return strings.ReplaceAll(s, "▁", " ")
}

// Encodings maps encoding names to their configurations
var Encodings = map[string]EncodingConfig{
	"cl100k_base": {
		Name:          "cl100k_base",
		Pattern:       cl100kPattern,
		SpecialTokens: cl100kSpecialTokens,
	},
	"p50k_base": {
		Name:          "p50k_base",
		Pattern:       p50kPattern,
		SpecialTokens: p50kSpecialTokens,
	},
	"o200k_base": {
		Name:          "o200k_base",
		Pattern:       o200kPattern,
		SpecialTokens: o200kSpecialTokens,
	},
	"gemma": {
		Name:          "gemma",
		Pattern:       gemmaPattern,
		SpecialTokens: gemmaSpecialTokens,
		IsBinary:      true,
		Normalize:     gemmaNormalize,
		Denormalize:   gemmaDenormalize,
	},
}

// DefaultEncoding is the encoding used when none is specified
const DefaultEncoding = "cl100k_base"
