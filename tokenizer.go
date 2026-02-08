package tokenizer

import (
	"bytes"
	"fmt"
	"os"
	"sync"

	lru "github.com/hashicorp/golang-lru/v2"

	"github.com/wow-look-at-my/go-tokenizer/embed"
)

// Tokenizer is the main interface for BPE tokenization
type Tokenizer interface {
	// Encode returns token IDs for the input text
	Encode(text string) ([]int, error)

	// CountTokens returns just the token count (optimized path)
	CountTokens(text string) (int, error)

	// Decode converts token IDs back to text
	Decode(tokens []int) (string, error)

	// VocabSize returns the vocabulary size
	VocabSize() int
}

// tokenizer implements the Tokenizer interface
type tokenizer struct {
	vocab        *Vocabulary
	preTokenizer *PreTokenizer
	bpe          *BPE
	cache        *lru.Cache[string, []int]
	cacheMu      sync.RWMutex
	normalize    func(string) string
	denormalize  func(string) string
}

// Option configures tokenizer behavior
type Option func(*tokenizerOptions)

type tokenizerOptions struct {
	pattern       string
	specialTokens map[string]int
	cacheSize     int
}

// WithPattern sets a custom regex pattern for pre-tokenization
func WithPattern(pattern string) Option {
	return func(o *tokenizerOptions) {
		o.pattern = pattern
	}
}

// WithSpecialTokens sets custom special tokens
func WithSpecialTokens(tokens map[string]int) Option {
	return func(o *tokenizerOptions) {
		o.specialTokens = tokens
	}
}

// WithCacheSize sets the LRU cache size (default 10000)
func WithCacheSize(size int) Option {
	return func(o *tokenizerOptions) {
		o.cacheSize = size
	}
}

// New creates a tokenizer with the default cl100k_base encoding
func New() (Tokenizer, error) {
	return NewWithEncoding(DefaultEncoding)
}

// NewWithEncoding creates a tokenizer with a named encoding
func NewWithEncoding(name string) (Tokenizer, error) {
	config, ok := Encodings[name]
	if !ok {
		return nil, fmt.Errorf("unknown encoding: %s", name)
	}

	// Load embedded vocabulary
	var vocabData []byte
	switch name {
	case "cl100k_base":
		vocabData = embed.Cl100kBase
	case "gemma":
		vocabData = embed.Gemma
	default:
		return nil, fmt.Errorf("no embedded vocabulary for encoding: %s", name)
	}

	var vocab *Vocabulary
	var err error
	if config.IsBinary {
		vocab, err = LoadBinary(vocabData)
	} else {
		vocab, err = LoadTiktoken(bytes.NewReader(vocabData))
	}
	if err != nil {
		return nil, fmt.Errorf("loading vocabulary: %w", err)
	}

	// Add special tokens
	vocab.AddSpecialTokens(config.SpecialTokens)

	return newTokenizer(vocab, config.Pattern, config.SpecialTokens, config.Normalize, config.Denormalize, 10000)
}

// NewFromFile creates a tokenizer from a vocabulary file
func NewFromFile(path string, opts ...Option) (Tokenizer, error) {
	// Apply options
	options := &tokenizerOptions{
		pattern:       cl100kPattern, // Default pattern
		specialTokens: nil,
		cacheSize:     10000,
	}
	for _, opt := range opts {
		opt(options)
	}

	// Load vocabulary from file
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening vocabulary file: %w", err)
	}
	defer f.Close()

	vocab, err := LoadTiktoken(f)
	if err != nil {
		return nil, fmt.Errorf("loading vocabulary: %w", err)
	}

	// Add special tokens if provided
	if options.specialTokens != nil {
		vocab.AddSpecialTokens(options.specialTokens)
	}

	return newTokenizer(vocab, options.pattern, options.specialTokens, nil, nil, options.cacheSize)
}

func newTokenizer(vocab *Vocabulary, pattern string, specialTokens map[string]int, normalize, denormalize func(string) string, cacheSize int) (*tokenizer, error) {
	preTokenizer, err := NewPreTokenizer(pattern, specialTokens)
	if err != nil {
		return nil, fmt.Errorf("creating pre-tokenizer: %w", err)
	}

	cache, err := lru.New[string, []int](cacheSize)
	if err != nil {
		return nil, fmt.Errorf("creating cache: %w", err)
	}

	return &tokenizer{
		vocab:        vocab,
		preTokenizer: preTokenizer,
		bpe:          NewBPE(vocab),
		cache:        cache,
		normalize:    normalize,
		denormalize:  denormalize,
	}, nil
}

// Encode tokenizes text and returns token IDs
func (t *tokenizer) Encode(text string) ([]int, error) {
	if text == "" {
		return nil, nil
	}

	// Apply normalization
	if t.normalize != nil {
		text = t.normalize(text)
	}

	// Pre-tokenize
	preTokens, err := t.preTokenizer.Tokenize(text)
	if err != nil {
		return nil, fmt.Errorf("pre-tokenization failed: %w", err)
	}

	var result []int
	for _, pt := range preTokens {
		if pt.IsSpecial {
			// Special tokens are looked up directly
			if rank, ok := t.vocab.Encode([]byte(pt.Text)); ok {
				result = append(result, rank)
			}
			continue
		}

		// Check cache
		t.cacheMu.RLock()
		if cached, ok := t.cache.Get(pt.Text); ok {
			t.cacheMu.RUnlock()
			result = append(result, cached...)
			continue
		}
		t.cacheMu.RUnlock()

		// BPE encode
		tokens := t.bpe.Encode([]byte(pt.Text))

		// Cache result
		t.cacheMu.Lock()
		t.cache.Add(pt.Text, tokens)
		t.cacheMu.Unlock()

		result = append(result, tokens...)
	}

	return result, nil
}

// CountTokens returns the number of tokens without full encoding
func (t *tokenizer) CountTokens(text string) (int, error) {
	if text == "" {
		return 0, nil
	}

	// Apply normalization
	if t.normalize != nil {
		text = t.normalize(text)
	}

	// Pre-tokenize
	preTokens, err := t.preTokenizer.Tokenize(text)
	if err != nil {
		return 0, fmt.Errorf("pre-tokenization failed: %w", err)
	}

	count := 0
	for _, pt := range preTokens {
		if pt.IsSpecial {
			count++
			continue
		}

		// Check cache first
		t.cacheMu.RLock()
		if cached, ok := t.cache.Get(pt.Text); ok {
			t.cacheMu.RUnlock()
			count += len(cached)
			continue
		}
		t.cacheMu.RUnlock()

		// Use optimized count path
		count += t.bpe.CountTokens([]byte(pt.Text))
	}

	return count, nil
}

// Decode converts token IDs back to text
func (t *tokenizer) Decode(tokens []int) (string, error) {
	text := string(t.bpe.Decode(tokens))
	if t.denormalize != nil {
		text = t.denormalize(text)
	}
	return text, nil
}

// VocabSize returns the vocabulary size
func (t *tokenizer) VocabSize() int {
	return t.vocab.Size()
}
