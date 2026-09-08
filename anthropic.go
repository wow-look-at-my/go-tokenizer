package tokenizer

import (
	"fmt"
	"math"
	"sync"
	"unicode"
)

// Anthropic has never published the vocabulary its current models use, and the
// count_tokens endpoint is the only authority on it. This file holds an offline
// estimator for those counts: a BPE tokenizer this repository does embed runs
// over the text, and a per-character-class correction, fitted against measured
// counts, maps its output onto what Anthropic reports. Nothing here reaches the
// network.
//
// Measurement found two live tokenizers, not one. They agree exactly on digits,
// punctuation, and every non-Latin script, and disagree on runs of ASCII
// letters, where the newer one is markedly more granular. AnthropicFamily names
// that split; EncodingForModel maps a model id onto it.

// AnthropicFamily identifies a tokenizer shared by a group of Claude models.
type AnthropicFamily string

const (
	// FamilyClaude45 covers the Claude 4.5 and 4.6 generation.
	FamilyClaude45 AnthropicFamily = "claude_4_5"
	// FamilyClaude5 covers the Claude 5 generation, including Opus 4.8.
	FamilyClaude5 AnthropicFamily = "claude_5"
)

// anthropicClass buckets a pre-token by script and case. The correction is
// fitted per class because the error of the embedded BPE against Anthropic's
// tokenizer is script-dependent: it tracks Latin text far better than it tracks
// Han or Cyrillic, and the two families differ only on ASCII letters.
type anthropicClass int

const (
	classWordLower anthropicClass = iota
	classWordCapital
	classWordUpper
	classWordLatin
	classWordCyrillic
	classWordCJK
	classWordHangul
	classWordOtherScript
	classNumber
	classPunct
	classSymbol
	classSpaceRun
	classNewlineRun
	classRepeatRun
	numAnthropicClasses
)

// repeatRunThreshold is the length past which a pre-token of one repeated
// character is charged from its length alone.
//
// The base encoding's cost for such a run says almost nothing about
// Anthropic's, because each vocabulary holds its own long single-character
// tokens: measured at four hundred characters, a run of "a" costs 2.7x what the
// base encoding spends while a run of "x" costs 1.04x. Reading the base cost
// there is worse than ignoring it, so these runs are priced by length in their
// own class. Which specific runs are cheap stays unknowable from outside, and
// docs/anthropic-estimator.md records the error that leaves.
const repeatRunThreshold = 4

// anthropicFeatures is what the correction is a linear function of: for every
// class, how many tokens the embedded BPE spent, how many pre-tokens fell in
// it, and how many runes those held.
type anthropicFeatures struct {
	BaseTokens [numAnthropicClasses]float64
	PreTokens  [numAnthropicClasses]float64
	Runes      [numAnthropicClasses]float64
	LongRunes  [numAnthropicClasses]float64
	HugeRunes  [numAnthropicClasses]float64
}

// A pre-token's length is split at these breakpoints so each span gets its own
// rate. A long run of one character is where the embedded BPE and Anthropic's
// tokenizer diverge hardest: the embedded one holds tokens covering such a run
// and spends few, Anthropic's spends many. One rate per rune cannot fit a short
// word, an indent, and a four-hundred-character run at once.
const (
	longRuneThreshold = 8
	hugeRuneThreshold = 64
)

// anthropicWeights is the fitted correction for one family.
type anthropicWeights struct {
	Base     [numAnthropicClasses]float64
	Pre      [numAnthropicClasses]float64
	Rune     [numAnthropicClasses]float64
	LongRune [numAnthropicClasses]float64
	HugeRune [numAnthropicClasses]float64
	Offset   float64
}

// predict applies the correction to extracted features.
func (w *anthropicWeights) predict(f *anthropicFeatures) float64 {
	total := w.Offset
	for c := anthropicClass(0); c < numAnthropicClasses; c++ {
		total += w.Base[c]*f.BaseTokens[c] + w.Pre[c]*f.PreTokens[c] +
			w.Rune[c]*f.Runes[c] + w.LongRune[c]*f.LongRunes[c] + w.HugeRune[c]*f.HugeRunes[c]
	}
	return total
}

// classifyPreToken buckets a pre-token by its letters, falling back to its
// symbols when it holds none. A pre-token produced by the cl100k pattern is
// homogeneous enough that its first letter decides the script.
// isRepeatRun reports whether s is one non-whitespace character repeated at
// least repeatRunThreshold times.
func isRepeatRun(s string) bool {
	var first rune = -1
	n := 0
	for _, r := range s {
		if unicode.IsSpace(r) {
			return false
		}
		if first < 0 {
			first = r
		} else if r != first {
			return false
		}
		n++
	}
	return n >= repeatRunThreshold
}

func classifyPreToken(s string) anthropicClass {
	if isRepeatRun(s) {
		return classRepeatRun
	}
	var (
		letters, upper, lower, digits, space, newline, nonASCII int
		firstLetter                                             rune = -1
	)
	for _, r := range s {
		switch {
		case unicode.IsLetter(r):
			letters++
			if firstLetter < 0 {
				firstLetter = r
			}
			if unicode.IsUpper(r) {
				upper++
			} else if unicode.IsLower(r) {
				lower++
			}
		case unicode.IsDigit(r):
			digits++
		case r == '\n' || r == '\r':
			newline++
		case unicode.IsSpace(r):
			space++
		}
		if r > unicode.MaxASCII {
			nonASCII++
		}
	}

	if letters == 0 {
		switch {
		case digits > 0:
			return classNumber
		// A run holding a line break is charged differently from a run of
		// spaces or tabs, so the two do not share a rate.
		case newline > 0:
			return classNewlineRun
		case space > 0:
			return classSpaceRun
		case nonASCII > 0:
			return classSymbol
		default:
			return classPunct
		}
	}

	switch {
	case unicode.Is(unicode.Han, firstLetter),
		unicode.Is(unicode.Hiragana, firstLetter),
		unicode.Is(unicode.Katakana, firstLetter):
		return classWordCJK
	case unicode.Is(unicode.Hangul, firstLetter):
		return classWordHangul
	case unicode.Is(unicode.Cyrillic, firstLetter), unicode.Is(unicode.Greek, firstLetter):
		return classWordCyrillic
	case !unicode.Is(unicode.Latin, firstLetter):
		return classWordOtherScript
	case nonASCII > 0:
		return classWordLatin
	case upper >= 2 && lower == 0:
		return classWordUpper
	case upper > 0 && unicode.IsUpper(firstLetter):
		return classWordCapital
	default:
		return classWordLower
	}
}

// AnthropicEstimator reports how many tokens Anthropic's tokenizer would spend
// on a text. It answers offline, from an embedded vocabulary and a fitted
// correction, and never contacts the API.
//
// It estimates a count and does not reproduce a segmentation: Anthropic
// publishes no vocabulary for these models, so there are no token IDs to
// return. Encode and Decode say so rather than inventing IDs.
type AnthropicEstimator struct {
	weights anthropicWeights
	family  AnthropicFamily
}

// The base tokenizer is shared: every family corrects the output of the same
// embedded encoding, and loading that vocabulary is the expensive part.
var (
	anthropicBaseOnce sync.Once
	anthropicBase     *tokenizer
	anthropicBaseErr  error
)

func loadAnthropicBase() (*tokenizer, error) {
	anthropicBaseOnce.Do(func() {
		tok, err := NewWithEncoding(anthropicBaseEncoding)
		if err != nil {
			anthropicBaseErr = fmt.Errorf("loading base encoding %s: %w", anthropicBaseEncoding, err)
			return
		}
		anthropicBase = tok.(*tokenizer)
	})
	return anthropicBase, anthropicBaseErr
}

// NewAnthropicEstimator builds an estimator for one tokenizer family.
func NewAnthropicEstimator(family AnthropicFamily) (*AnthropicEstimator, error) {
	weights, ok := anthropicWeightsByFamily[family]
	if !ok {
		return nil, fmt.Errorf("no fitted weights for Anthropic family %q", family)
	}
	if _, err := loadAnthropicBase(); err != nil {
		return nil, err
	}
	return &AnthropicEstimator{weights: weights, family: family}, nil
}

// extractAnthropicFeatures runs the base tokenizer and buckets its output.
func extractAnthropicFeatures(text string) (*anthropicFeatures, error) {
	base, err := loadAnthropicBase()
	if err != nil {
		return nil, err
	}
	preTokens, err := base.preTokenizer.Tokenize(text)
	if err != nil {
		return nil, fmt.Errorf("pre-tokenization failed: %w", err)
	}

	f := &anthropicFeatures{}
	for _, pt := range preTokens {
		class := classifyPreToken(pt.Text)
		runes := len([]rune(pt.Text))
		f.BaseTokens[class] += float64(base.bpe.CountTokens([]byte(pt.Text)))
		f.PreTokens[class]++
		f.Runes[class] += float64(runes)
		if runes > longRuneThreshold {
			f.LongRunes[class] += float64(runes - longRuneThreshold)
		}
		if runes > hugeRuneThreshold {
			f.HugeRunes[class] += float64(runes - hugeRuneThreshold)
		}
	}
	return f, nil
}

// AnthropicFeatureVector exposes the features the fitted correction is a linear
// function of, flattened in the order the weights are stored and ending in the
// constant term. A calibration tool fits weights against measured counts with
// it; the shipped estimator needs no such call.
func AnthropicFeatureVector(text string) ([]float64, error) {
	f, err := extractAnthropicFeatures(text)
	if err != nil {
		return nil, err
	}
	v := make([]float64, 0, 5*numAnthropicClasses+1)
	v = append(v, f.BaseTokens[:]...)
	v = append(v, f.PreTokens[:]...)
	v = append(v, f.Runes[:]...)
	v = append(v, f.LongRunes[:]...)
	v = append(v, f.HugeRunes[:]...)
	return append(v, 1), nil
}

// CountTokens estimates the tokens Anthropic's tokenizer spends on text.
func (e *AnthropicEstimator) CountTokens(text string) (int, error) {
	if text == "" {
		return 0, nil
	}
	f, err := extractAnthropicFeatures(text)
	if err != nil {
		return 0, err
	}
	n := int(math.Round(e.weights.predict(f)))
	if n < 1 {
		// Any non-empty text costs at least a token; a correction that
		// undershoots into zero or below would be reported as free.
		n = 1
	}
	return n, nil
}

// Encode reports that no segmentation is available. Returning invented IDs
// would let a caller decode them back into text that was never tokenized.
func (e *AnthropicEstimator) Encode(string) ([]int, error) {
	return nil, fmt.Errorf("encoding %s: Anthropic publishes no vocabulary for these models, so token IDs cannot be produced; use CountTokens", e.family)
}

// Decode reports the same absence as Encode.
func (e *AnthropicEstimator) Decode([]int) (string, error) {
	return "", fmt.Errorf("encoding %s: Anthropic publishes no vocabulary for these models, so token IDs cannot be decoded", e.family)
}

// VocabSize reports the size of the vocabulary actually consulted, which is the
// base encoding's rather than Anthropic's unpublished one.
func (e *AnthropicEstimator) VocabSize() int {
	base, err := loadAnthropicBase()
	if err != nil {
		return 0
	}
	return base.VocabSize()
}

// MessageOverhead is the token cost the Messages API adds around a single
// user text message, measured per family. Add it to a CountTokens result to
// predict what count_tokens reports for that message.
func (e *AnthropicEstimator) MessageOverhead() int { return anthropicOverheadByFamily[e.family] }

// EncodingForModel maps a Claude model id onto the encoding name that estimates
// its token counts. It reports false for an id whose family has not been
// measured, so a caller cannot silently budget with the wrong tokenizer.
func EncodingForModel(model string) (string, bool) {
	family, ok := anthropicFamilyByModel[model]
	if !ok {
		return "", false
	}
	return string(family), true
}
