package tokenizer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/wow-look-at-my/go-containers/set"
)

// countCase is one row of testdata/anthropic-counts.jsonl: a text and what the
// token counting API reported for it, per family. The counts are measurements,
// not expectations produced by this package, so a change that drifts away from
// Anthropic's tokenizer fails here.
type countCase struct {
	ID        string         `json:"id"`
	Category  string         `json:"category"`
	Text      string         `json:"text"`
	APICounts map[string]int `json:"apiCounts"`
}

// adversarialCategories hold shapes built to break the estimator rather than to
// represent real input. Their cost turns on which long single-character tokens
// each vocabulary happens to contain, which Anthropic does not publish, so they
// carry their own looser bound. docs/anthropic-estimator.md explains why.
var adversarialCategories = set.Of[string]("repetition", "whitespace")

func loadCountCases(t *testing.T) []countCase {
	t.Helper()

	f, err := os.Open("testdata/anthropic-counts.jsonl")
	require.NoError(t, err)
	defer f.Close()

	var cases []countCase
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1<<20), 1<<20)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var c countCase
		require.NoError(t, json.Unmarshal([]byte(line), &c))
		cases = append(cases, c)
	}
	require.NoError(t, scanner.Err())
	require.NotEmpty(t, cases)
	return cases
}

// errorPercentiles reports the mean and the given percentile of a sorted error
// sample, so a bound can be stated on the shape of the distribution rather than
// on a single worst case.
func errorPercentiles(errs []float64, p float64) (mean, pct float64) {
	sort.Float64s(errs)
	sum := 0.0
	for _, e := range errs {
		sum += e
	}
	return sum / float64(len(errs)), errs[int(float64(len(errs)-1)*p)]
}

// TestAnthropicEstimatorAccuracy holds the estimator to the accuracy it was
// measured at. The bounds sit a little above the fit so ordinary noise does not
// fail the build, and far enough below the uncorrected base encoding that a
// regression to it cannot pass.
func TestAnthropicEstimatorAccuracy(t *testing.T) {
	cases := loadCountCases(t)

	for _, family := range []AnthropicFamily{FamilyClaude45, FamilyClaude5} {
		t.Run(string(family), func(t *testing.T) {
			est, err := NewAnthropicEstimator(family)
			require.NoError(t, err)

			var ordinary, adversarial []float64
			for _, c := range cases {
				want, ok := c.APICounts[string(family)]
				if !ok {
					continue
				}
				got, err := est.CountTokens(c.Text)
				require.NoError(t, err, c.ID)

				// The fixture stores what the API reports for a whole message,
				// so the per-message overhead comes off before comparing.
				errPct := math.Abs(float64(got+est.MessageOverhead()-want)) / float64(want) * 100
				if adversarialCategories.Contains(c.Category) {
					adversarial = append(adversarial, errPct)
				} else {
					ordinary = append(ordinary, errPct)
				}
			}
			require.NotEmpty(t, ordinary)

			mean, p90 := errorPercentiles(ordinary, 0.9)
			t.Logf("representative text: n=%d mean %.2f%% p90 %.2f%%", len(ordinary), mean, p90)
			assert.Less(t, mean, 8.0, "mean error on representative text regressed")
			assert.Less(t, p90, 16.0, "90th percentile error on representative text regressed")

			if len(adversarial) > 0 {
				advMean, advP90 := errorPercentiles(adversarial, 0.9)
				t.Logf("adversarial runs: n=%d mean %.2f%% p90 %.2f%%", len(adversarial), advMean, advP90)
				assert.Less(t, advMean, 90.0, "error on single-character runs regressed")
			}
		})
	}
}

// TestAnthropicEstimatorBeatsBaseEncoding proves the correction is doing the
// work. Without it the base encoding is wrong by tens of percent, and a fit
// that silently degenerated to passing its output through would fail here.
func TestAnthropicEstimatorBeatsBaseEncoding(t *testing.T) {
	cases := loadCountCases(t)
	base, err := NewWithEncoding(anthropicBaseEncoding)
	require.NoError(t, err)

	for _, family := range []AnthropicFamily{FamilyClaude45, FamilyClaude5} {
		t.Run(string(family), func(t *testing.T) {
			est, err := NewAnthropicEstimator(family)
			require.NoError(t, err)

			var estErrs, baseErrs []float64
			for _, c := range cases {
				want, ok := c.APICounts[string(family)]
				if !ok || adversarialCategories.Contains(c.Category) {
					continue
				}
				got, err := est.CountTokens(c.Text)
				require.NoError(t, err)
				raw, err := base.CountTokens(c.Text)
				require.NoError(t, err)

				estErrs = append(estErrs, math.Abs(float64(got+est.MessageOverhead()-want))/float64(want)*100)
				baseErrs = append(baseErrs, math.Abs(float64(raw-want))/float64(want)*100)
			}

			estMean, _ := errorPercentiles(estErrs, 0.9)
			baseMean, _ := errorPercentiles(baseErrs, 0.9)
			t.Logf("estimator %.2f%% vs base encoding %.2f%%", estMean, baseMean)
			assert.Less(t, estMean*2, baseMean, "the correction should at least halve the base encoding's error")
		})
	}
}

// TestAnthropicFamiliesDisagreeOnASCII records the measurement that motivates
// separate flavors: the two tokenizers charge differently for ASCII letters.
// One flavor for both would be wrong for whichever it was not fitted to.
func TestAnthropicFamiliesDisagreeOnASCII(t *testing.T) {
	older, err := NewAnthropicEstimator(FamilyClaude45)
	require.NoError(t, err)
	newer, err := NewAnthropicEstimator(FamilyClaude5)
	require.NoError(t, err)

	const pangram = "The quick brown fox jumps over the lazy dog. "
	oldCount, err := older.CountTokens(strings.Repeat(pangram, 8))
	require.NoError(t, err)
	newCount, err := newer.CountTokens(strings.Repeat(pangram, 8))
	require.NoError(t, err)

	assert.Greater(t, newCount, oldCount, "the newer tokenizer was measured as the more granular one on ASCII text")
}

// TestAnthropicEstimatorRefusesToEncode guards the honesty of the API surface:
// Anthropic publishes no vocabulary for these models, so there are no token IDs
// to hand back, and inventing them would let a caller decode nonsense.
func TestAnthropicEstimatorRefusesToEncode(t *testing.T) {
	est, err := NewAnthropicEstimator(FamilyClaude5)
	require.NoError(t, err)

	_, err = est.Encode("hello")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "CountTokens")

	_, err = est.Decode([]int{1, 2, 3})
	require.Error(t, err)
}

// TestEncodingForModel maps model ids onto flavors, and reports rather than
// guesses when an id has not been measured.
func TestEncodingForModel(t *testing.T) {
	for model, want := range map[string]string{
		"claude-opus-5":              string(FamilyClaude5),
		"claude-sonnet-5":            string(FamilyClaude5),
		"claude-opus-4-8":            string(FamilyClaude5),
		"claude-sonnet-4-6":          string(FamilyClaude45),
		"claude-opus-4-5-20251101":   string(FamilyClaude45),
		"claude-haiku-4-5-20251001":  string(FamilyClaude45),
		"claude-sonnet-4-5-20250929": string(FamilyClaude45),
	} {
		got, ok := EncodingForModel(model)
		require.True(t, ok, model)
		assert.Equal(t, want, got, model)
	}

	_, ok := EncodingForModel("gpt-4o")
	assert.False(t, ok, "an unmeasured model must not be given a flavor by guesswork")
}

// TestAnthropicEstimatorViaEncoding checks the flavors are reachable by name,
// which is how the CLI and library callers select them.
func TestAnthropicEstimatorViaEncoding(t *testing.T) {
	for _, name := range []string{string(FamilyClaude45), string(FamilyClaude5)} {
		tok, err := NewWithEncoding(name)
		require.NoError(t, err, name)

		n, err := tok.CountTokens("The quick brown fox jumps over the lazy dog.")
		require.NoError(t, err)
		assert.Positive(t, n)
	}
}

// TestClassifyPreToken pins the bucketing the correction is fitted per, so a
// later edit cannot quietly reroute a script into another class and invalidate
// every weight.
func TestClassifyPreToken(t *testing.T) {
	for text, want := range map[string]anthropicClass{
		" the":     classWordLower,
		" The":     classWordCapital,
		" HTTP":    classWordUpper,
		" café":    classWordLatin,
		" Привет":  classWordCyrillic,
		"日本":       classWordCJK,
		" 한국어":     classWordHangul,
		" مرحبا":   classWordOtherScript,
		" 1234":    classNumber,
		"();":      classPunct,
		"    ":     classSpaceRun,
		"\n\n":     classNewlineRun,
		"aaaaaaaa": classRepeatRun,
		"--------": classRepeatRun,
	} {
		assert.Equal(t, want, classifyPreToken(text), fmt.Sprintf("%q", text))
	}
}
