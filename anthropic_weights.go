package tokenizer

// Fitted data for the Anthropic estimator. Regenerate it with
// `cmd/anthropic-calibrate`, which measures counts against the token counting
// API and refits; see docs/anthropic-estimator.md for the procedure and the
// error the current fit achieves.

// anthropicBaseEncoding is the embedded encoding whose output the correction
// adjusts. It is not Anthropic's tokenizer and is not claimed to be: it is a
// well-behaved BPE over the same text, and the correction carries the rest.
const anthropicBaseEncoding = "cl100k_base"

// anthropicFamilyByModel maps a measured model id onto its tokenizer family.
var anthropicFamilyByModel = map[string]AnthropicFamily{
	"claude-opus-4-5":            FamilyClaude45,
	"claude-opus-4-5-20251101":   FamilyClaude45,
	"claude-sonnet-4-5":          FamilyClaude45,
	"claude-sonnet-4-5-20250929": FamilyClaude45,
	"claude-haiku-4-5":           FamilyClaude45,
	"claude-haiku-4-5-20251001":  FamilyClaude45,
	"claude-sonnet-4-6":          FamilyClaude45,
	"claude-opus-4-6":            FamilyClaude45,
	"claude-opus-4-7":            FamilyClaude5,
	"claude-opus-4-8":            FamilyClaude5,
	"claude-opus-5":              FamilyClaude5,
	"claude-sonnet-5":            FamilyClaude5,
	"claude-fable-5":             FamilyClaude5,
	"claude-fable-5-1":           FamilyClaude5,
}

// anthropicOverheadByFamily is the measured token cost the Messages API adds
// around a single user text message.
var anthropicOverheadByFamily = map[AnthropicFamily]int{
	FamilyClaude45: 7,
	FamilyClaude5:  6,
}

// anthropicWeightsByFamily holds the fitted correction per family.
//
// The two families were fitted together against one design matrix. Every class
// outside runs of ASCII letters shares a single set of weights, because measured
// counts are identical there; only the ASCII-letter classes carry a per-family
// difference. That sharing is why the family with far fewer measurements still
// prices Han, Cyrillic, digits, and punctuation as accurately as the other.
var anthropicWeightsByFamily = map[AnthropicFamily]anthropicWeights{
	FamilyClaude45: {
		Base:     [numAnthropicClasses]float64{2.119086, 1.076662, 2.239959, -0.576697, -0.143732, 0.043224, 0.003309, 0.320614, 0.437105, 1.537911, -0.160627, 1.46204, 0.478689, 3.09254},
		Pre:      [numAnthropicClasses]float64{-1.097382, 0.09643, -1.641211, -1.525614, -0.077403, 0.03691, 0.769068, 0.087242, 0.437105, -1.898061, -0.048394, -0.147399, -0.041911, -0.732336},
		Rune:     [numAnthropicClasses]float64{0.017621, 0.00734, -0.015325, 0.379441, 0.380207, 0.671611, 0.906884, 0.4602, 0.232215, 0.817591, 1.052072, -0.126708, 0.840456, -1.308942},
		LongRune: [numAnthropicClasses]float64{-0.290038, 0.134692, 2.180673, 0.051025, 0.667671, 0.376333, 0.0, 0.013086, 0.0, -0.398056, 1.363577, -0.370896, -0.365872, 1.638421},
		HugeRune: [numAnthropicClasses]float64{-0.58286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.985556, 0.66412, -0.643729, -0.351837},
		Offset:   -7.86552,
	},
	FamilyClaude5: {
		Base:     [numAnthropicClasses]float64{2.049345, 0.492822, 0.637009, -0.576697, -0.143732, 0.043224, 0.003309, 0.320614, 0.437105, 1.537911, -0.160627, 1.46204, 0.478689, 3.09254},
		Pre:      [numAnthropicClasses]float64{-1.102396, -2.008023, -2.991464, -1.525614, -0.077403, 0.03691, 0.769068, 0.087242, 0.437105, -1.898061, -0.048394, -0.147399, -0.041911, -0.732336},
		Rune:     [numAnthropicClasses]float64{0.117945, 0.674581, 1.154371, 0.379441, 0.380207, 0.671611, 0.906884, 0.4602, 0.232215, 0.817591, 1.052072, -0.126708, 0.840456, -1.308942},
		LongRune: [numAnthropicClasses]float64{0.060843, -0.310841, -0.237875, 0.051025, 0.667671, 0.376333, 0.0, 0.013086, 0.0, -0.398056, 1.363577, -0.370896, -0.365872, 1.638421},
		HugeRune: [numAnthropicClasses]float64{-0.58286, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.985556, 0.66412, -0.643729, -0.351837},
		Offset:   -2.430681,
	},
}
