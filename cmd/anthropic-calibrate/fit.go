package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/wow-look-at-my/go-tokenizer/tokenizer"
)

// families fixes the emission order, so a rerun gives a stable diff.
var families = []tokenizer.AnthropicFamily{tokenizer.FamilyClaude45, tokenizer.FamilyClaude5}

// aggregateNames label the blocks of the feature vector, in its order.
var aggregateNames = []string{"Base", "Pre", "Rune", "LongRune", "HugeRune"}

// adversarial marks shapes built to break the estimator rather than to stand
// for real input. Their cost turns on which long single-character tokens a
// vocabulary holds, which Anthropic does not publish, so they are reported
// apart from the accuracy ordinary text gets.
func adversarial(category string) bool {
	return category == "repetition" || category == "whitespace"
}

// sample is a row of the design matrix.
type sample struct {
	family   tokenizer.AnthropicFamily
	category string
	text     string
	features []float64
	target   float64 // tokens of the text itself, message overhead removed
}

// buildSamples turns labelled rows into design rows, per family a row was
// measured for. The overhead the API adds around a message is removed here, so
// the fit predicts the cost of the text and nothing else.
func buildSamples(rows []labelledRow) ([]sample, error) {
	var out []sample
	for _, r := range rows {
		features, err := tokenizer.AnthropicFeatureVector(r.Text)
		if err != nil {
			return nil, fmt.Errorf("sample %s: %w", r.ID, err)
		}
		for _, family := range families {
			count, ok := r.APICounts[string(family)]
			if !ok {
				continue
			}
			est, err := tokenizer.NewAnthropicEstimator(family)
			if err != nil {
				return nil, err
			}
			out = append(out, sample{
				family:   family,
				category: r.Category,
				text:     r.Text,
				features: features,
				target:   float64(count - est.MessageOverhead()),
			})
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no row carries a count for a known family")
	}
	return out, nil
}

// jointWidth is the column count of the joint design matrix: a shared block,
// then a block covering only the classes the families disagree on, then a
// constant that applies to the newer family alone.
func jointWidth() int {
	shared := tokenizer.AnthropicFeatureAggregates*tokenizer.AnthropicFeatureClasses + 1
	return shared + tokenizer.AnthropicFeatureAggregates*len(tokenizer.AnthropicASCIILetterClasses()) + 1
}

// jointRow expands a sample's features into a joint design row.
func jointRow(features []float64, family tokenizer.AnthropicFamily) []float64 {
	shared := tokenizer.AnthropicFeatureAggregates*tokenizer.AnthropicFeatureClasses + 1
	row := make([]float64, jointWidth())
	copy(row, features)
	if family == tokenizer.FamilyClaude45 {
		return row
	}
	at := shared
	for a := 0; a < tokenizer.AnthropicFeatureAggregates; a++ {
		for _, c := range tokenizer.AnthropicASCIILetterClasses() {
			row[at] = features[a*tokenizer.AnthropicFeatureClasses+c]
			at++
		}
	}
	row[at] = 1
	return row
}

// splitJoint recovers a weight vector per family from the joint solution.
func splitJoint(w []float64) map[tokenizer.AnthropicFamily][]float64 {
	shared := tokenizer.AnthropicFeatureAggregates*tokenizer.AnthropicFeatureClasses + 1

	older := make([]float64, shared)
	copy(older, w[:shared])
	newer := make([]float64, shared)
	copy(newer, w[:shared])

	at := shared
	for a := 0; a < tokenizer.AnthropicFeatureAggregates; a++ {
		for _, c := range tokenizer.AnthropicASCIILetterClasses() {
			newer[a*tokenizer.AnthropicFeatureClasses+c] += w[at]
			at++
		}
	}
	newer[shared-1] += w[at]
	return map[tokenizer.AnthropicFamily][]float64{
		tokenizer.FamilyClaude45: older,
		tokenizer.FamilyClaude5:  newer,
	}
}

// ridge solves (XtX + lambda I) w = Xty by Gauss-Jordan elimination. The
// penalty stops a class seen a handful of times from taking an extreme weight.
func ridge(x [][]float64, y []float64, lambda float64) []float64 {
	p := len(x[0])
	a := make([][]float64, p)
	for i := range a {
		a[i] = make([]float64, p+1)
	}
	for i := range x {
		for j := 0; j < p; j++ {
			if x[i][j] == 0 {
				continue
			}
			for k := 0; k < p; k++ {
				a[j][k] += x[i][j] * x[i][k]
			}
			a[j][p] += x[i][j] * y[i]
		}
	}
	for j := 0; j < p; j++ {
		a[j][j] += lambda
	}

	for col := 0; col < p; col++ {
		pivot := col
		for r := col + 1; r < p; r++ {
			if math.Abs(a[r][col]) > math.Abs(a[pivot][col]) {
				pivot = r
			}
		}
		a[col], a[pivot] = a[pivot], a[col]
		if math.Abs(a[col][col]) < 1e-12 {
			continue
		}
		for r := 0; r < p; r++ {
			if r == col {
				continue
			}
			f := a[r][col] / a[col][col]
			for k := col; k <= p; k++ {
				a[r][k] -= f * a[col][k]
			}
		}
	}

	w := make([]float64, p)
	for j := 0; j < p; j++ {
		if math.Abs(a[j][j]) > 1e-12 {
			w[j] = a[j][p] / a[j][j]
		}
	}
	return w
}

func predict(w, features []float64) float64 {
	sum := 0.0
	for i := range w {
		sum += w[i] * features[i]
	}
	return math.Round(sum)
}

// fitJoint solves for both families together over the samples skip keeps.
func fitJoint(samples []sample, lambda float64, skip func(int) bool) map[tokenizer.AnthropicFamily][]float64 {
	var rows [][]float64
	var targets []float64
	for i, s := range samples {
		if skip != nil && skip(i) {
			continue
		}
		rows = append(rows, jointRow(s.features, s.family))
		targets = append(targets, s.target)
	}
	if len(rows) == 0 {
		return nil
	}
	return splitJoint(ridge(rows, targets, lambda))
}

// crossValidate reports held-out mean error for a given penalty.
func crossValidate(samples []sample, lambda float64) float64 {
	const folds = 5
	var errs []float64
	for fold := 0; fold < folds; fold++ {
		weights := fitJoint(samples, lambda, func(i int) bool { return i%folds == fold })
		if weights == nil {
			continue
		}
		for i, s := range samples {
			if i%folds != fold {
				continue
			}
			errs = append(errs, math.Abs(predict(weights[s.family], s.features)-s.target)/s.target*100)
		}
	}
	return mean(errs)
}

func mean(v []float64) float64 {
	if len(v) == 0 {
		return math.NaN()
	}
	sum := 0.0
	for _, x := range v {
		sum += x
	}
	return sum / float64(len(v))
}

func runFit(args []string) error {
	fs := flag.NewFlagSet("fit", flag.ExitOnError)
	counts := fs.String("counts", "", "labelled rows from measure")
	lambda := fs.Float64("lambda", 0, "ridge penalty; cross-validated when unset")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *counts == "" {
		return fmt.Errorf("fit needs -counts")
	}

	rows, err := readRows(*counts)
	if err != nil {
		return err
	}
	samples, err := buildSamples(rows)
	if err != nil {
		return err
	}

	chosen := *lambda
	if chosen == 0 {
		best := math.Inf(1)
		for _, candidate := range []float64{0.03, 0.1, 0.3, 1, 3, 10, 30, 100} {
			score := crossValidate(samples, candidate)
			fmt.Fprintf(os.Stderr, "  lambda %-6g cross-validated mean %6.2f%%\n", candidate, score)
			if score < best {
				best, chosen = score, candidate
			}
		}
		fmt.Fprintf(os.Stderr, "chosen lambda %g\n", chosen)
	}

	weights := fitJoint(samples, chosen, nil)
	scoreSamples(os.Stderr, samples, func(s sample) float64 {
		return predict(weights[s.family], s.features)
	})
	emitWeights(os.Stdout, weights)
	return nil
}

// emitWeights prints the weight map of anthropic_weights.go, ready to paste.
func emitWeights(out *os.File, weights map[tokenizer.AnthropicFamily][]float64) {
	classes := tokenizer.AnthropicFeatureClasses
	fmt.Fprintln(out, "var anthropicWeightsByFamily = map[AnthropicFamily]anthropicWeights{")
	for _, family := range families {
		w := weights[family]
		name := "FamilyClaude45"
		if family == tokenizer.FamilyClaude5 {
			name = "FamilyClaude5"
		}
		fmt.Fprintf(out, "\t%s: {\n", name)
		for a, label := range aggregateNames {
			fmt.Fprintf(out, "\t\t%s: [numAnthropicClasses]float64{", label)
			for c := 0; c < classes; c++ {
				if c > 0 {
					fmt.Fprint(out, ", ")
				}
				fmt.Fprintf(out, "%g", w[a*classes+c])
			}
			fmt.Fprintln(out, "},")
		}
		fmt.Fprintf(out, "\t\tOffset: %g,\n\t},\n", w[len(aggregateNames)*classes])
	}
	fmt.Fprintln(out, "}")
}

// scoreSamples prints error for representative text and adversarial runs apart,
// per family, which is how the accuracy claim is stated and tested.
func scoreSamples(out *os.File, samples []sample, predictFn func(sample) float64) {
	for _, family := range families {
		var ordinary, runs []float64
		for _, s := range samples {
			if s.family != family {
				continue
			}
			e := math.Abs(predictFn(s)-s.target) / s.target * 100
			if adversarial(s.category) {
				runs = append(runs, e)
			} else {
				ordinary = append(ordinary, e)
			}
		}
		fmt.Fprintf(out, "\n%s\n", family)
		report(out, "representative text", ordinary)
		report(out, "single-character runs", runs)
	}
}

func report(out *os.File, label string, errs []float64) {
	if len(errs) == 0 {
		return
	}
	sort.Float64s(errs)
	fmt.Fprintf(out, "  %-22s n=%3d  mean %6.2f%%  median %6.2f%%  p90 %6.2f%%  max %7.2f%%\n",
		label, len(errs), mean(errs), errs[len(errs)/2], errs[int(float64(len(errs)-1)*0.9)], errs[len(errs)-1])
}

func runReport(args []string) error {
	fs := flag.NewFlagSet("report", flag.ExitOnError)
	counts := fs.String("counts", "", "labelled rows from measure")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *counts == "" {
		return fmt.Errorf("report needs -counts")
	}

	rows, err := readRows(*counts)
	if err != nil {
		return err
	}
	samples, err := buildSamples(rows)
	if err != nil {
		return err
	}

	estimators := map[tokenizer.AnthropicFamily]*tokenizer.AnthropicEstimator{}
	for _, family := range families {
		est, err := tokenizer.NewAnthropicEstimator(family)
		if err != nil {
			return err
		}
		estimators[family] = est
	}

	// Score through the estimator's own entry point, so this reports what a
	// caller actually gets rather than what a fit believed it would give.
	var failed error
	scoreSamples(os.Stdout, samples, func(s sample) float64 {
		n, err := estimators[s.family].CountTokens(s.text)
		if err != nil && failed == nil {
			failed = err
		}
		return float64(n)
	})
	return failed
}
