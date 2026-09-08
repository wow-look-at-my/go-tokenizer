# Anthropic token-count estimators

Anthropic has never published the vocabulary its current models use. The `count_tokens` endpoint is the only authority on what they charge. This package ships encodings that answer the same question offline.

| encoding | models |
| --- | --- |
| `claude_4_5` | Claude 4.5 and 4.6 (Opus 4.5, Sonnet 4.5 and 4.6, Haiku 4.5) |
| `claude_5` | Claude 5, Opus 4.7 and 4.8, Fable 5 |

`EncodingForModel` maps a model id onto the right encoding. It reports `false` for an id whose family nobody measured, rather than guessing.

```go
name, ok := tokenizer.EncodingForModel("claude-opus-5") // "claude_5", true
est, _ := tokenizer.NewAnthropicEstimator(tokenizer.FamilyClaude5)
n, _ := est.CountTokens(text)                            // tokens in the text
total := n + est.MessageOverhead()                       // what count_tokens reports
```

Nothing here contacts the network. The API produced the fitted constants, through the maintenance command described below.

## The families use different tokenizers

Measurement across the live models found a split that no public writeup records. The families agree exactly on digits, punctuation, and every non-Latin script. They differ only on runs of ASCII letters. The newer family is far more granular there.

| text | claude_4_5 | claude_5 |
| --- | --- | --- |
| `The quick brown fox jumps over the lazy dog. ` | 11 | 18 |
| `The Quick Brown Fox` | 4 | 10 |
| ` Здравствуй` | 6 | 6 |
| ` 日本語` | 3 | 3 |
| `();` | 1 | 1 |

A single flavor is therefore wrong by roughly 60% on English, for whichever family nobody fitted it to. `TestAnthropicFamiliesDisagreeOnASCII` pins the direction of the split.

The message overhead also differs per family. `MessageOverhead` is exposed separately from `CountTokens` because of it. The cost of a text and the cost of a message holding that text are different questions.

## How the estimate works

The embedded `cl100k_base` BPE runs over the text. A correction fitted per character class then maps its output onto the counts the API reports.

Each pre-token goes into a class by script and case. A word splits by case into lower-case, capitalized and upper-case. It splits by script into ASCII, non-ASCII Latin, Cyrillic or Greek, Han or Kana, Hangul, and anything else. The remaining classes are number, punctuation, non-ASCII symbol, space run, line-break run, and single-character run.

Within a class the estimate is linear in four aggregates. Those are the tokens the base encoding spent, the pre-token count, the rune count, and the rune count past each of two length breakpoints. The breakpoints exist because a single rate per rune cannot fit a short word, an indent, and a four-hundred-character run at once.

Both families are fitted together, against one design matrix. Every weight outside the ASCII-letter classes is shared, because the measurements say the families charge identically there. Only the ASCII-letter classes carry a per-family difference.

That sharing is what lets the family with far fewer measurements price Han, Cyrillic, digits and punctuation as accurately as the other one.

Five-fold cross-validation chooses the penalty on the fit. It scores on held-out rows that the API measured.

## Accuracy

The numbers below come from API-measured counts. The text stands for real input. It holds prose in several languages and scripts, source in several languages, structured data, identifiers, emoji, and recorded assistant output.

| | mean | median | p90 |
| --- | --- | --- | --- |
| `claude_4_5` | 4.2% | 3.0% | 11.0% |
| `claude_5` | 4.2% | 3.4% | 9.5% |
| uncorrected `cl100k_base`, against `claude_4_5` | 19.7% | 14.3% | 34.3% |
| uncorrected `cl100k_base`, against `claude_5` | 35.6% | 34.1% | 52.9% |

`TestAnthropicEstimatorAccuracy` holds these bounds in CI, against `testdata/anthropic-counts.jsonl`. `TestAnthropicEstimatorBeatsBaseEncoding` fails if the correction ever degenerates into passing the base encoding through.

## What it gets wrong

A run of a single repeated character stays inaccurate. The tests bound those runs separately, rather than hiding them in the average.

The cost of such a run turns on which long single-character tokens a vocabulary holds. That is exactly what stays unpublished. The table below measures runs of four hundred characters, against what the base encoding spends.

| input | ratio to base encoding |
| --- | --- |
| `a` repeated | 2.70 |
| `x` repeated | 1.04 |
| `ab` repeated | 0.50 |
| a space repeated | 4.50 |
| a line break repeated | 0.57 |

No function of script and length separates those cases. So a run gets its own class, and its cost comes from its length alone. That bounds the damage without fixing it. The worst case fell from about 750% to about 400% when the class was added.

A context window that is mostly padding, ASCII art, or a hex dump will therefore be estimated badly. Ordinary prose, code and markup will not.

`Encode` and `Decode` return an error. Anthropic publishes no vocabulary for these models, so no token IDs exist to return. Inventing IDs lets a caller decode them back into text that nothing ever tokenized.

## Regenerating the constants

`cmd/anthropic-calibrate` is a maintenance command, run by hand. The library never calls it. The build does not depend on it.

```sh
# Measure a corpus. Needs ANTHROPIC_API_KEY. Rate limited, and resumable.
anthropic-calibrate measure -corpus corpus.jsonl -out counts.jsonl -rate 4

# Fit the correction and print Go source for anthropic_weights.go.
anthropic-calibrate fit -counts counts.jsonl

# Score the weights the library currently ships. No network.
anthropic-calibrate report -counts testdata/anthropic-counts.jsonl
```

A corpus row is a JSON object on its own line, with `id`, `category` and `text`. The `measure` step adds `apiCounts`. A `category` of `repetition` or `whitespace` marks a row as an adversarial run, which gets reported apart from representative text.

A corpus must cover the scripts and shapes the estimate is expected to hold for. The committed fixture is a small subset that is clean on licensing. It holds public-domain prose, this repository's own sources, and authored strings. It stays small enough to read in a diff.

The shipped weights were fitted on more than that fixture. Further labels came from recorded assistant responses. The API's own usage record states the exact output token count, for text the recording also stores. Those labels cost no requests. A spot check confirmed that they match `count_tokens` for the same text, up to a constant.

Those recordings are not committed. So `fit` over the committed fixture alone will not reproduce the shipped numbers exactly.

## Comparison with a mined-vocabulary implementation

[`rohangpta/ctoc`](https://github.com/rohangpta/ctoc) takes the other approach. It probed `count_tokens` about 276,000 times to reconstruct a vocabulary. It then matches greedily against that vocabulary, taking the longest match. The table below measures it here, on the same corpus the API labelled.

| | representative, `claude_4_5` | representative, `claude_5` | throughput |
| --- | --- | --- | --- |
| this package | 4.2% | 4.2% | 4.7 MB/s |
| ctoc | 9.9% | 26.5% | 20.1 MB/s |

The accuracy gap on `claude_5` is why the split above matters. A mined vocabulary is a snapshot of whichever tokenizer was live when somebody mined it. This one predates the newer family.

The throughput gap is real, and it favours ctoc. Greedy longest-match over a trie is less work than BPE merges. Its implementation is also a C++ binary.
