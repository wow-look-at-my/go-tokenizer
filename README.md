# go-tokenizer

A Go library for BPE (Byte Pair Encoding) tokenization, compatible with OpenAI's tiktoken and Google's Gemma, with offline estimators for Anthropic's Claude.

| encoding | tokenizer | error against reference | load | throughput |
| --- | --- | --- | --- | --- |
| `cl100k_base` | published vocabulary, exact BPE | exact | 101 ms | 5.8 MB/s |
| `gemma` | published vocabulary, exact BPE | exact | 554 ms | 1.4 MB/s |
| `claude_4_5` | estimator over `cl100k_base` | **4.2%** mean, 3.0% median, 11.0% p90 | shares `cl100k_base` | 6.0 MB/s |
| `claude_5` | estimator over `cl100k_base` | **4.2%** mean, 3.4% median, 9.5% p90 | shares `cl100k_base` | 5.9 MB/s |

Error is measured against Anthropic's `count_tokens` endpoint, over text meant to stand for real input. `cl100k_base` and `gemma` reproduce vocabularies their owners publish, so they are exact by construction and have no error to report. Throughput counts a 0.84 MB mixed corpus in process, best of several runs on a shared machine.

These numbers are worse in two places, and both are measured rather than hidden:

| case | `claude_4_5` | `claude_5` |
| --- | --- | --- |
| runs of a single repeated character | 53% mean, 21% median | 37% mean |
| uncorrected `cl100k_base`, for comparison | 19.7% mean | 35.6% mean |

A run of one repeated character costs whatever long single-character tokens a vocabulary holds. That is the part Anthropic does not publish. [docs/anthropic-estimator.md](docs/anthropic-estimator.md) records the method, the full error tables, and this limit.

Against [`rohangpta/ctoc`](https://github.com/rohangpta/ctoc), which mined a vocabulary through about 276,000 API probes, on the same corpus:

| | `claude_4_5` | `claude_5` | throughput |
| --- | --- | --- | --- |
| this library | **4.2%** | **4.2%** | 4.7 MB/s |
| ctoc | 9.9% | 26.5% | **20.1 MB/s** |

ctoc is the faster of the two, by roughly four times. It matches greedily against a trie, which is less work than BPE merges. It is also a C++ binary. Its accuracy on Claude 5 reflects a vocabulary mined before that tokenizer existed.

## Installation

```bash
go get github.com/wow-look-at-my/go-tokenizer
```

## Usage

Import the package:

```go
import "github.com/wow-look-at-my/go-tokenizer"
```

Then use it:

```go
package main

import (
	"fmt"

	"github.com/wow-look-at-my/go-tokenizer"
)

func main() {
	// Create tokenizer with default encoding (cl100k_base / GPT-4)
	tok, err := tokenizer.New()
	if err != nil {
		panic(err)
	}

	// Encode text to tokens
	tokens, _ := tok.Encode("Hello World")
	fmt.Println(tokens) // [9906 4435]

	// Decode tokens back to text
	text, _ := tok.Decode(tokens)
	fmt.Println(text) // Hello World

	// Count tokens without full encoding
	count, _ := tok.CountTokens("Hello World")
	fmt.Println(count) // 2
}
```

### Gemma

```go
tok, err := tokenizer.NewWithEncoding("gemma")
if err != nil {
	panic(err)
}

tokens, _ := tok.Encode("Hello World")
```

### Custom Vocabulary

```go
tok, err := tokenizer.NewFromFile("path/to/vocab.tiktoken",
	tokenizer.WithPattern(`[a-zA-Z]+|\d+|\s+`),
	tokenizer.WithCacheSize(50000),
)
```

## Command-line interface

A `go-tokenizer` CLI lives under `cmd/go-tokenizer`.

### Install

```bash
go install github.com/wow-look-at-my/go-tokenizer/cmd/go-tokenizer@latest
```

Or build from a checkout:

```bash
go build -o go-tokenizer ./cmd/go-tokenizer
```

### Usage

```bash
# Encode text into token IDs (reads positional args, --input file, or stdin)
$ go-tokenizer encode "Hello World"
9906 4435

# JSON output
$ go-tokenizer encode --format json "Hello World"
[9906,4435]

# Show each token ID alongside its text
$ go-tokenizer encode --format pretty "Hello World"
ID    TOKEN
9906  "Hello"
4435  " World"

# Count tokens
$ go-tokenizer count "Hello World"
2

# Batch counting: one long-lived process, vocabulary loaded once. Each stdin
# line is a JSON-encoded string; each stdout line is its token count (bare
# decimal, input order), written as soon as it is computed — so a driving
# process can bulk-stream sections or interleave request/response over pipes.
# Blank lines are skipped; a non-JSON-string line aborts with the line number.
$ printf '"Hello World"\n"Hello"\n' | go-tokenizer count --batch
2
1

# Decode token IDs back into text (accepts spaces, commas, or a JSON array)
$ go-tokenizer decode 9906 4435
Hello World

# Pipe-friendly: encode | decode round-trips
$ printf 'The quick brown fox' | go-tokenizer encode | go-tokenizer decode
The quick brown fox

# Use a different encoding
$ go-tokenizer encode --encoding gemma "Hello World"

# List available encodings
$ go-tokenizer encodings
ENCODING     STATUS
cl100k_base  embedded (100261 tokens) (default)
gemma        embedded (262144 tokens)
o200k_base   not embedded
p50k_base    not embedded
```

### Flags

| Flag | Commands | Description |
|------|----------|-------------|
| `-e, --encoding` | all | Named encoding to use (default `cl100k_base`) |
| `--vocab` | all | Path to a custom `.tiktoken` vocabulary file (overrides `--encoding`) |
| `--pattern` | all | Custom pre-tokenization regex (only used with `--vocab`) |
| `-i, --input` | `encode`, `count` | Read input text from a file instead of args/stdin |
| `--batch` | `count` | JSON-lines batch mode: each stdin line is a JSON-encoded string, each stdout line its token count (mutually exclusive with args/`--input`) |
| `-f, --format` | `encode` | Output format: `ids` (default), `json`, or `pretty` |
| `-n, --no-newline` | `decode` | Do not print a trailing newline |

## Supported Encodings

| Encoding | Models | Vocab Size |
|----------|--------|------------|
| `cl100k_base` | GPT-4, GPT-3.5-turbo | 100k |
| `p50k_base` | text-davinci-003 | 50k |
| `o200k_base` | GPT-4o | 200k |
| `gemma` | Gemma | 256k |
| `claude_4_5` | Claude 4.5, 4.6 | estimator |
| `claude_5` | Claude 5, Opus 4.7 and 4.8 | estimator |

Only `cl100k_base` and `gemma` ship with embedded vocabularies. `p50k_base` and `o200k_base` define their patterns and special tokens but need a vocabulary file supplied via `--vocab` (CLI) or `NewFromFile` (library).

## Claude token counts

Anthropic publishes no vocabulary for its current models. The `claude_*` encodings therefore estimate counts, instead of reproducing a segmentation. They answer offline, within about 4% of what the token counting API reports. `Encode` and `Decode` return an error rather than inventing token IDs.

```go
name, _ := tokenizer.EncodingForModel("claude-opus-5") // "claude_5"
est, _ := tokenizer.NewAnthropicEstimator(tokenizer.FamilyClaude5)
n, _ := est.CountTokens(text)
total := n + est.MessageOverhead() // what count_tokens reports for a message
```

Claude 4.5/4.6 and Claude 5 use different tokenizers, so pick the flavor that matches the model. Runs of a single repeated character are estimated poorly. See [docs/anthropic-estimator.md](docs/anthropic-estimator.md).

## Features

- Embedded vocabularies - no external files needed
- LRU caching for repeated text
- Special token handling
- `CountTokens` optimized path
- Thread-safe
