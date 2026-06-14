# go-tokenizer

A Go library for BPE (Byte Pair Encoding) tokenization, compatible with OpenAI's tiktoken and Google's Gemma.

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
| `-f, --format` | `encode` | Output format: `ids` (default), `json`, or `pretty` |
| `-n, --no-newline` | `decode` | Do not print a trailing newline |

## Supported Encodings

| Encoding | Models | Vocab Size |
|----------|--------|------------|
| `cl100k_base` | GPT-4, GPT-3.5-turbo | 100k |
| `p50k_base` | text-davinci-003 | 50k |
| `o200k_base` | GPT-4o | 200k |
| `gemma` | Gemma | 256k |

Only `cl100k_base` and `gemma` ship with embedded vocabularies. `p50k_base` and
`o200k_base` define their patterns and special tokens but need a vocabulary file
supplied via `--vocab` (CLI) or `NewFromFile` (library).

## Features

- Embedded vocabularies - no external files needed
- LRU caching for repeated text
- Special token handling
- `CountTokens` optimized path
- Thread-safe
