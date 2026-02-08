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

## Supported Encodings

| Encoding | Models | Vocab Size |
|----------|--------|------------|
| `cl100k_base` | GPT-4, GPT-3.5-turbo | 100k |
| `p50k_base` | text-davinci-003 | 50k |
| `o200k_base` | GPT-4o | 200k |
| `gemma` | Gemma | 256k |

## Features

- Embedded vocabularies - no external files needed
- LRU caching for repeated text
- Special token handling
- `CountTokens` optimized path
- Thread-safe
