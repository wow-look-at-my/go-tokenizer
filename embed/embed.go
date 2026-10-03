package embed

import (
	_ "embed"
)

// Vocabularies are committed as compressed blobs, embedded at compile time.

//go:embed cl100k_base.bin.zst
var Cl100kBaseZst []byte

//go:embed gemma.bin.zst
var GemmaZst []byte

//go:embed deepseek.bin.zst
var DeepSeekZst []byte
