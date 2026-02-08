package embed

import (
	_ "embed"
)

//go:generate go run ../cmd/convert/main.go ../embeddinggemma_files/tokenizer(1).json gemma.bin

//go:embed cl100k_base.tiktoken
var Cl100kBase []byte

//go:embed gemma.bin.zst
var GemmaZst []byte
