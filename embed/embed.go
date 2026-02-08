package embed

import (
	_ "embed"
)

//go:generate go run ../cmd/convert ../embeddinggemma_files/tokenizer.json gemma.bin
//go:generate zstd -19 -q --rm -f gemma.bin

//go:generate curl -sfLo cl100k_base.tiktoken https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
//go:generate go run ../cmd/convert cl100k_base.tiktoken cl100k_base.bin
//go:generate zstd -19 -q --rm -f cl100k_base.bin

//go:embed cl100k_base.bin.zst
var Cl100kBaseZst []byte

//go:embed gemma.bin.zst
var GemmaZst []byte
