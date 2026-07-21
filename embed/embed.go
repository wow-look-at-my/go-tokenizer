package embed

import (
	_ "embed"
)

// The embedded vocabularies below are committed as compressed blobs and embedded
// at compile time. To rebuild them from upstream sources (OpenAI's published
// cl100k_base.tiktoken and the HuggingFace gemma tokenizer), run `just regen-vocab`
// (see the justfile). They are regenerated manually, not as part of the build.

//go:embed cl100k_base.bin.zst
var Cl100kBaseZst []byte

//go:embed gemma.bin.zst
var GemmaZst []byte
