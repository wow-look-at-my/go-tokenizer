package embed

import (
	_ "embed"
)

// Vocabularies are committed as compressed blobs, embedded at compile time.
// Rebuild them from upstream with `just regen-vocab`, never in the build.

//go:embed cl100k_base.bin.zst
var Cl100kBaseZst []byte

//go:embed gemma.bin.zst
var GemmaZst []byte
