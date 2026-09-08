package embed

import (
	_ "embed"
)

// The vocabularies below are committed as compressed blobs and embedded at
// compile time. Rebuild them from their upstream sources with
// `just regen-vocab`; regeneration is manual, never part of the build.

//go:embed cl100k_base.bin.zst
var Cl100kBaseZst []byte

//go:embed gemma.bin.zst
var GemmaZst []byte
