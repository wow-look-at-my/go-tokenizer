# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
go-safe-build
```

## Architecture

This is a Go BPE (Byte Pair Encoding) tokenizer library supporting OpenAI's tiktoken encodings and Google's Gemma.

### Core Components

- **tokenizer.go** - Main `Tokenizer` interface with `Encode`, `Decode`, `CountTokens`. Uses functional options pattern (`WithPattern`, `WithCacheSize`, etc.)
- **bpe.go** - BPE algorithm with two modes:
  - `encodeWithRanks` - tiktoken style, uses token rank as merge priority
  - `encodeWithMerges` - HuggingFace/SentencePiece style, uses explicit merge rules
- **vocab.go** - Vocabulary loading from tiktoken format (base64 text) or binary format
- **pretokenizer.go** - Text splitting before BPE. `regexp2` handles an arbitrary pattern. The cl100k pattern takes a hand-written scanner instead, because backtracking dominated the cost of counting. `TestFastScanMatchesRegexp` holds the two to identical output.
- **encodings.go** - Encoding configs (cl100k_base, p50k_base, o200k_base, gemma, claude_4_5, claude_5) with patterns and special tokens
- **anthropic.go** - Offline estimators for Anthropic's unpublished tokenizers. `claude_4_5` and `claude_5` are separate flavors because measurement found two live tokenizers that differ only on ASCII letters. They count and refuse to encode, since no vocabulary exists to give IDs from. See docs/anthropic-estimator.md for the method, the measured error, and how to refit.
- **cmd/anthropic-calibrate/** - Maintenance command that measures a corpus against the token counting API, refits `anthropic_weights.go`, and scores the shipped weights. Never invoked by the library or the build.

### Command-line tools

- **cmd/go-tokenizer/** - User-facing CLI (cobra) with `encode`, `decode`, `count`, and `encodings` subcommands. A command per file registers itself through `init()`. `newRootCmd` builds a fresh tree per invocation, so flags never live in package state. Data goes to `cmd.OutOrStdout()` rather than `cmd.Println`, which cobra sends to stderr and which breaks an `encode | decode` pipeline. See docs/count-batch.md for the `count --batch` streaming protocol.
- **cmd/convert/** - Build-time tool that converts HuggingFace/tiktoken vocabularies into the compact binary format embedded in `embed/`.

### Vocabulary Formats

1. **Tiktoken** (.tiktoken) - `<base64_token> <rank>` per line
2. **Binary** - Compact format with "BPEV" magic, grouped by token length, includes merge rules

### Embedded Vocabularies

The `embed/` package embeds vocabulary files at compile time:
- `cl100k_base.bin.zst` - GPT-4 encoding (binary format, zstd-compressed)
- `gemma.bin.zst` - Gemma encoding (binary format with merges, zstd-compressed)

These blobs are committed to the repo. They are regenerated from upstream sources
(OpenAI's published `cl100k_base.tiktoken` and the HuggingFace gemma tokenizer) by
running `just regen-vocab`, which downloads, converts via `cmd/convert`, and
zstd-compresses them. Regeneration is a manual maintenance step, not part of the
build, so there are no `//go:generate` directives (CI does not download vocab).

### Text Normalization

Gemma uses SentencePiece-style normalization (space → ▁). Normalization functions are configured per encoding in `EncodingConfig`.
