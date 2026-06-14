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
- **pretokenizer.go** - Regex-based text splitting before BPE. Uses `regexp2` for Unicode support
- **encodings.go** - Encoding configs (cl100k_base, p50k_base, o200k_base, gemma) with patterns and special tokens

### Command-line tools

- **cmd/go-tokenizer/** - User-facing CLI (cobra) with `encode`, `decode`, `count`, and `encodings` subcommands. One command per file, each self-registering via `init()`; `root.go` holds the shared `--encoding`/`--vocab`/`--pattern` flags and the tokenizer constructor. Note: data is written with `fmt.Fprintln(cmd.OutOrStdout(), ...)` rather than `cmd.Println` (which cobra sends to stderr), so pipelines like `encode | decode` work.
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
