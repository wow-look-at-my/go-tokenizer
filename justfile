[private]
help:
	@just --list

build:
	go-safe-build

test:
	go-safe-build --min-coverage 0

compare-compression:
	#!/usr/bin/env bash
	set -euo pipefail
	src="embed/gemma.bin"
	t="/tmp/gemma-cmp"
	rm -rf "$t"
	mkdir -p "$t"
	echo "=== Compression ==="
	hyperfine --warmup 1 --runs 5 \
		"gzip -9 -k -c $src > $t/g.gz" \
		"zstd -q -c $src > $t/g.zst" \
		"zstd -q -19 -c $src > $t/g.zst19" \
		"xz -9 -k -c $src > $t/g.xz" \
		"bzip2 -9 -k -c $src > $t/g.bz2" \
		"lz4 -9 -q -c $src > $t/g.lz4"
	echo ""
	echo "=== Sizes ==="
	ls -lhS "$t"
	echo ""
	echo "=== Decompression ==="
	hyperfine --warmup 3 \
		"gzip -d -c $t/g.gz" \
		"zstd -d -c $t/g.zst" \
		"zstd -d -c $t/g.zst19" \
		"xz -d -c $t/g.xz" \
		"bzip2 -d -c $t/g.bz2" \
		"lz4 -d -c $t/g.lz4"
	rm -rf "$t"

download-gemma:
	hf download google/embeddinggemma-300m \
		--include "tokenizer*" --include "special_tokens_map.json" \
		--local-dir embeddinggemma_files

# Regenerate the embedded cl100k_base vocab from OpenAI's published tiktoken file
regen-cl100k:
	curl -sfLo embed/cl100k_base.tiktoken https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
	go run ./cmd/convert embed/cl100k_base.tiktoken embed/cl100k_base.bin
	zstd -19 -q --rm -f embed/cl100k_base.bin

# Regenerate the embedded gemma vocab (downloads the HuggingFace tokenizer first)
regen-gemma: download-gemma
	go run ./cmd/convert embeddinggemma_files/tokenizer.json embed/gemma.bin
	zstd -19 -q --rm -f embed/gemma.bin

# Regenerate the embedded DeepSeek vocab. V3 through V4.1 share one vocabulary;
# V4.1 carries the largest added-token set.
regen-deepseek:
	curl -sfLo deepseek_tokenizer.json https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/tokenizer.json
	go run ./cmd/convert deepseek_tokenizer.json embed/deepseek.bin
	zstd -19 -q --rm -f embed/deepseek.bin
	rm deepseek_tokenizer.json

# Regenerate hfclasses.ranges, the Unicode classes HuggingFace tokenizers applies to \p{..} and \s.
# Needs node; installs the tokenizers package into a temporary directory.
regen-hf-classes:
	#!/usr/bin/env bash
	set -euo pipefail
	t="$(mktemp -d)"
	npm install --silent --prefix "$t" tokenizers@0.23.2
	curl -sfLo "$t/tokenizer.json" https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/tokenizer.json
	NODE_PATH="$t/node_modules" node scripts/hf-unicode-classes/classes.cjs "$t/tokenizer.json" hfclasses.ranges

# Regenerate all embedded vocabularies
regen-vocab: regen-cl100k regen-gemma regen-deepseek
