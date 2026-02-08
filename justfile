[private]
help:
	@just --list

build:
	go-safe-build

test:
	go-safe-build --min-coverage 0

download-gemma:
	#!/usr/bin/env bash
	set -euo pipefail
	if [ ! -d "embeddinggemma_files" ]; then
		huggingface-cli download google/embeddinggemma-300m \
			--include "tokenizer*" "special_tokens_map.json" \
			--local-dir embeddinggemma_files
	else
		echo "embeddinggemma_files already exists"
	fi
