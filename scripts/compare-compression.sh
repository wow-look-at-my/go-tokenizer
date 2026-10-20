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
