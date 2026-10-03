# DeepSeek encoding

The `deepseek` encoding reproduces the tokenizer that DeepSeek publishes in `tokenizer.json`. It gives the real token IDs, so `Encode`, `Decode` and `CountTokens` all work. It is not an estimate.

## Which models it covers

DeepSeek V3, V3.2, V4-Pro, V4-Flash and V4.1-Flash share one vocabulary, one merge list and one pre-tokenizer. The files differ only in their added tokens, and each release appends to the list before it. The embedded vocabulary comes from V4.1-Flash. As a result, it holds every added token of the earlier releases with the same IDs.

| part | value |
| --- | --- |
| model | byte-level BPE, merge rules from `tokenizer.json` |
| vocabulary | 127,997 BPE tokens, plus 1,283 added tokens, IDs up to 129,279 |
| normalizer | none |
| pre-tokenizer | a `Sequence` of `Split` steps in `Isolated` mode, then `ByteLevel` |
| embedded size | 1.06 MB, zstd-compressed |

`Encode` adds no beginning-of-sentence token. A chat template adds one, along with role markers, so a chat request costs more than the bare text. The raw completions endpoint reports the bare count.

## How it differs from the tiktoken path

Details of the HuggingFace model have no tiktoken equivalent. Each changes token IDs when it is wrong.

- **Split steps nest.** Each `Split` runs over every piece the step before it produced. Text between matches stays as its own piece rather than being dropped. A run of characters that no alternative matches therefore forms one piece, and BPE can merge across its code points.
- **Merges start from bytes.** A byte-level model reaches a token only through its merges. The whole-input lookup that tiktoken uses can skip a split the merges will make, so the byte-level path turns it off.
- **Equal merges resolve from the left.** When the same pair occurs twice, the leftmost merges first. `"aaa"` is `aa` + `a`, never `a` + `aa`.
- **Added tokens match first, leftmost and longest.** They are matched on their literal text before pre-tokenization. That includes the tokens marked `special: false`, such as `<think>` and `<dsml:`.

## Unicode classes

The pre-tokenizer patterns use `\p{L}`, `\p{M}`, `\p{N}`, `\p{P}`, `\p{S}` and `\s`. HuggingFace tokenizers and the Go toolchain disagree on which code points those hold, because their Unicode tables are different versions. U+32BD4 is the measured case: Go calls it a letter, while HuggingFace and the DeepSeek API do not. Put it next to an unassigned code point, and the Go tables give one more token than the HuggingFace tables. The API agrees with HuggingFace.

`hfclasses.txt` records the HuggingFace classes for every scalar value. `scripts/hf-unicode-classes` extracts them from the Rust library through its Node binding, and `just regen-hf-classes` reruns it. The DeepSeek pre-tokenizer is a hand scanner over those tables. As a result, it does not use regexp2 and its Go tables. `TestDeepseekScanMatchesRegexp` holds the scanner to the regex path on every fixture row where both tables agree.

## How it was verified

The reference is HuggingFace's Rust `tokenizers` library, version 0.23.2, loading DeepSeek's own `tokenizer.json`. The DeepSeek V4.1-Flash raw completions endpoint on synthetic.new supplied API counts. That endpoint applies no chat template, so its `prompt_tokens` is the bare count.

| corpus | rows | Go equals reference IDs | API equals reference count |
| --- | --- | --- | --- |
| committed fixture | 1,908 | all | all 1,908 |
| wider run with Wikipedia | 4,544 | all | 1,207 sampled, all equal |

The committed fixture is `testdata/deepseek-golden.jsonl`. It holds chunks of this repository's sources, authored mixes of scripts, whitespace, digits, emoji. It added tokens, runs of one repeated string, and random code points across every plane. Each row carries the reference IDs and the API count. `TestDeepseekMatchesReference` checks `Encode`, `CountTokens` and a `Decode` round trip on every row.

The wider run added 1,250 Wikipedia extracts in multiple languages. Wikipedia text is CC BY-SA. As a result, it stays out of the repository.

The `messages/count_tokens` endpoint on synthetic.new does not give the DeepSeek count. It returned 14 for both `"a"` and `"Hello world"`, and 19 for a text the completions endpoint counts as 43. It was not used.

## Regenerating

`just regen-deepseek` rebuilds `embed/deepseek.bin.zst` from the V4.1-Flash `tokenizer.json`. Point it at a later release once that release ships with the same pre-tokenizer, and the added tokens follow. A release that changes the pre-tokenizer needs a new `deepseekSplits` and a new scanner.
