// Command go-tokenizer is a command-line interface for the go-tokenizer BPE
// tokenizer library. It encodes text into token IDs, decodes token IDs back
// into text, and counts tokens using OpenAI tiktoken-compatible and Google
// Gemma encodings.
package main

func main() {
	Execute()
}
