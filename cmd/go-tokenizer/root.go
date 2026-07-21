package main

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	tokenizer "github.com/wow-look-at-my/go-tokenizer"
)

// Persistent flags shared by every subcommand.
var (
	flagEncoding string // named encoding to use (cl100k_base, gemma, ...)
	flagVocab    string // path to a custom .tiktoken vocabulary file
	flagPattern  string // custom pre-tokenization pattern for --vocab
)

// inputFile is bound by the subcommands that read text input (encode, count).
var inputFile string

var rootCmd = &cobra.Command{
	Use:   "go-tokenizer",
	Short: "BPE tokenizer for OpenAI tiktoken and Google Gemma encodings",
	Long: `go-tokenizer encodes text into BPE token IDs, decodes token IDs back into
text, and counts tokens using OpenAI tiktoken-compatible and Google Gemma
encodings.

Text is read from positional arguments, a file (--input), or standard input.`,
	SilenceUsage:  true,
	SilenceErrors: true,
}

// Execute runs the root command and exits non-zero on error.
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func init() {
	pf := rootCmd.PersistentFlags()
	pf.StringVarP(&flagEncoding, "encoding", "e", tokenizer.DefaultEncoding, "named encoding to use")
	pf.StringVar(&flagVocab, "vocab", "", "path to a custom .tiktoken vocabulary file (overrides --encoding)")
	pf.StringVar(&flagPattern, "pattern", "", "custom pre-tokenization regex (only used with --vocab)")
}

// newTokenizer builds a tokenizer from the shared --encoding / --vocab flags.
func newTokenizer() (tokenizer.Tokenizer, error) {
	if flagVocab != "" {
		var opts []tokenizer.Option
		if flagPattern != "" {
			opts = append(opts, tokenizer.WithPattern(flagPattern))
		}
		return tokenizer.NewFromFile(flagVocab, opts...)
	}
	return tokenizer.NewWithEncoding(flagEncoding)
}

// addInputFlag registers the shared -i/--input flag on a command.
func addInputFlag(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&inputFile, "input", "i", "", "read input from this file instead of args/stdin")
}

// readText resolves the text to operate on: --input file, else positional
// arguments joined by spaces, else all of standard input.
func readText(cmd *cobra.Command, args []string) (string, error) {
	if inputFile != "" {
		b, err := os.ReadFile(inputFile)
		if err != nil {
			return "", err
		}
		return string(b), nil
	}
	if len(args) > 0 {
		return strings.Join(args, " "), nil
	}
	b, err := io.ReadAll(cmd.InOrStdin())
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// joinInts renders a slice of ints with the given separator.
func joinInts(ints []int, sep string) string {
	parts := make([]string, len(ints))
	for i, n := range ints {
		parts[i] = strconv.Itoa(n)
	}
	return strings.Join(parts, sep)
}
