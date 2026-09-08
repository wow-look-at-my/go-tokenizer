package main

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/spf13/cobra"

	"github.com/wow-look-at-my/go-tokenizer/tokenizer"
)

// options carries the flag values of a single command invocation. The command
// tree binds flags here rather than into package variables, so concurrent
// invocations inside a process never share flag state.
type options struct {
	encoding string // named encoding to use (cl100k_base, gemma, claude, ...)
	vocab    string // path to a custom .tiktoken vocabulary file
	pattern  string // custom pre-tokenization pattern for --vocab
	input    string // read text from this file instead of args/stdin
}

// subcommands holds a constructor per subcommand, appended by its own init().
var subcommands []func(*options) *cobra.Command

// register adds a subcommand constructor to the command tree.
func register(newCmd func(*options) *cobra.Command) {
	subcommands = append(subcommands, newCmd)
}

// newRootCmd builds a complete command tree with its own flag storage.
func newRootCmd() *cobra.Command {
	opts := &options{}

	root := &cobra.Command{
		Use:   "go-tokenizer",
		Short: "BPE tokenizer for OpenAI tiktoken, Google Gemma, and Anthropic Claude encodings",
		Long: `go-tokenizer encodes text into BPE token IDs, decodes token IDs back into
text, and counts tokens using OpenAI tiktoken-compatible, Google Gemma, and
Anthropic Claude encodings.

Text is read from positional arguments, a file (--input), or standard input.`,
		SilenceUsage:  true,
		SilenceErrors: true,
	}

	pf := root.PersistentFlags()
	pf.StringVarP(&opts.encoding, "encoding", "e", tokenizer.DefaultEncoding, "named encoding to use")
	pf.StringVar(&opts.vocab, "vocab", "", "path to a custom .tiktoken vocabulary file (overrides --encoding)")
	pf.StringVar(&opts.pattern, "pattern", "", "custom pre-tokenization regex (only used with --vocab)")

	for _, newCmd := range subcommands {
		root.AddCommand(newCmd(opts))
	}
	return root
}

// Execute runs the root command and reports a failure in the exit status.
func Execute() {
	if err := newRootCmd().Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// newTokenizer builds a tokenizer from the shared --encoding / --vocab flags.
func (o *options) newTokenizer() (tokenizer.Tokenizer, error) {
	if o.vocab != "" {
		var opts []tokenizer.Option
		if o.pattern != "" {
			opts = append(opts, tokenizer.WithPattern(o.pattern))
		}
		return tokenizer.NewFromFile(o.vocab, opts...)
	}
	return tokenizer.NewWithEncoding(o.encoding)
}

// addInputFlag registers the shared -i/--input flag on a command.
func (o *options) addInputFlag(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&o.input, "input", "i", "", "read input from this file instead of args/stdin")
}

// readText resolves the text to operate on: --input file, else positional
// arguments joined by spaces, else all of standard input.
func (o *options) readText(cmd *cobra.Command, args []string) (string, error) {
	if o.input != "" {
		b, err := os.ReadFile(o.input)
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
