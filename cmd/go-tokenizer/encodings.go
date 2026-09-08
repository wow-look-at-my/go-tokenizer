package main

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"

	tokenizer "github.com/wow-look-at-my/go-tokenizer"
)

func init() { register(newEncodingsCmd) }

func newEncodingsCmd(*options) *cobra.Command {
	return &cobra.Command{
		Use:   "encodings",
		Short: "List available encodings",
		Long: `List the encodings known to go-tokenizer and whether each one ships with an
embedded vocabulary. Encodings marked "not embedded" can still be used by
supplying a vocabulary file with --vocab.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return writeEncodings(cmd.OutOrStdout())
		},
	}
}

// writeEncodings renders the encoding table: every known encoding and whether
// its vocabulary is embedded in this binary.
func writeEncodings(out io.Writer) error {
	names := make([]string, 0, len(tokenizer.Encodings))
	for name := range tokenizer.Encodings {
		names = append(names, name)
	}
	sort.Strings(names)

	w := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "ENCODING\tSTATUS")
	for _, name := range names {
		tok, err := tokenizer.NewWithEncoding(name)
		var status string
		switch {
		case err == nil && tokenizer.Encodings[name].AnthropicEstimator:
			status = "estimator (counts only, no encode/decode)"
		case err == nil:
			status = fmt.Sprintf("embedded (%d tokens)", tok.VocabSize())
		case strings.Contains(err.Error(), "no embedded vocabulary"):
			status = "not embedded"
		default:
			status = "error: " + err.Error()
		}
		if name == tokenizer.DefaultEncoding {
			status += " (default)"
		}
		fmt.Fprintf(w, "%s\t%s\n", name, status)
	}
	return w.Flush()
}
