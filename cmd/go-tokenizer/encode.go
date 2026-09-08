package main

import (
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/wow-look-at-my/go-tokenizer/tokenizer"
)

func init() { register(newEncodeCmd) }

func newEncodeCmd(o *options) *cobra.Command {
	var encodeFormat string

	cmd := &cobra.Command{
		Use:   "encode [text...]",
		Short: "Encode text into BPE token IDs",
		Long: `Encode reads text from the positional arguments, a file (--input), or
standard input, and prints the resulting BPE token IDs.

Output formats (--format):
  ids     space-separated token IDs (default)
  json    a JSON array of token IDs
  pretty  a table of each token ID and its decoded text`,
		Args: cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			text, err := o.readText(cmd, args)
			if err != nil {
				return err
			}

			tok, err := o.newTokenizer()
			if err != nil {
				return err
			}

			ids, err := tok.Encode(text)
			if err != nil {
				return err
			}

			return writeEncoded(cmd.OutOrStdout(), tok, ids, encodeFormat)
		},
	}

	cmd.Flags().StringVarP(&encodeFormat, "format", "f", "ids", "output format: ids, json, or pretty")
	o.addInputFlag(cmd)
	return cmd
}

// writeEncoded renders token IDs in the named format. An unknown format is an
// error rather than a silent fallback, so a typo cannot masquerade as output.
func writeEncoded(out io.Writer, tok tokenizer.Tokenizer, ids []int, format string) error {
	switch format {
	case "ids":
		fmt.Fprintln(out, joinInts(ids, " "))
	case "json":
		b, err := json.Marshal(ids)
		if err != nil {
			return err
		}
		fmt.Fprintln(out, string(b))
	case "pretty":
		w := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
		fmt.Fprintln(w, "ID\tTOKEN")
		for _, id := range ids {
			piece, err := tok.Decode([]int{id})
			if err != nil {
				return err
			}
			fmt.Fprintf(w, "%d\t%s\n", id, strconv.Quote(piece))
		}
		return w.Flush()
	default:
		return fmt.Errorf("unknown --format %q (want ids, json, or pretty)", format)
	}
	return nil
}
