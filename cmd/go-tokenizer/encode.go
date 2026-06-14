package main

import (
	"encoding/json"
	"fmt"
	"strconv"
	"text/tabwriter"

	"github.com/spf13/cobra"
)

var encodeFormat string

var encodeCmd = &cobra.Command{
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
		text, err := readText(cmd, args)
		if err != nil {
			return err
		}

		tok, err := newTokenizer()
		if err != nil {
			return err
		}

		ids, err := tok.Encode(text)
		if err != nil {
			return err
		}

		out := cmd.OutOrStdout()
		switch encodeFormat {
		case "ids":
			fmt.Fprintln(out, joinInts(ids, " "))
		case "json":
			b, err := json.Marshal(ids)
			if err != nil {
				return err
			}
			fmt.Fprintln(out, string(b))
		case "pretty":
			w := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 0, 2, ' ', 0)
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
			return fmt.Errorf("unknown --format %q (want ids, json, or pretty)", encodeFormat)
		}
		return nil
	},
}

func init() {
	encodeCmd.Flags().StringVarP(&encodeFormat, "format", "f", "ids", "output format: ids, json, or pretty")
	addInputFlag(encodeCmd)
	rootCmd.AddCommand(encodeCmd)
}
