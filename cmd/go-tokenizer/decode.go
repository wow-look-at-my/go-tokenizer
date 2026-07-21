package main

import (
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
)

var decodeNoNewline bool

var decodeCmd = &cobra.Command{
	Use:   "decode [id...]",
	Short: "Decode BPE token IDs back into text",
	Long: `Decode reads BPE token IDs from the positional arguments or standard input
and prints the decoded text. IDs may be separated by whitespace or commas, or
given as a JSON array (e.g. "9906 4435", "9906,4435", or "[9906, 4435]").`,
	Args: cobra.ArbitraryArgs,
	RunE: func(cmd *cobra.Command, args []string) error {
		raw := strings.Join(args, " ")
		if strings.TrimSpace(raw) == "" {
			b, err := io.ReadAll(cmd.InOrStdin())
			if err != nil {
				return err
			}
			raw = string(b)
		}

		ids, err := parseIDs(raw)
		if err != nil {
			return err
		}

		tok, err := newTokenizer()
		if err != nil {
			return err
		}

		text, err := tok.Decode(ids)
		if err != nil {
			return err
		}

		if decodeNoNewline {
			fmt.Fprint(cmd.OutOrStdout(), text)
		} else {
			fmt.Fprintln(cmd.OutOrStdout(), text)
		}
		return nil
	},
}

func init() {
	decodeCmd.Flags().BoolVarP(&decodeNoNewline, "no-newline", "n", false, "do not print a trailing newline")
	rootCmd.AddCommand(decodeCmd)
}

// parseIDs extracts integer token IDs from a free-form string, tolerating
// whitespace, comma, and JSON-array ("[1, 2]") separators.
func parseIDs(s string) ([]int, error) {
	cleaned := strings.NewReplacer("[", " ", "]", " ", ",", " ").Replace(s)
	fields := strings.Fields(cleaned)
	ids := make([]int, 0, len(fields))
	for _, f := range fields {
		n, err := strconv.Atoi(f)
		if err != nil {
			return nil, fmt.Errorf("invalid token id %q: %w", f, err)
		}
		ids = append(ids, n)
	}
	return ids, nil
}
