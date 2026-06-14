package main

import (
	"fmt"

	"github.com/spf13/cobra"
)

var countCmd = &cobra.Command{
	Use:   "count [text...]",
	Short: "Count the number of BPE tokens in text",
	Long: `Count reads text from the positional arguments, a file (--input), or
standard input, and prints the number of BPE tokens.`,
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

		n, err := tok.CountTokens(text)
		if err != nil {
			return err
		}

		fmt.Fprintln(cmd.OutOrStdout(), n)
		return nil
	},
}

func init() {
	addInputFlag(countCmd)
	rootCmd.AddCommand(countCmd)
}
