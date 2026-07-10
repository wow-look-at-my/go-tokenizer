package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/spf13/cobra"

	tokenizer "github.com/wow-look-at-my/go-tokenizer"
)

// countBatch is bound by the --batch flag on the count command.
var countBatch bool

var countCmd = &cobra.Command{
	Use:   "count [text...]",
	Short: "Count the number of BPE tokens in text",
	Long: `Count reads text from the positional arguments, a file (--input), or
standard input, and prints the number of BPE tokens.

With --batch, count instead counts MANY independent texts over one process
(loading the vocabulary once, not per text): each input line is one
JSON-encoded string (JSON Lines), and its token count is printed to stdout
as a bare decimal on its own line, in input order, as soon as it is
computed. That per-line reply makes the mode usable both for bulk streaming
and interactively -- a driving process may write a few sections, read their
counts, and decide what to send next, all over the same long-lived process.
Blank lines are skipped. Any line that is not a valid JSON string, or any
count that fails, aborts with an error naming the line number -- a count is
never silently dropped or guessed.`,
	Args: cobra.ArbitraryArgs,
	RunE: func(cmd *cobra.Command, args []string) error {
		if countBatch {
			// Batch mode has exactly one input source: stdin. Rejecting the
			// others loudly beats silently ignoring them.
			if len(args) > 0 {
				return errors.New("--batch reads JSON-encoded lines from standard input; positional arguments are not allowed")
			}
			if inputFile != "" {
				return errors.New("--batch reads JSON-encoded lines from standard input; --input is not allowed")
			}
			tok, err := newTokenizer()
			if err != nil {
				return err
			}
			return batchCount(tok, cmd.InOrStdin(), cmd.OutOrStdout())
		}

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

// batchCount implements `count --batch`: one JSON-encoded string per input
// line in, one decimal token count per output line out, in input order. Each
// count is written as soon as it is computed (stdout is unbuffered here), so
// a peer driving the process over pipes can rely on request/response
// behavior. EOF on stdin ends the run with exit 0; any malformed line or
// failed count aborts the whole run with an error naming the line number,
// because a caller that sent N sections must either get N counts or a loud
// failure -- never a silent gap.
func batchCount(tok tokenizer.Tokenizer, in io.Reader, out io.Writer) error {
	// ReadString (not bufio.Scanner) so a single line can grow to any size:
	// a JSON-encoded section of a multi-megabyte diff far exceeds Scanner's
	// default token limit.
	r := bufio.NewReaderSize(in, 1<<20)
	for lineNo := 1; ; lineNo++ {
		line, err := r.ReadString('\n')
		// A final line without a trailing newline arrives together with
		// io.EOF; process the data before honoring the EOF.
		if trimmed := strings.TrimSpace(line); trimmed != "" {
			var text string
			if jerr := json.Unmarshal([]byte(trimmed), &text); jerr != nil {
				return fmt.Errorf("batch line %d: not a JSON string: %v", lineNo, jerr)
			}
			n, cerr := tok.CountTokens(text)
			if cerr != nil {
				return fmt.Errorf("batch line %d: counting tokens: %v", lineNo, cerr)
			}
			if _, werr := fmt.Fprintln(out, n); werr != nil {
				return fmt.Errorf("batch line %d: writing count: %v", lineNo, werr)
			}
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("reading batch input: %v", err)
		}
	}
}

func init() {
	addInputFlag(countCmd)
	countCmd.Flags().BoolVar(&countBatch, "batch", false, "read JSON-encoded strings from stdin (one per line) and print one token count per line")
	rootCmd.AddCommand(countCmd)
}
