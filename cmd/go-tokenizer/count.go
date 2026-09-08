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

func init() { register(newCountCmd) }

func newCountCmd(o *options) *cobra.Command {
	var countBatch bool

	cmd := &cobra.Command{
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
				// Batch mode reads stdin only. Rejecting the other
				// sources loudly beats ignoring them silently.
				if len(args) > 0 {
					return errors.New("--batch reads JSON-encoded lines from standard input; positional arguments are not allowed")
				}
				if o.input != "" {
					return errors.New("--batch reads JSON-encoded lines from standard input; --input is not allowed")
				}
				tok, err := o.newTokenizer()
				if err != nil {
					return err
				}
				return batchCount(tok, cmd.InOrStdin(), cmd.OutOrStdout())
			}

			text, err := o.readText(cmd, args)
			if err != nil {
				return err
			}

			tok, err := o.newTokenizer()
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

	o.addInputFlag(cmd)
	cmd.Flags().BoolVar(&countBatch, "batch", false, "read JSON-encoded strings from stdin (one per line) and print one token count per line")
	return cmd
}

// batchCount implements `count --batch`. It reads a JSON-encoded string per
// input line and writes that text's decimal token count per output line, in
// input order. Each count is written as it is computed, so a peer driving the
// process over pipes gets request/response behavior. A malformed line or a
// failed count aborts the run with an error naming the line: a caller that
// sent N sections gets N counts or a loud failure, never a silent gap.
func batchCount(tok tokenizer.Tokenizer, in io.Reader, out io.Writer) error {
	// ReadString, not Scanner: a JSON-encoded diff section outgrows Scanner.
	r := bufio.NewReaderSize(in, 1<<20)
	for lineNo := 1; ; lineNo++ {
		line, err := r.ReadString('\n')
		// A final line lacking its newline arrives with io.EOF, so the
		// data is processed before the EOF is honored below.
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
