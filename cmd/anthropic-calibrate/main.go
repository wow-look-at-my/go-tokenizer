// Command anthropic-calibrate regenerates the weights the Anthropic estimator
// ships with. It is a maintenance tool, run by hand: the library itself never
// contacts the network, and nothing in the build depends on this command.
//
// The procedure runs in stages.
//
//	measure  ask the token counting API what a corpus costs, per family
//	fit      solve for the correction and print it as Go source
//	report   score the weights the library currently ships
//
// Only measure needs a key, in ANTHROPIC_API_KEY. It is rate limited and prints
// what it spends, because the endpoint is a shared service and the corpus is
// walked per family.
package main

import (
	"encoding/json"
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	var err error
	switch os.Args[1] {
	case "measure":
		err = runMeasure(os.Args[2:])
	case "fit":
		err = runFit(os.Args[2:])
	case "report":
		err = runReport(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `usage: anthropic-calibrate <command> [flags]

  measure -corpus FILE -out FILE [-rate N] [-model-4-5 ID] [-model-5 ID]
      Ask the token counting API what each corpus sample costs, per family,
      and write labelled rows. Reads ANTHROPIC_API_KEY. Already-labelled rows
      in an existing -out file are kept, so an interrupted run resumes.

  fit -counts FILE [-lambda L]
      Fit the correction against labelled rows and print Go source for
      anthropic_weights.go. Makes no network call.

  report -counts FILE
      Score the weights the library currently ships against labelled rows,
      split into representative text and single-character runs. No network.

A labelled row is a JSON object per line:
      {"id":..., "category":..., "text":..., "apiCounts":{"claude_5":123}}
`)
}

// labelledRow is a measured sample. The counts are what the API reported for
// a whole message, so the per-family message overhead is part of them.
type labelledRow struct {
	ID        string         `json:"id"`
	Category  string         `json:"category"`
	Text      string         `json:"text"`
	APICounts map[string]int `json:"apiCounts"`
}

func readRows(path string) ([]labelledRow, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rows []labelledRow
	for i, line := range splitLines(data) {
		var r labelledRow
		if err := json.Unmarshal(line, &r); err != nil {
			return nil, fmt.Errorf("line %d: %w", i+1, err)
		}
		rows = append(rows, r)
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("%s holds no rows", path)
	}
	return rows, nil
}

func writeRows(path string, rows []labelledRow) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	enc := json.NewEncoder(f)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	return nil
}

// splitLines returns the non-blank lines of data.
func splitLines(data []byte) [][]byte {
	var out [][]byte
	start := 0
	for i := 0; i <= len(data); i++ {
		if i == len(data) || data[i] == '\n' {
			if line := trimSpace(data[start:i]); len(line) > 0 {
				out = append(out, line)
			}
			start = i + 1
		}
	}
	return out
}

func trimSpace(b []byte) []byte {
	for len(b) > 0 && (b[0] == ' ' || b[0] == '\t' || b[0] == '\r') {
		b = b[1:]
	}
	for len(b) > 0 && (b[len(b)-1] == ' ' || b[len(b)-1] == '\t' || b[len(b)-1] == '\r') {
		b = b[:len(b)-1]
	}
	return b
}
