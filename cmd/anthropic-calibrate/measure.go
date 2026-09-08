package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"time"

	tokenizer "github.com/wow-look-at-my/go-tokenizer"
)

// countTokensURL is the sole authority on what these tokenizers charge.
const countTokensURL = "https://api.anthropic.com/v1/messages/count_tokens"

func runMeasure(args []string) error {
	fs := flag.NewFlagSet("measure", flag.ExitOnError)
	corpus := fs.String("corpus", "", "JSON Lines file of samples to measure")
	out := fs.String("out", "", "labelled rows to write, resumed if it exists")
	rate := fs.Float64("rate", 4, "requests per second, at most")
	model45 := fs.String("model-4-5", "claude-sonnet-4-6", "model id standing for the claude_4_5 family")
	model5 := fs.String("model-5", "claude-opus-5", "model id standing for the claude_5 family")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *corpus == "" || *out == "" {
		return fmt.Errorf("measure needs -corpus and -out")
	}

	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		// Failing here beats writing a file of zeros that looks like data.
		return fmt.Errorf("ANTHROPIC_API_KEY is not set; measure cannot label anything without it")
	}

	rows, err := readRows(*corpus)
	if err != nil {
		return err
	}

	// Resume: keep counts an earlier run already paid for.
	existing := map[string]map[string]int{}
	if prior, err := readRows(*out); err == nil {
		for _, r := range prior {
			existing[r.ID] = r.APICounts
		}
	}

	models := map[string]string{
		string(tokenizer.FamilyClaude45): *model45,
		string(tokenizer.FamilyClaude5):  *model5,
	}

	client := &http.Client{Timeout: 60 * time.Second}
	interval := time.Duration(float64(time.Second) / *rate)
	next := time.Now()
	spent := 0

	for i := range rows {
		if rows[i].APICounts == nil {
			rows[i].APICounts = map[string]int{}
		}
		for family, model := range models {
			if n, ok := existing[rows[i].ID][family]; ok {
				rows[i].APICounts[family] = n
				continue
			}
			if wait := time.Until(next); wait > 0 {
				time.Sleep(wait)
			}
			next = time.Now().Add(interval)

			n, err := countTokens(client, key, model, rows[i].Text)
			if err != nil {
				// Save what was already paid for before giving up.
				_ = writeRows(*out, rows)
				return fmt.Errorf("sample %s on %s: %w", rows[i].ID, model, err)
			}
			rows[i].APICounts[family] = n
			spent++
			if spent%25 == 0 {
				fmt.Fprintf(os.Stderr, "%d requests\n", spent)
				_ = writeRows(*out, rows)
			}
		}
	}

	if err := writeRows(*out, rows); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "measured %d samples with %d requests\n", len(rows), spent)
	return nil
}

func countTokens(client *http.Client, key, model, text string) (int, error) {
	body, err := json.Marshal(map[string]any{
		"model":    model,
		"messages": []map[string]string{{"role": "user", "content": text}},
	})
	if err != nil {
		return 0, err
	}

	req, err := http.NewRequest(http.MethodPost, countTokensURL, bytes.NewReader(body))
	if err != nil {
		return 0, err
	}
	req.Header.Set("x-api-key", key)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("content-type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	var parsed struct {
		InputTokens *int `json:"input_tokens"`
		Error       struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return 0, fmt.Errorf("status %d: %w", resp.StatusCode, err)
	}
	if parsed.InputTokens == nil {
		return 0, fmt.Errorf("status %d: %s", resp.StatusCode, parsed.Error.Message)
	}
	return *parsed.InputTokens, nil
}
