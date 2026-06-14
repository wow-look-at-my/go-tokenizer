package main

import (
	"bytes"
	"strings"
	"testing"

	tokenizer "github.com/wow-look-at-my/go-tokenizer"
)

// run executes the root command with the given stdin and arguments. It returns
// stdout only, so tests verify that data lands on stdout (not stderr). Use runErr
// when stderr is also of interest. Flag-bound vars are reset for independence.
func run(t *testing.T, stdin string, args ...string) (string, error) {
	t.Helper()
	stdout, _, err := runErr(t, stdin, args...)
	return stdout, err
}

// runErr is like run but returns stdout and stderr separately.
func runErr(t *testing.T, stdin string, args ...string) (string, string, error) {
	t.Helper()

	flagEncoding = tokenizer.DefaultEncoding
	flagVocab = ""
	flagPattern = ""
	inputFile = ""
	encodeFormat = "ids"
	decodeNoNewline = false

	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}
	rootCmd.SetOut(stdout)
	rootCmd.SetErr(stderr)
	rootCmd.SetIn(strings.NewReader(stdin))
	rootCmd.SetArgs(args)

	err := rootCmd.Execute()
	return stdout.String(), stderr.String(), err
}

func TestEncodeArgs(t *testing.T) {
	out, err := run(t, "", "encode", "Hello World")
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if got := strings.TrimSpace(out); got != "9906 4435" {
		t.Errorf("encode = %q, want %q", got, "9906 4435")
	}
}

func TestEncodeJSON(t *testing.T) {
	out, err := run(t, "", "encode", "--format", "json", "Hello World")
	if err != nil {
		t.Fatalf("encode json: %v", err)
	}
	if got := strings.TrimSpace(out); got != "[9906,4435]" {
		t.Errorf("encode json = %q, want %q", got, "[9906,4435]")
	}
}

func TestEncodePretty(t *testing.T) {
	out, err := run(t, "", "encode", "-f", "pretty", "Hello World")
	if err != nil {
		t.Fatalf("encode pretty: %v", err)
	}
	if !strings.Contains(out, "9906") || !strings.Contains(out, "ID") {
		t.Errorf("encode pretty missing expected content:\n%s", out)
	}
}

func TestEncodeBadFormat(t *testing.T) {
	if _, err := run(t, "", "encode", "-f", "bogus", "hi"); err == nil {
		t.Fatal("expected error for unknown format")
	}
}

func TestEncodeStdin(t *testing.T) {
	out, err := run(t, "Hello World", "encode")
	if err != nil {
		t.Fatalf("encode stdin: %v", err)
	}
	if got := strings.TrimSpace(out); got != "9906 4435" {
		t.Errorf("encode stdin = %q, want %q", got, "9906 4435")
	}
}

func TestCount(t *testing.T) {
	out, err := run(t, "", "count", "Hello World")
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	if got := strings.TrimSpace(out); got != "2" {
		t.Errorf("count = %q, want %q", got, "2")
	}
}

func TestDecodeArgs(t *testing.T) {
	out, err := run(t, "", "decode", "9906", "4435")
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got := strings.TrimSpace(out); got != "Hello World" {
		t.Errorf("decode = %q, want %q", got, "Hello World")
	}
}

func TestDecodeJSONInputNoNewline(t *testing.T) {
	out, err := run(t, "", "decode", "-n", "[9906, 4435]")
	if err != nil {
		t.Fatalf("decode json: %v", err)
	}
	if out != "Hello World" {
		t.Errorf("decode = %q, want %q", out, "Hello World")
	}
}

func TestDecodeStdin(t *testing.T) {
	out, err := run(t, "9906,4435", "decode")
	if err != nil {
		t.Fatalf("decode stdin: %v", err)
	}
	if got := strings.TrimSpace(out); got != "Hello World" {
		t.Errorf("decode stdin = %q, want %q", got, "Hello World")
	}
}

func TestDecodeBadID(t *testing.T) {
	if _, err := run(t, "", "decode", "not-a-number"); err == nil {
		t.Fatal("expected error for invalid token id")
	}
}

func TestRoundTrip(t *testing.T) {
	enc, err := run(t, "", "encode", "The quick brown fox")
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	dec, err := run(t, "", "decode", strings.TrimSpace(enc))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got := strings.TrimSpace(dec); got != "The quick brown fox" {
		t.Errorf("round trip = %q, want %q", got, "The quick brown fox")
	}
}

func TestEncodingSelection(t *testing.T) {
	out, err := run(t, "", "encode", "--encoding", "gemma", "Hello World")
	if err != nil {
		t.Fatalf("encode gemma: %v", err)
	}
	if strings.TrimSpace(out) == "" {
		t.Error("expected gemma token output")
	}
}

func TestUnknownEncoding(t *testing.T) {
	if _, err := run(t, "", "count", "--encoding", "does-not-exist", "hi"); err == nil {
		t.Fatal("expected error for unknown encoding")
	}
}

func TestEncodingsList(t *testing.T) {
	out, err := run(t, "", "encodings")
	if err != nil {
		t.Fatalf("encodings: %v", err)
	}
	for _, want := range []string{"cl100k_base", "(default)", "embedded", "gemma"} {
		if !strings.Contains(out, want) {
			t.Errorf("encodings output missing %q:\n%s", want, out)
		}
	}
}

// TestOutputGoesToStdout guards against regressing to cobra's cmd.Println,
// which writes to stderr by default and breaks shell pipelines like
// `encode ... | decode`.
func TestOutputGoesToStdout(t *testing.T) {
	cases := [][]string{
		{"encode", "Hello World"},
		{"encode", "-f", "json", "Hello World"},
		{"count", "Hello World"},
		{"decode", "9906", "4435"},
		{"encodings"},
	}
	for _, args := range cases {
		stdout, stderr, err := runErr(t, "", args...)
		if err != nil {
			t.Fatalf("%v: %v", args, err)
		}
		if strings.TrimSpace(stdout) == "" {
			t.Errorf("%v: expected output on stdout, got none (stderr=%q)", args, stderr)
		}
		if strings.TrimSpace(stderr) != "" {
			t.Errorf("%v: unexpected stderr output %q", args, stderr)
		}
	}
}

func TestParseIDs(t *testing.T) {
	ids, err := parseIDs(" [1, 2,3]\n4 ")
	if err != nil {
		t.Fatalf("parseIDs: %v", err)
	}
	want := []int{1, 2, 3, 4}
	if len(ids) != len(want) {
		t.Fatalf("parseIDs = %v, want %v", ids, want)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("parseIDs = %v, want %v", ids, want)
		}
	}
}

func TestJoinInts(t *testing.T) {
	if got := joinInts([]int{1, 2, 3}, " "); got != "1 2 3" {
		t.Errorf("joinInts = %q, want %q", got, "1 2 3")
	}
	if got := joinInts(nil, " "); got != "" {
		t.Errorf("joinInts(nil) = %q, want empty", got)
	}
}
