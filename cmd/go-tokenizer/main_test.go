package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	require.Nil(t, err)

	got := strings.TrimSpace(out)
	assert.Equal(t, "9906 4435", got)

}

func TestEncodeJSON(t *testing.T) {
	out, err := run(t, "", "encode", "--format", "json", "Hello World")
	require.Nil(t, err)

	got := strings.TrimSpace(out)
	assert.Equal(t, "[9906,4435]", got)

}

func TestEncodePretty(t *testing.T) {
	out, err := run(t, "", "encode", "-f", "pretty", "Hello World")
	require.Nil(t, err)

	assert.False(t, !strings.Contains(out, "9906") || !strings.Contains(out, "ID"))

}

func TestEncodeBadFormat(t *testing.T) {
	_, err := run(t, "", "encode", "-f", "bogus", "hi")
	require.NotNil(t, err)

}

func TestEncodeStdin(t *testing.T) {
	out, err := run(t, "Hello World", "encode")
	require.Nil(t, err)

	got := strings.TrimSpace(out)
	assert.Equal(t, "9906 4435", got)

}

func TestCount(t *testing.T) {
	out, err := run(t, "", "count", "Hello World")
	require.Nil(t, err)

	got := strings.TrimSpace(out)
	assert.Equal(t, "2", got)

}

func TestDecodeArgs(t *testing.T) {
	out, err := run(t, "", "decode", "9906", "4435")
	require.Nil(t, err)

	got := strings.TrimSpace(out)
	assert.Equal(t, "Hello World", got)

}

func TestDecodeJSONInputNoNewline(t *testing.T) {
	out, err := run(t, "", "decode", "-n", "[9906, 4435]")
	require.Nil(t, err)

	assert.Equal(t, "Hello World", out)

}

func TestDecodeStdin(t *testing.T) {
	out, err := run(t, "9906,4435", "decode")
	require.Nil(t, err)

	got := strings.TrimSpace(out)
	assert.Equal(t, "Hello World", got)

}

func TestDecodeBadID(t *testing.T) {
	_, err := run(t, "", "decode", "not-a-number")
	require.NotNil(t, err)

}

func TestRoundTrip(t *testing.T) {
	enc, err := run(t, "", "encode", "The quick brown fox")
	require.Nil(t, err)

	dec, err := run(t, "", "decode", strings.TrimSpace(enc))
	require.Nil(t, err)

	got := strings.TrimSpace(dec)
	assert.Equal(t, "The quick brown fox", got)

}

func TestEncodingSelection(t *testing.T) {
	out, err := run(t, "", "encode", "--encoding", "gemma", "Hello World")
	require.Nil(t, err)

	assert.NotEqual(t, "", strings.TrimSpace(out))

}

func TestUnknownEncoding(t *testing.T) {
	_, err := run(t, "", "count", "--encoding", "does-not-exist", "hi")
	require.NotNil(t, err)

}

func TestEncodingsList(t *testing.T) {
	out, err := run(t, "", "encodings")
	require.Nil(t, err)

	for _, want := range []string{"cl100k_base", "(default)", "embedded", "gemma"} {
		assert.Contains(t, out, want)

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
		require.Nil(t, err)

		assert.NotEqual(t, "", strings.TrimSpace(stdout))

		assert.Equal(t, "", strings.TrimSpace(stderr))

	}
}

func TestParseIDs(t *testing.T) {
	ids, err := parseIDs(" [1, 2,3]\n4 ")
	require.Nil(t, err)

	want := []int{1, 2, 3, 4}
	require.Equal(t, len(want), len(ids))

	for i := range want {
		require.Equal(t, want[i], ids[i])

	}
}

func TestJoinInts(t *testing.T) {
	assert.Equal(t, "1 2 3", joinInts([]int{1, 2, 3}, " "))
	assert.Equal(t, "", joinInts(nil, " "))
}
