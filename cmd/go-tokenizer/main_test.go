package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// run executes a command with the given stdin and arguments, returning stdout
// alone so tests prove data lands there. Use runErr to inspect stderr.
func run(t *testing.T, stdin string, args ...string) (string, error) {
	t.Helper()
	stdout, _, err := runErr(t, stdin, args...)
	return stdout, err
}

// runErr is like run but returns stdout and stderr separately. Each call builds
// its own command tree, so tests hold no shared state and run in parallel.
func runErr(t *testing.T, stdin string, args ...string) (string, string, error) {
	t.Helper()

	stdout := &bytes.Buffer{}
	stderr := &bytes.Buffer{}
	root := newRootCmd()
	root.SetOut(stdout)
	root.SetErr(stderr)
	root.SetIn(strings.NewReader(stdin))
	root.SetArgs(args)

	err := root.Execute()
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

// TestCountBatch verifies the JSON-lines batch protocol: a JSON-encoded string
// per input line yields its decimal count per output line, in input order.
func TestCountBatch(t *testing.T) {
	out, err := run(t, "\"Hello World\"\n\"Hello\"\n\"The quick brown fox\"\n", "count", "--batch")
	require.Nil(t, err)

	assert.Equal(t, "2\n1\n4\n", out)
}

func TestCountBatchMatchesSingleCount(t *testing.T) {
	text := "func main() {\n\tfmt.Println(\"hi\")\n}\n"
	single, err := run(t, text, "count")
	require.Nil(t, err)

	enc, jerr := json.Marshal(text)
	require.Nil(t, jerr)

	batch, err := run(t, string(enc)+"\n", "count", "--batch")
	require.Nil(t, err)

	assert.Equal(t, strings.TrimSpace(single), strings.TrimSpace(batch))
}

func TestCountBatchGemma(t *testing.T) {
	out, err := run(t, "\"Hello World\"\n", "count", "--batch", "--encoding", "gemma")
	require.Nil(t, err)

	single, err := run(t, "Hello World", "count", "--encoding", "gemma")
	require.Nil(t, err)

	assert.Equal(t, strings.TrimSpace(single), strings.TrimSpace(out))
}

// Newlines inside a section arrive JSON-escaped, so a multi-line text stays a
// single input line and yields a single count.
func TestCountBatchMultilineSection(t *testing.T) {
	out, err := run(t, `"line one\nline two\nline three"`+"\n", "count", "--batch")
	require.Nil(t, err)

	require.Equal(t, 1, len(strings.Fields(out)))
}

func TestCountBatchEmptyStringCountsZero(t *testing.T) {
	out, err := run(t, "\"\"\n", "count", "--batch")
	require.Nil(t, err)

	assert.Equal(t, "0\n", out)
}

func TestCountBatchSkipsBlankLines(t *testing.T) {
	out, err := run(t, "\n\"Hello World\"\n\n  \n\"Hello\"\n\n", "count", "--batch")
	require.Nil(t, err)

	assert.Equal(t, "2\n1\n", out)
}

func TestCountBatchNoTrailingNewline(t *testing.T) {
	out, err := run(t, "\"Hello World\"", "count", "--batch")
	require.Nil(t, err)

	assert.Equal(t, "2\n", out)
}

// A malformed line must abort the run with an error naming the line number --
// a silent gap in the counts would desynchronize the caller.
func TestCountBatchMalformedLine(t *testing.T) {
	_, err := run(t, "\"ok\"\nnot json\n", "count", "--batch")
	require.NotNil(t, err)

	assert.Contains(t, err.Error(), "line 2")
}

// A JSON value that is not a string (a number, an object) is rejected too.
func TestCountBatchNonStringJSON(t *testing.T) {
	_, err := run(t, "42\n", "count", "--batch")
	require.NotNil(t, err)

	assert.Contains(t, err.Error(), "line 1")
}

func TestCountBatchRejectsArgs(t *testing.T) {
	_, err := run(t, "", "count", "--batch", "some text")
	require.NotNil(t, err)
}

func TestCountBatchRejectsInputFlag(t *testing.T) {
	_, err := run(t, "", "count", "--batch", "--input", "somefile.txt")
	require.NotNil(t, err)
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
