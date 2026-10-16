package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	tokenizer "github.com/wow-look-at-my/go-tokenizer"
)

func TestWriteVarint(t *testing.T) {
	tests := []struct {
		val  uint32
		want []byte
	}{
		{0, []byte{0}},
		{1, []byte{1}},
		{127, []byte{127}},
		{128, []byte{0x80, 1}},
		{255, []byte{0xff, 1}},
		{16383, []byte{0xff, 0x7f}},
		{16384, []byte{0x80, 0x80, 1}},
	}

	for _, tc := range tests {
		buf := &bytes.Buffer{}
		writeVarint(buf, tc.val)
		assert.True(t, bytes.Equal(buf.Bytes(), tc.want))

	}
}

func TestWriteBinary(t *testing.T) {
	v := &Vocab{
		Tokens: map[string]uint32{
			"a": 0,
			"b": 1,
		},
		Merges: [][2]uint32{{0, 1}},
	}

	buf := &bytes.Buffer{}
	WriteBinary(v, buf)

	// Check magic
	assert.Equal(t, "BPEV", string(buf.Bytes()[:4]))

}

func TestLoadTiktoken(t *testing.T) {
	// Create temp tiktoken file: base64(token) rank
	tmp := filepath.Join(t.TempDir(), "test.tiktoken")
	// "hello" in base64 is "aGVsbG8="
	content := "aGVsbG8= 0\nd29ybGQ= 1\n" // hello=0, world=1
	if err := os.WriteFile(tmp, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	v, err := loadTiktoken(tmp)
	if err != nil {
		t.Fatalf("loadTiktoken: %v", err)
	}
	if v.Tokens["hello"] != 0 {
		t.Errorf("hello = %d, want 0", v.Tokens["hello"])
	}
	if v.Tokens["world"] != 1 {
		t.Errorf("world = %d, want 1", v.Tokens["world"])
	}
}

func TestLoadTiktokenNotFound(t *testing.T) {
	_, err := loadTiktoken("/nonexistent/file.tiktoken")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestLoadHuggingFace(t *testing.T) {
	content := []byte(`{
		"model": {
			"vocab": {"a": 0, "b": 1, "ab": 2},
			"merges": [["a", "b"]]
		}
	}`)

	v, err := loadHuggingFace(content)
	if err != nil {
		t.Fatalf("loadHuggingFace: %v", err)
	}
	if v.Tokens["a"] != 0 {
		t.Errorf("a = %d, want 0", v.Tokens["a"])
	}
	if v.Tokens["ab"] != 2 {
		t.Errorf("ab = %d, want 2", v.Tokens["ab"])
	}
	if len(v.Merges) != 1 {
		t.Errorf("len(Merges) = %d, want 1", len(v.Merges))
	}
}

func TestLoadJSONNotFound(t *testing.T) {
	_, err := loadJSON("/nonexistent/tokenizer.json")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestLoadHuggingFaceInvalid(t *testing.T) {
	_, err := loadHuggingFace([]byte("not json"))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestByteLevelDecode(t *testing.T) {
	// Ġ spells the space byte and Ċ the newline; printable ASCII stands for itself.
	got, err := byteLevelDecode("ĠhelloĊ")
	require.NoError(t, err)
	assert.Equal(t, []byte(" hello\n"), got)

	// Every byte has exactly one spelling.
	assert.Len(t, byteLevelRunes, 256)

	_, err = byteLevelDecode("<｜User｜>")
	assert.ErrorContains(t, err, "not a byte-level rune")
}

func TestHFMergeBothSpellings(t *testing.T) {
	var merges []hfMerge
	require.NoError(t, json.Unmarshal([]byte(`[["Ġ","t"],"i n"]`), &merges))
	assert.Equal(t, []hfMerge{{"Ġ", "t"}, {"i", "n"}}, merges)

	assert.Error(t, json.Unmarshal([]byte(`["nospace"]`), &merges))
	assert.Error(t, json.Unmarshal([]byte(`[["a","b","c"]]`), &merges))
	assert.Error(t, json.Unmarshal([]byte(`[7]`), &merges))
}

// byteLevelJSON is a byte-level tokenizer.json in miniature: specials inside
// model.vocab, as DeepSeek ships them, and merges in the string spelling.
const byteLevelJSON = `{
	"added_tokens": [{"id": 0, "content": "<s>"}, {"id": 5, "content": "<think>"}],
	"pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "ByteLevel"}]},
	"model": {
		"vocab": {"<s>": 0, "a": 1, "b": 2, "Ġ": 3, "ab": 4},
		"merges": ["a b"]
	}
}`

func TestLoadHuggingFaceByteLevelRoundTrip(t *testing.T) {
	vocab, err := loadHuggingFace([]byte(byteLevelJSON))
	require.NoError(t, err)
	assert.Equal(t, map[string]uint32{"a": 1, "b": 2, " ": 3, "ab": 4}, vocab.Tokens)
	assert.Equal(t, map[string]uint32{"<s>": 0, "<think>": 5}, vocab.Added)
	assert.Equal(t, [][2]uint32{{1, 2}}, vocab.Merges)

	buf := &bytes.Buffer{}
	require.NoError(t, WriteBinary(vocab, buf))
	loaded, err := tokenizer.LoadBinary(buf.Bytes())
	require.NoError(t, err)
	assert.Equal(t, map[string]int{"<s>": 0, "<think>": 5}, loaded.AddedTokens())
	id, ok := loaded.Encode([]byte(" "))
	assert.True(t, ok)
	assert.Equal(t, 3, id)
	id, ok = loaded.Encode([]byte("<think>"))
	assert.True(t, ok)
	assert.Equal(t, 5, id)
	priority, ok := loaded.MergePriority(1, 2)
	assert.True(t, ok)
	assert.Equal(t, 0, priority)
}

func TestLoadHuggingFaceByteLevelRejectsUnknownMerge(t *testing.T) {
	bad := strings.Replace(byteLevelJSON, `"a b"`, `"a z"`, 1)
	_, err := loadHuggingFace([]byte(bad))
	assert.ErrorContains(t, err, "outside the vocabulary")
}
