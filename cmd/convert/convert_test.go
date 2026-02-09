package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
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
		if !bytes.Equal(buf.Bytes(), tc.want) {
			t.Errorf("writeVarint(%d) = %v, want %v", tc.val, buf.Bytes(), tc.want)
		}
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
	if string(buf.Bytes()[:4]) != "BPEV" {
		t.Errorf("Bad magic: %q", buf.Bytes()[:4])
	}
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
	tmp := filepath.Join(t.TempDir(), "tokenizer.json")
	content := `{
		"model": {
			"vocab": {"a": 0, "b": 1, "ab": 2},
			"merges": [["a", "b"]]
		}
	}`
	if err := os.WriteFile(tmp, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	v, err := loadHuggingFace(tmp)
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

func TestLoadHuggingFaceNotFound(t *testing.T) {
	_, err := loadHuggingFace("/nonexistent/tokenizer.json")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestLoadHuggingFaceInvalid(t *testing.T) {
	tmp := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(tmp, []byte("not json"), 0644); err != nil {
		t.Fatal(err)
	}
	_, err := loadHuggingFace(tmp)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}
