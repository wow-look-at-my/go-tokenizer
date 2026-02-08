package main

import (
	"bytes"
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
