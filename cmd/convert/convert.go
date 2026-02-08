package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sort"
	"sync"
)

type Vocab struct {
	Tokens map[string]uint32
	Merges [][2]uint32
}

func writeVarint(buf *bytes.Buffer, val uint32) {
	for val >= 0x80 {
		buf.WriteByte(byte(val) | 0x80)
		val >>= 7
	}
	buf.WriteByte(byte(val))
}

func WriteBinary(v *Vocab, out io.Writer) error {
	fmt.Fprintf(os.Stderr, "Grouping %d tokens by length...\n", len(v.Tokens))

	byLength := make(map[int][]struct {
		token []byte
		id    uint32
	})

	for token, id := range v.Tokens {
		b := []byte(token)
		byLength[len(b)] = append(byLength[len(b)], struct {
			token []byte
			id    uint32
		}{b, id})
	}

	lengths := make([]int, 0, len(byLength))
	for l := range byLength {
		lengths = append(lengths, l)
	}
	sort.Ints(lengths)
	fmt.Fprintf(os.Stderr, "Found %d length groups\n", len(lengths))

	// Parallel encode each length group
	buffers := make([]*bytes.Buffer, len(lengths))
	var wg sync.WaitGroup
	for i, length := range lengths {
		wg.Add(1)
		go func(i, length int) {
			defer wg.Done()
			tokens := byLength[length]
			sort.Slice(tokens, func(a, b int) bool {
				return tokens[a].id < tokens[b].id
			})

			buf := &bytes.Buffer{}
			binary.Write(buf, binary.LittleEndian, uint16(length))
			writeVarint(buf, uint32(len(tokens)))
			var prevID uint32
			for _, t := range tokens {
				buf.Write(t.token)
				writeVarint(buf, t.id-prevID)
				prevID = t.id
			}
			buffers[i] = buf
		}(i, length)
	}
	wg.Wait()

	// Write header
	out.Write([]byte("BPEV"))
	binary.Write(out, binary.LittleEndian, uint32(2)) // version 2: delta varint
	binary.Write(out, binary.LittleEndian, uint32(len(v.Tokens)))
	binary.Write(out, binary.LittleEndian, uint32(len(lengths)))

	// Concat buffers
	for _, buf := range buffers {
		out.Write(buf.Bytes())
	}

	// Merges (also delta-varint)
	fmt.Fprintf(os.Stderr, "Writing %d merges...\n", len(v.Merges))
	mergeBuf := &bytes.Buffer{}
	writeVarint(mergeBuf, uint32(len(v.Merges)))
	for _, merge := range v.Merges {
		writeVarint(mergeBuf, merge[0])
		writeVarint(mergeBuf, merge[1])
	}
	out.Write(mergeBuf.Bytes())

	fmt.Fprintf(os.Stderr, "Done! Wrote %d vocab entries, %d merges\n", len(v.Tokens), len(v.Merges))
	return nil
}
