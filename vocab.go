package tokenizer

import (
	"bufio"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// Vocabulary holds the BPE token mappings
type Vocabulary struct {
	encoder map[string]int // token bytes -> rank
	decoder map[int][]byte // rank -> token bytes
	merges  map[string]int // "id1,id2" -> merge priority (lower = higher priority)
}

// NewVocabulary creates an empty vocabulary
func NewVocabulary() *Vocabulary {
	return &Vocabulary{
		encoder: make(map[string]int),
		decoder: make(map[int][]byte),
		merges:  make(map[string]int),
	}
}

// LoadTiktoken parses a .tiktoken format vocabulary file
// Format: <base64_token> <rank> per line
func LoadTiktoken(r io.Reader) (*Vocabulary, error) {
	v := NewVocabulary()

	scanner := bufio.NewScanner(r)
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) != 2 {
			return nil, fmt.Errorf("line %d: expected 2 fields, got %d", lineNum, len(parts))
		}

		tokenBytes, err := base64.StdEncoding.DecodeString(parts[0])
		if err != nil {
			return nil, fmt.Errorf("line %d: invalid base64: %w", lineNum, err)
		}

		rank, err := strconv.Atoi(parts[1])
		if err != nil {
			return nil, fmt.Errorf("line %d: invalid rank: %w", lineNum, err)
		}

		v.encoder[string(tokenBytes)] = rank
		v.decoder[rank] = tokenBytes
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("reading vocabulary: %w", err)
	}

	return v, nil
}

// LoadBinary loads vocabulary from compact binary format
// Format: "BPEV" magic, version, vocab section, merges section
func LoadBinary(data []byte) (*Vocabulary, error) {
	if len(data) < 8 || string(data[:4]) != "BPEV" {
		return nil, fmt.Errorf("invalid binary vocab: bad magic")
	}

	v := NewVocabulary()
	pos := 4

	version := binary.LittleEndian.Uint32(data[pos:])
	pos += 4
	if version != 1 {
		return nil, fmt.Errorf("unsupported version: %d", version)
	}

	// Read vocab
	vocabSize := binary.LittleEndian.Uint32(data[pos:])
	pos += 4
	numGroups := binary.LittleEndian.Uint32(data[pos:])
	pos += 4

	for i := uint32(0); i < numGroups; i++ {
		tokenLen := binary.LittleEndian.Uint16(data[pos:])
		pos += 2
		count := binary.LittleEndian.Uint32(data[pos:])
		pos += 4

		for j := uint32(0); j < count; j++ {
			token := make([]byte, tokenLen)
			copy(token, data[pos:pos+int(tokenLen)])
			pos += int(tokenLen)
			id := int(binary.LittleEndian.Uint32(data[pos:]))
			pos += 4

			v.encoder[string(token)] = id
			v.decoder[id] = token
		}
	}

	if len(v.encoder) != int(vocabSize) {
		return nil, fmt.Errorf("vocab size mismatch: got %d, expected %d", len(v.encoder), vocabSize)
	}

	// Read merges
	numMerges := binary.LittleEndian.Uint32(data[pos:])
	pos += 4

	for i := uint32(0); i < numMerges; i++ {
		id1 := binary.LittleEndian.Uint32(data[pos:])
		pos += 4
		id2 := binary.LittleEndian.Uint32(data[pos:])
		pos += 4

		key := mergePairKey(int(id1), int(id2))
		v.merges[key] = int(i) // priority = index (lower = higher priority)
	}

	return v, nil
}

func mergePairKey(id1, id2 int) string {
	return fmt.Sprintf("%d,%d", id1, id2)
}

// MergePriority returns the merge priority for a pair of token IDs
// Lower value = higher priority. Returns -1 if no merge exists.
func (v *Vocabulary) MergePriority(id1, id2 int) (int, bool) {
	priority, ok := v.merges[mergePairKey(id1, id2)]
	return priority, ok
}

// HasMerges returns true if this vocabulary has explicit merge rules
func (v *Vocabulary) HasMerges() bool {
	return len(v.merges) > 0
}

// Encode returns the rank for a token, or -1 if not found
func (v *Vocabulary) Encode(token []byte) (int, bool) {
	rank, ok := v.encoder[string(token)]
	return rank, ok
}

// Decode returns the bytes for a rank, or nil if not found
func (v *Vocabulary) Decode(rank int) ([]byte, bool) {
	bytes, ok := v.decoder[rank]
	return bytes, ok
}

// Size returns the vocabulary size
func (v *Vocabulary) Size() int {
	return len(v.encoder)
}

// AddSpecialTokens adds special tokens to the vocabulary with specified ranks
func (v *Vocabulary) AddSpecialTokens(tokens map[string]int) {
	for token, rank := range tokens {
		v.encoder[token] = rank
		v.decoder[rank] = []byte(token)
	}
}
