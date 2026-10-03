package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"github.com/wow-look-at-my/go-containers/set"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s <input.json|input.tiktoken> <output.bin>\n", os.Args[0])
		os.Exit(1)
	}

	var vocab *Vocab
	var err error

	switch filepath.Ext(os.Args[1]) {
	case ".json":
		vocab, err = loadJSON(os.Args[1])
	case ".tiktoken":
		vocab, err = loadTiktoken(os.Args[1])
	default:
		fmt.Fprintf(os.Stderr, "Unknown format: %s\n", os.Args[1])
		os.Exit(1)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading: %v\n", err)
		os.Exit(1)
	}

	out, err := os.Create(os.Args[2])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output: %v\n", err)
		os.Exit(1)
	}
	defer out.Close()

	WriteBinary(vocab, out)
}

// loadJSON reads either JSON vocabulary dialect. HuggingFace keeps its tokens
// under "model", while Anthropic's published tokenizer keeps ranks in a single
// "bpe_ranks" string; the key that is present selects the reader.
func loadJSON(path string) (*Vocab, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var probe struct {
		BPERanks string `json:"bpe_ranks"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, err
	}
	if probe.BPERanks != "" {
		return loadAnthropic(probe.BPERanks)
	}
	return loadHuggingFace(data)
}

// loadAnthropic reads the vocabulary Anthropic publishes for Claude in
// anthropic-tokenizer-typescript. bpe_ranks is a space-separated string: a
// leading marker pair, then every token base64-encoded in rank order. The
// lowest rank is "!", the same token tiktoken files open with, which is what
// identifies the marker pair as a header rather than as vocabulary.
func loadAnthropic(ranks string) (*Vocab, error) {
	const headerFields = 2

	fields := strings.Fields(ranks)
	if len(fields) <= headerFields {
		return nil, fmt.Errorf("bpe_ranks holds %d fields, too few to carry a vocabulary", len(fields))
	}

	vocab := &Vocab{Tokens: make(map[string]uint32)}
	for i, field := range fields[headerFields:] {
		token, err := base64.StdEncoding.DecodeString(field)
		if err != nil {
			return nil, fmt.Errorf("rank %d: invalid base64 %q: %w", i, field, err)
		}
		if _, dup := vocab.Tokens[string(token)]; dup {
			return nil, fmt.Errorf("rank %d: token %q already has a rank", i, token)
		}
		vocab.Tokens[string(token)] = uint32(i)
	}

	if first := vocab.Tokens["!"]; first != 0 {
		return nil, fmt.Errorf(`rank of "!" is %d, want the first rank: the header is not %d fields wide`, first, headerFields)
	}
	return vocab, nil
}

// hfMerge is one merge rule.
type hfMerge [2]string

func (m *hfMerge) UnmarshalJSON(data []byte) error {
	var pair []string
	if err := json.Unmarshal(data, &pair); err == nil {
		if len(pair) != 2 {
			return fmt.Errorf("merge %s: want 2 parts, got %d", data, len(pair))
		}
		*m = hfMerge{pair[0], pair[1]}
		return nil
	}
	var joined string
	if err := json.Unmarshal(data, &joined); err != nil {
		return fmt.Errorf("merge %s: neither an array nor a string", data)
	}
	left, right, ok := strings.Cut(joined, " ")
	if !ok {
		return fmt.Errorf("merge %q: no space between the halves", joined)
	}
	*m = hfMerge{left, right}
	return nil
}

type hfTokenizer struct {
	Model struct {
		Vocab  map[string]int `json:"vocab"`
		Merges []hfMerge      `json:"merges"`
	} `json:"model"`
	PreTokenizer json.RawMessage `json:"pre_tokenizer"`
	AddedTokens  []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
}

// isByteLevel reports whether the pre-tokenizer chain holds a ByteLevel step.
// Such a vocabulary spells each byte as a printable rune, which the converter
// must map back to the raw byte.
func (hf *hfTokenizer) isByteLevel() bool {
	return bytes.Contains(hf.PreTokenizer, []byte(`"type":"ByteLevel"`)) ||
		bytes.Contains(hf.PreTokenizer, []byte(`"type": "ByteLevel"`))
}

func loadHuggingFace(data []byte) (*Vocab, error) {
	var hf hfTokenizer
	if err := json.Unmarshal(data, &hf); err != nil {
		return nil, err
	}
	if len(hf.Model.Vocab) == 0 {
		return nil, fmt.Errorf("no model.vocab and no bpe_ranks: unrecognized JSON vocabulary")
	}

	vocab := &Vocab{Tokens: make(map[string]uint32)}
	byteLevel := hf.isByteLevel()

	// Added tokens match on their literal text before any splitting.
	if len(hf.AddedTokens) > 0 {
		vocab.Added = make(map[string]uint32, len(hf.AddedTokens))
	}
	added := set.New[string]()
	for _, t := range hf.AddedTokens {
		vocab.Added[t.Content] = uint32(t.ID)
		added.Add(t.Content)
	}

	for token, id := range hf.Model.Vocab {
		if added.Contains(token) {
			continue
		}
		raw := token
		if byteLevel {
			b, err := byteLevelDecode(token)
			if err != nil {
				return nil, fmt.Errorf("token %d: %w", id, err)
			}
			raw = string(b)
		}
		vocab.Tokens[raw] = uint32(id)
	}

	for i, merge := range hf.Model.Merges {
		id1, ok1 := hf.Model.Vocab[merge[0]]
		id2, ok2 := hf.Model.Vocab[merge[1]]
		if !ok1 || !ok2 {
			if byteLevel {
				return nil, fmt.Errorf("merge %d (%q %q) names a token outside the vocabulary", i, merge[0], merge[1])
			}
			continue
		}
		vocab.Merges = append(vocab.Merges, [2]uint32{uint32(id1), uint32(id2)})
	}

	return vocab, nil
}

var byteLevelRunes = func() map[rune]byte {
	m := make(map[rune]byte, 256)
	next := rune(256)
	for b := 0; b < 256; b++ {
		if ('!' <= b && b <= '~') || (0xA1 <= b && b <= 0xAC) || (0xAE <= b && b <= 0xFF) {
			m[rune(b)] = byte(b)
			continue
		}
		m[next] = byte(b)
		next++
	}
	return m
}()

// byteLevelDecode maps a byte-level token spelling back to its raw bytes.
func byteLevelDecode(token string) ([]byte, error) {
	out := make([]byte, 0, len(token))
	for _, r := range token {
		b, ok := byteLevelRunes[r]
		if !ok {
			return nil, fmt.Errorf("%q holds %U, which is not a byte-level rune", token, r)
		}
		out = append(out, b)
	}
	return out, nil
}

func loadTiktoken(path string) (*Vocab, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	vocab := &Vocab{Tokens: make(map[string]uint32)}
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) != 2 {
			continue
		}
		tokenBytes, err := base64.StdEncoding.DecodeString(parts[0])
		if err != nil {
			continue
		}
		rank, err := strconv.Atoi(parts[1])
		if err != nil {
			continue
		}
		vocab.Tokens[string(tokenBytes)] = uint32(rank)
	}

	return vocab, scanner.Err()
}
