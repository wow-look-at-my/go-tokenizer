package main

import (
	"bufio"
	"encoding/base64"
	"encoding/json"
	"fmt"
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
		vocab, err = loadHuggingFace(os.Args[1])
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

func loadHuggingFace(path string) (*Vocab, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var hf struct {
		Model struct {
			Vocab  map[string]int `json:"vocab"`
			Merges [][]string     `json:"merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(data, &hf); err != nil {
		return nil, err
	}

	vocab := &Vocab{Tokens: make(map[string]uint32)}
	for token, id := range hf.Model.Vocab {
		vocab.Tokens[token] = uint32(id)
	}

	for _, merge := range hf.Model.Merges {
		id1, ok1 := hf.Model.Vocab[merge[0]]
		id2, ok2 := hf.Model.Vocab[merge[1]]
		if ok1 && ok2 {
			vocab.Merges = append(vocab.Merges, [2]uint32{uint32(id1), uint32(id2)})
		}
	}

	return vocab, nil
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
