package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

type HFTokenizer struct {
	Model struct {
		Vocab  map[string]int `json:"vocab"`
		Merges [][]string     `json:"merges"`
	} `json:"model"`
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintf(os.Stderr, "Usage: %s <input.json> <output.bin>\n", os.Args[0])
		os.Exit(1)
	}

	data, err := os.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		os.Exit(1)
	}

	var hf HFTokenizer
	if err := json.Unmarshal(data, &hf); err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing JSON: %v\n", err)
		os.Exit(1)
	}

	out, err := os.Create(os.Args[2])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating output: %v\n", err)
		os.Exit(1)
	}
	defer out.Close()

	fmt.Fprintf(os.Stderr, "Grouping %d tokens by length...\n", len(hf.Model.Vocab))

	// Group tokens by byte length
	byLength := make(map[int][]struct {
		token []byte
		id    uint32
	})

	for token, id := range hf.Model.Vocab {
		b := []byte(token)
		byLength[len(b)] = append(byLength[len(b)], struct {
			token []byte
			id    uint32
		}{b, uint32(id)})
	}
	fmt.Fprintf(os.Stderr, "Found %d length groups\n", len(byLength))

	// Sort lengths
	lengths := make([]int, 0, len(byLength))
	for l := range byLength {
		lengths = append(lengths, l)
	}
	sort.Ints(lengths)

	// Write header: "BPEV" magic + version
	out.Write([]byte("BPEV"))
	binary.Write(out, binary.LittleEndian, uint32(1)) // version

	// Write vocab section
	fmt.Fprintf(os.Stderr, "Writing vocab...\n")
	binary.Write(out, binary.LittleEndian, uint32(len(hf.Model.Vocab))) // total vocab size
	binary.Write(out, binary.LittleEndian, uint32(len(lengths)))        // number of length groups

	for _, length := range lengths {
		tokens := byLength[length]
		// Sort by ID for consistency
		sort.Slice(tokens, func(i, j int) bool {
			return tokens[i].id < tokens[j].id
		})

		binary.Write(out, binary.LittleEndian, uint16(length))       // token length
		binary.Write(out, binary.LittleEndian, uint32(len(tokens)))  // count

		for _, t := range tokens {
			out.Write(t.token)
			binary.Write(out, binary.LittleEndian, t.id)
		}
	}

	// Write merges section
	fmt.Fprintf(os.Stderr, "Writing %d merges...\n", len(hf.Model.Merges))
	binary.Write(out, binary.LittleEndian, uint32(len(hf.Model.Merges)))
	for i, merge := range hf.Model.Merges {
		if i%100000 == 0 && i > 0 {
			fmt.Fprintf(os.Stderr, "  %d/%d merges\n", i, len(hf.Model.Merges))
		}
		id1, ok1 := hf.Model.Vocab[merge[0]]
		id2, ok2 := hf.Model.Vocab[merge[1]]
		if !ok1 || !ok2 {
			fmt.Fprintf(os.Stderr, "Warning: merge references unknown token\n")
			continue
		}
		binary.Write(out, binary.LittleEndian, uint32(id1))
		binary.Write(out, binary.LittleEndian, uint32(id2))
	}

	fmt.Fprintf(os.Stderr, "Done! Wrote %d vocab entries, %d merges\n", len(hf.Model.Vocab), len(hf.Model.Merges))
}
