package tokenizer

// BPE implements the Byte Pair Encoding algorithm
type BPE struct {
	vocab *Vocabulary
}

// NewBPE creates a new BPE encoder with the given vocabulary
func NewBPE(vocab *Vocabulary) *BPE {
	return &BPE{vocab: vocab}
}

// Encode converts a byte sequence to token IDs using BPE
func (b *BPE) Encode(input []byte) []int {
	if len(input) == 0 {
		return nil
	}

	// Check if entire input is a single token
	if rank, ok := b.vocab.Encode(input); ok {
		return []int{rank}
	}

	if b.vocab.HasMerges() {
		return b.encodeWithMerges(input)
	}
	return b.encodeWithRanks(input)
}

// encodeWithRanks uses token rank as merge priority (tiktoken style)
func (b *BPE) encodeWithRanks(input []byte) []int {
	pieces := make([][]byte, len(input))
	for i, by := range input {
		pieces[i] = []byte{by}
	}

	for len(pieces) > 1 {
		bestIdx := -1
		bestRank := -1

		for i := 0; i < len(pieces)-1; i++ {
			merged := append(pieces[i], pieces[i+1]...)
			if rank, ok := b.vocab.Encode(merged); ok {
				if bestIdx == -1 || rank < bestRank {
					bestIdx = i
					bestRank = rank
				}
			}
		}

		if bestIdx == -1 {
			break
		}

		merged := append(pieces[bestIdx], pieces[bestIdx+1]...)
		newPieces := make([][]byte, 0, len(pieces)-1)
		newPieces = append(newPieces, pieces[:bestIdx]...)
		newPieces = append(newPieces, merged)
		newPieces = append(newPieces, pieces[bestIdx+2:]...)
		pieces = newPieces
	}

	return b.piecesToIDs(pieces)
}

// encodeWithMerges uses explicit merge rules (HuggingFace style)
func (b *BPE) encodeWithMerges(input []byte) []int {
	// Start with token IDs for each character (not byte)
	// SentencePiece-style tokenizers use characters as base units
	ids := make([]int, 0, len(input))
	text := string(input)
	for _, r := range text {
		charBytes := []byte(string(r))
		if id, ok := b.vocab.Encode(charBytes); ok {
			ids = append(ids, id)
		} else {
			// Unknown character - try byte fallback
			for _, by := range charBytes {
				if id, ok := b.vocab.Encode([]byte{by}); ok {
					ids = append(ids, id)
				}
			}
		}
	}

	for len(ids) > 1 {
		bestIdx := -1
		bestPriority := -1

		for i := 0; i < len(ids)-1; i++ {
			if priority, ok := b.vocab.MergePriority(ids[i], ids[i+1]); ok {
				if bestIdx == -1 || priority < bestPriority {
					bestIdx = i
					bestPriority = priority
				}
			}
		}

		if bestIdx == -1 {
			break
		}

		// Get merged token
		token1, _ := b.vocab.Decode(ids[bestIdx])
		token2, _ := b.vocab.Decode(ids[bestIdx+1])
		merged := append(token1, token2...)
		mergedID, ok := b.vocab.Encode(merged)
		if !ok {
			break
		}

		// Apply merge
		newIDs := make([]int, 0, len(ids)-1)
		newIDs = append(newIDs, ids[:bestIdx]...)
		newIDs = append(newIDs, mergedID)
		newIDs = append(newIDs, ids[bestIdx+2:]...)
		ids = newIDs
	}

	return ids
}

func (b *BPE) piecesToIDs(pieces [][]byte) []int {
	result := make([]int, 0, len(pieces))
	for _, piece := range pieces {
		if rank, ok := b.vocab.Encode(piece); ok {
			result = append(result, rank)
		}
	}
	return result
}

// CountTokens returns the number of tokens without allocating the full result
func (b *BPE) CountTokens(input []byte) int {
	if len(input) == 0 {
		return 0
	}

	if _, ok := b.vocab.Encode(input); ok {
		return 1
	}

	if b.vocab.HasMerges() {
		return b.countWithMerges(input)
	}
	return b.countWithRanks(input)
}

func (b *BPE) countWithRanks(input []byte) int {
	pieceCount := len(input)
	pieces := make([][]byte, len(input))
	for i, by := range input {
		pieces[i] = []byte{by}
	}

	for pieceCount > 1 {
		bestIdx := -1
		bestRank := -1

		for i := 0; i < len(pieces)-1; i++ {
			if pieces[i] == nil {
				continue
			}
			nextIdx := -1
			for j := i + 1; j < len(pieces); j++ {
				if pieces[j] != nil {
					nextIdx = j
					break
				}
			}
			if nextIdx == -1 {
				break
			}

			merged := append(pieces[i], pieces[nextIdx]...)
			if rank, ok := b.vocab.Encode(merged); ok {
				if bestIdx == -1 || rank < bestRank {
					bestIdx = i
					bestRank = rank
				}
			}
		}

		if bestIdx == -1 {
			break
		}

		nextIdx := -1
		for j := bestIdx + 1; j < len(pieces); j++ {
			if pieces[j] != nil {
				nextIdx = j
				break
			}
		}

		pieces[bestIdx] = append(pieces[bestIdx], pieces[nextIdx]...)
		pieces[nextIdx] = nil
		pieceCount--
	}

	return pieceCount
}

func (b *BPE) countWithMerges(input []byte) int {
	// Start with token IDs for each character
	ids := make([]int, 0, len(input))
	text := string(input)
	for _, r := range text {
		charBytes := []byte(string(r))
		if id, ok := b.vocab.Encode(charBytes); ok {
			ids = append(ids, id)
		} else {
			for _, by := range charBytes {
				if id, ok := b.vocab.Encode([]byte{by}); ok {
					ids = append(ids, id)
				}
			}
		}
	}

	idCount := len(ids)

	for idCount > 1 {
		bestIdx := -1
		bestPriority := -1

		for i := 0; i < len(ids)-1; i++ {
			if ids[i] == -1 {
				continue
			}
			nextIdx := -1
			for j := i + 1; j < len(ids); j++ {
				if ids[j] != -1 {
					nextIdx = j
					break
				}
			}
			if nextIdx == -1 {
				break
			}

			if priority, ok := b.vocab.MergePriority(ids[i], ids[nextIdx]); ok {
				if bestIdx == -1 || priority < bestPriority {
					bestIdx = i
					bestPriority = priority
				}
			}
		}

		if bestIdx == -1 {
			break
		}

		// Find next non-sentinel after bestIdx
		nextIdx := -1
		for j := bestIdx + 1; j < len(ids); j++ {
			if ids[j] != -1 {
				nextIdx = j
				break
			}
		}

		// Get merged token ID
		token1, _ := b.vocab.Decode(ids[bestIdx])
		token2, _ := b.vocab.Decode(ids[nextIdx])
		merged := append(token1, token2...)
		mergedID, ok := b.vocab.Encode(merged)
		if !ok {
			break
		}

		ids[bestIdx] = mergedID
		ids[nextIdx] = -1
		idCount--
	}

	return idCount
}

// Decode converts token IDs back to bytes
func (b *BPE) Decode(tokens []int) []byte {
	var result []byte
	for _, token := range tokens {
		if bytes, ok := b.vocab.Decode(token); ok {
			result = append(result, bytes...)
		}
	}
	return result
}
