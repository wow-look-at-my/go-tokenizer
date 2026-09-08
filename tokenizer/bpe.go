package tokenizer

import "container/heap"

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

// unmergeable is the rank of a pair the vocabulary does not hold. Real ranks
// are vocabulary indices, so any of them compares lower and wins the search.
const unmergeable = int(^uint(0) >> 1)

// mergeByRanks applies the tiktoken merge rule to one pre-token: repeatedly
// merge the adjacent pair with the lowest rank until no pair is in the
// vocabulary. It returns the piece boundaries, where piece k covers
// input[bounds[k]:bounds[k+1]], so a caller can count them or resolve them to
// IDs without the merge loop having to build either.
//
// Boundaries are offsets rather than byte slices, so no candidate pair is ever
// copied to be looked up, and only the two ranks a merge invalidates are
// recomputed.
func (b *BPE) mergeByRanks(input []byte) []int {
	bounds := make([]int, len(input)+1)
	for i := range bounds {
		bounds[i] = i
	}
	if len(input) < 2 {
		return bounds
	}

	// rankAt reports the rank of the piece formed by joining the pieces that
	// start at i and i+1.
	rankAt := func(i int) int {
		if i+2 >= len(bounds) {
			return unmergeable
		}
		if rank, ok := b.vocab.Encode(input[bounds[i]:bounds[i+2]]); ok {
			return rank
		}
		return unmergeable
	}

	ranks := make([]int, len(bounds))
	for i := range ranks {
		ranks[i] = rankAt(i)
	}

	for {
		best, bestIdx := unmergeable, -1
		for i := 0; i < len(bounds)-1; i++ {
			if ranks[i] < best {
				best, bestIdx = ranks[i], i
			}
		}
		if bestIdx < 0 {
			break
		}

		// Dropping the boundary after bestIdx joins that pair into one piece.
		bounds = append(bounds[:bestIdx+1], bounds[bestIdx+2:]...)
		ranks = append(ranks[:bestIdx+1], ranks[bestIdx+2:]...)

		// Only the merged piece and the one before it changed neighbours.
		ranks[bestIdx] = rankAt(bestIdx)
		if bestIdx > 0 {
			ranks[bestIdx-1] = rankAt(bestIdx - 1)
		}
	}
	return bounds
}

// encodeWithRanks uses token rank as merge priority (tiktoken style)
func (b *BPE) encodeWithRanks(input []byte) []int {
	bounds := b.mergeByRanks(input)
	result := make([]int, 0, len(bounds)-1)
	for i := 0; i+1 < len(bounds); i++ {
		if rank, ok := b.vocab.Encode(input[bounds[i]:bounds[i+1]]); ok {
			result = append(result, rank)
		}
	}
	return result
}

// bpeNode is an element in the linked list of tokens during BPE merging.
type bpeNode struct {
	id      int
	prev    *bpeNode
	next    *bpeNode
	removed bool
}

// mergePair is a candidate merge stored in the priority queue.
type mergePair struct {
	priority int
	left     *bpeNode
	right    *bpeNode
	leftID   int // id of left node when this entry was created
	rightID  int // id of right node when this entry was created
}

// mergeHeap implements heap.Interface for mergePair, ordered by priority (lower = better).
type mergeHeap []mergePair

func (h mergeHeap) Len() int           { return len(h) }
func (h mergeHeap) Less(i, j int) bool { return h[i].priority < h[j].priority }
func (h mergeHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *mergeHeap) Push(x any)        { *h = append(*h, x.(mergePair)) }
func (h *mergeHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// encodeWithMerges uses explicit merge rules (HuggingFace/SentencePiece style).
// Uses a linked list + min-heap for O(n log n) instead of O(n²).
func (b *BPE) encodeWithMerges(input []byte) []int {
	// Build initial linked list of character tokens
	text := string(input)
	var head, tail *bpeNode
	nodeCount := 0

	appendNode := func(id int) {
		node := &bpeNode{id: id}
		if tail != nil {
			tail.next = node
			node.prev = tail
		} else {
			head = node
		}
		tail = node
		nodeCount++
	}

	for _, r := range text {
		charBytes := []byte(string(r))
		if id, ok := b.vocab.Encode(charBytes); ok {
			appendNode(id)
		} else {
			for _, by := range charBytes {
				if id, ok := b.vocab.Encode([]byte{by}); ok {
					appendNode(id)
				}
			}
		}
	}

	if nodeCount <= 1 {
		result := make([]int, 0, nodeCount)
		for n := head; n != nil; n = n.next {
			result = append(result, n.id)
		}
		return result
	}

	// Build priority queue of all mergeable adjacent pairs
	h := &mergeHeap{}
	for n := head; n != nil && n.next != nil; n = n.next {
		if priority, ok := b.vocab.MergePriority(n.id, n.next.id); ok {
			*h = append(*h, mergePair{
				priority: priority,
				left:     n,
				right:    n.next,
				leftID:   n.id,
				rightID:  n.next.id,
			})
		}
	}
	heap.Init(h)

	// Process merges
	for h.Len() > 0 {
		best := heap.Pop(h).(mergePair)

		// Skip stale entries: node removed, id changed, or no longer adjacent
		if best.left.removed || best.right.removed ||
			best.left.id != best.leftID || best.right.id != best.rightID ||
			best.left.next != best.right {
			continue
		}

		// Compute merged token
		token1, _ := b.vocab.Decode(best.left.id)
		token2, _ := b.vocab.Decode(best.right.id)
		merged := append(token1, token2...)
		mergedID, ok := b.vocab.Encode(merged)
		if !ok {
			continue
		}

		// Apply merge: update left node, remove right node from list
		best.left.id = mergedID
		best.left.next = best.right.next
		if best.right.next != nil {
			best.right.next.prev = best.left
		}
		best.right.removed = true
		nodeCount--

		// Enqueue new neighbor pairs
		if best.left.prev != nil {
			if priority, ok := b.vocab.MergePriority(best.left.prev.id, best.left.id); ok {
				heap.Push(h, mergePair{
					priority: priority,
					left:     best.left.prev,
					right:    best.left,
					leftID:   best.left.prev.id,
					rightID:  best.left.id,
				})
			}
		}
		if best.left.next != nil {
			if priority, ok := b.vocab.MergePriority(best.left.id, best.left.next.id); ok {
				heap.Push(h, mergePair{
					priority: priority,
					left:     best.left,
					right:    best.left.next,
					leftID:   best.left.id,
					rightID:  best.left.next.id,
				})
			}
		}
	}

	// Collect results
	result := make([]int, 0, nodeCount)
	for n := head; n != nil; n = n.next {
		result = append(result, n.id)
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
	return len(b.mergeByRanks(input)) - 1
}

func (b *BPE) countWithMerges(input []byte) int {
	return len(b.encodeWithMerges(input))
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
