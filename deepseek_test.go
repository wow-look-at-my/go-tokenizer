package tokenizer

import (
	"bufio"
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// deepseekGolden holds reference IDs from HuggingFace's Rust tokenizers.
const deepseekGolden = "testdata/deepseek-golden.jsonl"

type deepseekRow struct {
	ID       string `json:"id"`
	Category string `json:"category"`
	Text     string `json:"text"`
	RefIDs   []int  `json:"refIds"`
	APICount *int   `json:"apiCount"`
}

func loadDeepseekGolden(t *testing.T) []deepseekRow {
	t.Helper()
	f, err := os.Open(deepseekGolden)
	require.NoError(t, err)
	defer f.Close()

	var rows []deepseekRow
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 1<<24)
	for sc.Scan() {
		var row deepseekRow
		require.NoError(t, json.Unmarshal(sc.Bytes(), &row))
		rows = append(rows, row)
	}
	require.NoError(t, sc.Err())
	require.NotEmpty(t, rows)
	return rows
}

func TestDeepseekMatchesReference(t *testing.T) {
	tok, err := NewWithEncoding("deepseek")
	require.NoError(t, err)
	assert.Equal(t, 129280, tok.VocabSize())

	mismatches := 0
	for _, row := range loadDeepseekGolden(t) {
		ids, err := tok.Encode(row.Text)
		require.NoError(t, err)
		if ids == nil {
			ids = []int{}
		}
		if !assert.Equal(t, row.RefIDs, ids, "row %s (%s): %q", row.ID, row.Category, row.Text) {
			mismatches++
			require.Less(t, mismatches, 10)

			continue
		}

		n, err := tok.CountTokens(row.Text)
		require.NoError(t, err)
		assert.Equal(t, len(row.RefIDs), n, "count for row %s", row.ID)

		text, err := tok.Decode(ids)
		require.NoError(t, err)
		assert.Equal(t, row.Text, text, "round trip for row %s", row.ID)
	}
}

func TestDeepseekReferenceAgreesWithAPI(t *testing.T) {
	measured := 0
	for _, row := range loadDeepseekGolden(t) {
		if row.APICount == nil {
			continue
		}
		measured++
		assert.Equal(t, *row.APICount, len(row.RefIDs), "row %s: %q", row.ID, row.Text)
	}
	assert.NotZero(t, measured, "the fixture holds no apiCount rows")
}

func TestDeepseekSpecialTokens(t *testing.T) {
	tok, err := NewWithEncoding("deepseek")
	require.NoError(t, err)

	ids, err := tok.Encode("<｜User｜>hi<｜Assistant｜><think>")
	require.NoError(t, err)
	assert.Equal(t, []int{128803, 6366, 128804, 128821}, ids)

	text, err := tok.Decode([]int{0})
	require.NoError(t, err)
	assert.Equal(t, "<｜begin▁of▁sentence｜>", text)
}
