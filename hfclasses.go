package tokenizer

import (
	_ "embed"
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

//go:embed hfclasses.ranges
var hfClassesText string

// hfClass holds the code points HuggingFace tokenizers matches with one regex class.
type hfClass struct{ table *unicode.RangeTable }

func (c hfClass) has(r rune) bool { return unicode.Is(c.table, r) }

var hfL, hfM, hfN, hfP, hfS, hfSpace = loadHFClasses()

func loadHFClasses() (l, m, n, p, s, space hfClass) {
	byName := map[string]*hfClass{"L": &l, "M": &m, "N": &n, "P": &p, "S": &s, "Space": &space}
	for _, line := range strings.Split(strings.TrimSpace(hfClassesText), "\n") {
		fields := strings.Fields(line)
		dst, ok := byName[fields[0]]
		if !ok {
			panic(fmt.Sprintf("hfclasses.ranges: unknown class %q", fields[0]))
		}
		dst.table = parseRanges(fields[0], fields[1:])
		delete(byName, fields[0])
	}
	for name := range byName {
		panic(fmt.Sprintf("hfclasses.ranges: class %s is missing", name))
	}
	return
}

func parseRanges(class string, fields []string) *unicode.RangeTable {
	t := &unicode.RangeTable{}
	for _, f := range fields {
		loText, hiText, isRange := strings.Cut(f, "-")
		if !isRange {
			hiText = loText
		}
		lo, errLo := strconv.ParseUint(loText, 16, 32)
		hi, errHi := strconv.ParseUint(hiText, 16, 32)
		if errLo != nil || errHi != nil || hi < lo {
			panic(fmt.Sprintf("hfclasses.ranges: class %s: bad range %q", class, f))
		}
		if hi <= 0xFFFF {
			t.R16 = append(t.R16, unicode.Range16{Lo: uint16(lo), Hi: uint16(hi), Stride: 1})
			if hi <= unicode.MaxLatin1 {
				t.LatinOffset++
			}
			continue
		}
		if lo <= 0xFFFF {
			t.R16 = append(t.R16, unicode.Range16{Lo: uint16(lo), Hi: 0xFFFF, Stride: 1})
			lo = 0x10000
		}
		t.R32 = append(t.R32, unicode.Range32{Lo: uint32(lo), Hi: uint32(hi), Stride: 1})
	}
	return t
}
