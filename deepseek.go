package tokenizer

import (
	"slices"
	"unicode/utf8"
)

// splitDeepseek runs those Split steps of deepseekSplits without a regex
// engine. It uses the HuggingFace Unicode classes, so it agrees with DeepSeek
// holds it to the regex path wherever both tables agree.
func splitDeepseek(text string, emit func(string)) {
	splitIsolatedFunc(text, scanDigits, func(piece string) {
		splitIsolatedFunc(piece, scanCJK, func(piece string) {
			splitIsolatedFunc(piece, scanDeepseekWord, emit)
		})
	})
}

// splitIsolatedFunc emits each match of scan in s and each run of text where no
// match starts, in order. scan returns the end of the match at at, or at itself.
func splitIsolatedFunc(s string, scan func(s string, at int) int, emit func(string)) {
	gap := 0
	for at := 0; at < len(s); {
		end := scan(s, at)
		if end <= at {
			_, size := utf8.DecodeRuneInString(s[at:])
			at += size
			continue
		}
		if gap < at {
			emit(s[gap:at])
		}
		emit(s[at:end])
		at, gap = end, end
	}
	if gap < len(s) {
		emit(s[gap:])
	}
}

func scanDigits(s string, at int) int {
	end := at
	for taken := 0; taken < 3 && end < len(s); taken++ {
		r, size := utf8.DecodeRuneInString(s[end:])
		if !hfN.has(r) {
			break
		}
		end += size
	}
	return end
}

// scanCJK matches a run of CJK ideographs, hiragana and katakana.
func scanCJK(s string, at int) int {
	return scanRun(s, at, func(r rune) bool {
		return ('一' <= r && r <= '龥') || ('぀' <= r && r <= 'ゟ') || ('゠' <= r && r <= 'ヿ')
	})
}

func isHFLetterOrMark(r rune) bool { return hfL.has(r) || hfM.has(r) }

func isHFPunctOrSymbol(r rune) bool { return hfP.has(r) || hfS.has(r) }

func isASCIILetter(c byte) bool { return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') }

// scanDeepseekWord follows the alternatives of the third Split pattern in the
// order the regex tries them, and reports at when none of them matches.
func scanDeepseekWord(s string, at int) int {
	r, size := utf8.DecodeRuneInString(s[at:])
	next, _ := utf8.DecodeRuneInString(s[at+size:])

	// [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
	if r < utf8.RuneSelf && isASCIIPunct(byte(r)) && at+1 < len(s) && isASCIILetter(s[at+1]) {
		end := at + 1
		for end < len(s) && isASCIILetter(s[end]) {
			end++
		}
		return end
	}

	// [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
	if r != '\r' && r != '\n' && !hfL.has(r) && !isHFPunctOrSymbol(r) && isHFLetterOrMark(next) {
		return scanRun(s, at+size, isHFLetterOrMark)
	}
	if isHFLetterOrMark(r) {
		return scanRun(s, at, isHFLetterOrMark)
	}

	// ' ?[\p{P}\p{S}]+[\r\n]*'
	symbolStart := at
	if r == ' ' && isHFPunctOrSymbol(next) {
		symbolStart = at + size
	}
	if end := scanRun(s, symbolStart, isHFPunctOrSymbol); end > symbolStart {
		for end < len(s) && (s[end] == '\r' || s[end] == '\n') {
			end++
		}
		return end
	}

	if !hfSpace.has(r) {
		return at
	}

	// \s*[\r\n]+ ends after the last line break of the whitespace run.
	runEnd, lastBreakEnd := at, -1
	for runEnd < len(s) {
		w, wSize := utf8.DecodeRuneInString(s[runEnd:])
		if !hfSpace.has(w) {
			break
		}
		runEnd += wSize
		if w == '\r' || w == '\n' {
			lastBreakEnd = runEnd
		}
	}
	if lastBreakEnd > at {
		return lastBreakEnd
	}

	// \s+(?!\S) gives the last space to a following word; \s+ takes a lone one.
	if runEnd == len(s) {
		return runEnd
	}
	if _, lastSize := utf8.DecodeLastRuneInString(s[at:runEnd]); runEnd-lastSize > at {
		return runEnd - lastSize
	}
	return runEnd
}

func isASCIIPunct(c byte) bool {
	return slices.Contains([]byte("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"), c)
}
