// Writes the Unicode classes that HuggingFace tokenizers uses for \p{L}, \p{M}, \p{N}, \p{P}, \p{S} and \s, as hex code point ranges.
const { Tokenizer } = require('tokenizers');
const fs = require('node:fs');

const [tokenizerPath, outPath] = process.argv.slice(2);
const base = JSON.parse(fs.readFileSync(tokenizerPath, 'utf8'));
const classes = { L: '\\p{L}', M: '\\p{M}', N: '\\p{N}', P: '\\p{P}', S: '\\p{S}', Space: '\\s' };

const scalars = [];
for (let cp = 0; cp <= 0x10ffff; cp++) if (cp < 0xd800 || cp > 0xdfff) scalars.push(cp);

const lines = [];
for (const [name, pattern] of Object.entries(classes)) {
	// An inverted Split with Removed behavior keeps only the code points that match.
	const cfg = { ...base, normalizer: null, pre_tokenizer: { type: 'Split', pattern: { Regex: pattern }, behavior: 'Removed', invert: true } };
	const pre = Tokenizer.fromString(JSON.stringify(cfg)).getPreTokenizer();
	const members = new Set();
	for (let i = 0; i < scalars.length; i += 4096) {
		const s = String.fromCodePoint(...scalars.slice(i, i + 4096));
		for (const [piece] of pre.preTokenizeString(s)) for (const ch of piece) members.add(ch.codePointAt(0));
	}
	const ranges = [];
	for (const cp of scalars) {
		if (!members.has(cp)) continue;
		const last = ranges[ranges.length - 1];
		if (last && last[1] === cp - 1) last[1] = cp;
		else ranges.push([cp, cp]);
	}
	const hex = (n) => n.toString(16);
	lines.push([name, ...ranges.map(([lo, hi]) => (lo === hi ? hex(lo) : `${hex(lo)}-${hex(hi)}`))].join(' '));
	console.error(`${name}: ${members.size} code points in ${ranges.length} ranges`);
}
fs.writeFileSync(outPath, lines.join('\n') + '\n');
