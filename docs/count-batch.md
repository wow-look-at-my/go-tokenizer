# `count --batch`

`count --batch` counts many independent texts over a single process. The vocabulary loads once, rather than once per text.

Each input line is a JSON-encoded string, in JSON Lines form. Each output line is that text's token count, as a bare decimal. Counts come back in input order, and each is written as soon as it is computed.

That per-line reply makes the mode usable for bulk streaming and interactively alike. A driving process can write a few sections, read their counts, and decide what to send next, all over the same long-lived process.

```sh
$ printf '%s\n' '"Hello World"' '"The quick brown fox"' | go-tokenizer count --batch
2
4
```

Blank input lines are skipped.

A line that is not a valid JSON string aborts the run with an error naming the line number. A count that fails aborts the same way. A caller that sent N sections therefore gets N counts, or a loud failure. It never gets a silent gap, which leaves it reading the wrong count against the wrong section.

Batch mode reads standard input only. Positional arguments and `--input` are rejected rather than ignored.

Newlines inside a section arrive JSON-escaped, so a multi-line text stays a single input line and yields a single count.
