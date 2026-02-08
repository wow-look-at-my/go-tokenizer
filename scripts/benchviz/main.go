// benchviz reads Go benchmark output from stdin and writes an SVG
// horizontal bar chart to stdout.
//
// Usage:
//
//	go test -bench=. -benchmem ./... | go run ./scripts/benchviz > bench.svg
package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"regexp"
	"strconv"
	"strings"
)

type result struct {
	name string
	ns   float64
}

var benchLine = regexp.MustCompile(
	`^(Benchmark\S+)\s+\d+\s+([\d.]+)\s+ns/op`,
)

func main() {
	var results []result
	sc := bufio.NewScanner(os.Stdin)
	for sc.Scan() {
		m := benchLine.FindStringSubmatch(sc.Text())
		if m == nil {
			continue
		}
		name := m[1]
		name = strings.TrimPrefix(name, "Benchmark")
		if i := strings.LastIndex(name, "-"); i > 0 {
			if _, err := strconv.Atoi(name[i+1:]); err == nil {
				name = name[:i]
			}
		}
		ns, _ := strconv.ParseFloat(m[2], 64)
		results = append(results, result{name, ns})
	}
	if len(results) == 0 {
		fmt.Fprintln(os.Stderr, "benchviz: no benchmark results found on stdin")
		os.Exit(1)
	}

	fmt.Print(renderSVG(results))
}

func fmtTime(ns float64) string {
	switch {
	case ns < 1_000:
		return fmt.Sprintf("%.0f ns", ns)
	case ns < 1_000_000:
		return fmt.Sprintf("%.2f µs", ns/1_000)
	case ns < 1_000_000_000:
		return fmt.Sprintf("%.2f ms", ns/1_000_000)
	case ns < 60_000_000_000:
		return fmt.Sprintf("%.2f s", ns/1_000_000_000)
	default:
		return fmt.Sprintf("%.1f min", ns/60_000_000_000)
	}
}

func renderSVG(results []result) string {
	const (
		barHeight  = 22
		barGap     = 6
		labelWidth = 340
		barMax     = 400
		rightPad   = 100
		topPad     = 40
		bottomPad  = 20
		fontSize   = 12
	)

	n := len(results)
	totalWidth := labelWidth + barMax + rightPad
	totalHeight := topPad + n*(barHeight+barGap) + bottomPad

	var maxNS float64
	for _, r := range results {
		if r.ns > maxNS {
			maxNS = r.ns
		}
	}
	logMax := math.Log10(maxNS + 1)

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="monospace" font-size="%d">`, totalWidth, totalHeight, fontSize))
	sb.WriteString("\n")
	sb.WriteString(`<style>`)
	sb.WriteString(`rect.bar{rx:3;ry:3} `)
	sb.WriteString(`text.label{fill:#333;text-anchor:end;dominant-baseline:middle} `)
	sb.WriteString(`text.value{fill:#555;dominant-baseline:middle} `)
	sb.WriteString(`text.title{font-size:16px;font-weight:bold;fill:#222} `)
	sb.WriteString("</style>\n")
	sb.WriteString(fmt.Sprintf(`<text x="%d" y="24" class="title">Benchmark Results (log scale, time/op)</text>`+"\n", totalWidth/2-120))

	palette := []string{"#4285f4", "#ea4335", "#fbbc04", "#34a853", "#ff6d01", "#46bdc6", "#7b61ff", "#f538a0"}

	for i, r := range results {
		y := topPad + i*(barHeight+barGap)
		logVal := math.Log10(r.ns + 1)
		barW := int(logVal / logMax * barMax)
		if barW < 2 {
			barW = 2
		}
		color := palette[i%len(palette)]

		sb.WriteString(fmt.Sprintf(`<text x="%d" y="%d" class="label">%s</text>`, labelWidth-8, y+barHeight/2, escXML(r.name)))
		sb.WriteString("\n")
		sb.WriteString(fmt.Sprintf(`<rect class="bar" x="%d" y="%d" width="%d" height="%d" fill="%s"/>`, labelWidth, y, barW, barHeight, color))
		sb.WriteString("\n")
		sb.WriteString(fmt.Sprintf(`<text x="%d" y="%d" class="value"> %s</text>`, labelWidth+barW+4, y+barHeight/2, fmtTime(r.ns)))
		sb.WriteString("\n")
	}

	sb.WriteString("</svg>\n")
	return sb.String()
}

func escXML(s string) string {
	s = strings.ReplaceAll(s, "&", "&amp;")
	s = strings.ReplaceAll(s, "<", "&lt;")
	s = strings.ReplaceAll(s, ">", "&gt;")
	return s
}
