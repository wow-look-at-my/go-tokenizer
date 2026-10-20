package main

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFmtTime(t *testing.T) {
	tests := []struct {
		ns   float64
		want string
	}{
		{500, "500 ns"},
		{1500, "1.50 µs"},
		{1500000, "1.50 ms"},
		{1500000000, "1.50 s"},
		{90000000000, "1.5 min"},
	}
	for _, tc := range tests {
		assert.Equal(t, tc.want, fmtTime(tc.ns))
	}
}

func TestEscXML(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"hello", "hello"},
		{"a < b", "a &lt; b"},
		{"a > b", "a &gt; b"},
		{"a & b", "a &amp; b"},
		{"<>&", "&lt;&gt;&amp;"},
	}
	for _, tc := range tests {
		assert.Equal(t, tc.want, escXML(tc.in))
	}
}

func TestRenderSVG(t *testing.T) {
	results := []result{
		{"Test1", 1000},
		{"Test2", 2000},
	}
	svg := renderSVG(results)

	assert.True(t, strings.HasPrefix(svg, "<svg"), "SVG should start with <svg")
	assert.True(t, strings.HasSuffix(svg, "</svg>\n"), "SVG should end with </svg>")
	assert.Contains(t, svg, "Test1")
	assert.Contains(t, svg, "Test2")
}

func TestBenchLineRegex(t *testing.T) {
	tests := []struct {
		line string
		name string
		ns   string
	}{
		{"BenchmarkFoo-8    1000    500.00 ns/op", "BenchmarkFoo-8", "500.00"},
		{"BenchmarkBar/sub-8    500    1234.56 ns/op", "BenchmarkBar/sub-8", "1234.56"},
	}
	for _, tc := range tests {
		m := benchLine.FindStringSubmatch(tc.line)
		require.NotNil(t, m, "benchLine should match %q", tc.line)
		assert.Equal(t, tc.name, m[1])
		assert.Equal(t, tc.ns, m[2])
	}
}
