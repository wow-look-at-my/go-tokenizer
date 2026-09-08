package main

import (
	"strings"
	"testing"
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
		got := fmtTime(tc.ns)
		if got != tc.want {
			t.Errorf("fmtTime(%v) = %q, want %q", tc.ns, got, tc.want)
		}
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
		got := escXML(tc.in)
		if got != tc.want {
			t.Errorf("escXML(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestRenderSVG(t *testing.T) {
	results := []result{
		{"Test1", 1000},
		{"Test2", 2000},
	}
	svg := renderSVG(results)

	if !strings.HasPrefix(svg, "<svg") {
		t.Error("SVG should start with <svg")
	}
	if !strings.HasSuffix(svg, "</svg>\n") {
		t.Error("SVG should end with </svg>")
	}
	if !strings.Contains(svg, "Test1") {
		t.Error("SVG should contain Test1")
	}
	if !strings.Contains(svg, "Test2") {
		t.Error("SVG should contain Test2")
	}
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
		if m == nil {
			t.Errorf("benchLine should match %q", tc.line)
			continue
		}
		if m[1] != tc.name {
			t.Errorf("name = %q, want %q", m[1], tc.name)
		}
		if m[2] != tc.ns {
			t.Errorf("ns = %q, want %q", m[2], tc.ns)
		}
	}
}
