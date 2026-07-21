#!/usr/bin/awk -f
# Format Go benchmark output as a human-readable markdown table.
# Usage: go test -bench=. -benchmem ./... | awk -f fmt-bench.awk

/^goos:/   { goos   = $2; next }
/^goarch:/ { goarch = $2; next }
/^cpu:/    { sub(/^cpu: */, ""); cpu = $0; next }

/^Benchmark/ {
    if (!hdr) {
        printf "_Platform: %s/%s &mdash; %s_\n\n", goos, goarch, cpu
        print  "| Benchmark | Iters | Time/op | Throughput | Mem/op | Allocs/op |"
        print  "|:---|---:|---:|---:|---:|---:|"
        hdr = 1
    }

    name = $1
    sub(/^Benchmark/, "", name)
    sub(/-[0-9]+$/, "", name)

    iters = $2
    time_ns = ""; tp = ""; mem = ""; allocs = ""
    for (i = 3; i <= NF; i++) {
        if ($i == "ns/op")      time_ns = $(i-1)
        if ($i == "MB/s")       tp      = $(i-1)
        if ($i == "B/op")       mem     = $(i-1)
        if ($i == "allocs/op")  allocs  = $(i-1)
    }

    printf "| %s | %s | %s | %s | %s | %s |\n",
        name,
        fmt_count(iters),
        fmt_time(time_ns),
        (tp != "") ? tp " MB/s" : "-",
        fmt_bytes(mem),
        fmt_count(allocs)
}

function fmt_time(ns,    v) {
    if (ns == "") return "-"
    v = ns + 0
    if (v < 1000)          return sprintf("%.0f ns",  v)
    if (v < 1000000)       return sprintf("%.2f \302\265s", v / 1000)
    if (v < 1000000000)    return sprintf("%.2f ms",  v / 1000000)
    if (v < 60000000000)   return sprintf("%.2f s",   v / 1000000000)
    return sprintf("%.1f min", v / 60000000000)
}

function fmt_bytes(b,    v) {
    if (b == "") return "-"
    v = b + 0
    if (v < 1024)       return sprintf("%d B",     v)
    if (v < 1048576)    return sprintf("%.1f KB",  v / 1024)
    if (v < 1073741824) return sprintf("%.1f MB",  v / 1048576)
    return sprintf("%.2f GB", v / 1073741824)
}

function fmt_count(n,    v) {
    if (n == "") return "-"
    v = n + 0
    if (v < 1000)    return sprintf("%d", v)
    if (v < 1000000) return sprintf("%.1fk", v / 1000)
    return sprintf("%.1fM", v / 1000000)
}
