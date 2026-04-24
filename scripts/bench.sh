#!/usr/bin/env bash
set -euo pipefail

# Run all qjl-sketch benchmarks and collect results into a report.
#
# Usage:
#   ./scripts/bench.sh              # CPU benchmarks only
#   ./scripts/bench.sh --gpu        # include GPU benchmarks
#   ./scripts/bench.sh --gpu --save # save report to artifacts/
#
# Requires: jq

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

command -v jq &>/dev/null || { echo "error: jq is required"; exit 1; }

GPU=false
SAVE=false
for arg in "$@"; do
    case "$arg" in
        --gpu) GPU=true ;;
        --save) SAVE=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
REPORT_DIR="artifacts/bench-$TIMESTAMP"
CRITERION_DIR="target/criterion"

echo "=== qjl-sketch benchmarks ==="
echo "Date: $(date)"
echo "Rust: $(rustc --version)"
echo "GPU:  $GPU"
echo ""

# Collect system info
SYSTEM_INFO="$(uname -ms)"
if command -v sysctl &>/dev/null; then
    CPU_INFO="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'unknown')"
else
    CPU_INFO="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo 'unknown')"
fi

# Format nanoseconds to human-readable
format_ns() {
    local ns="${1%.*}"  # truncate to integer
    if [ "$ns" -ge 1000000000 ]; then
        echo "$1 1000000000" | awk '{printf "%.2f s", $1/$2}'
    elif [ "$ns" -ge 1000000 ]; then
        echo "$1 1000000" | awk '{printf "%.2f ms", $1/$2}'
    elif [ "$ns" -ge 1000 ]; then
        echo "$1 1000" | awk '{printf "%.2f µs", $1/$2}'
    else
        echo "$1" | awk '{printf "%.0f ns", $1}'
    fi
}

# Read mean point_estimate from criterion JSON via jq
read_estimate() {
    local json="$CRITERION_DIR/$1/new/estimates.json"
    if [ -f "$json" ]; then
        jq -r '.mean.point_estimate' "$json"
    else
        echo "0"
    fi
}

# Print one benchmark result
print_result() {
    local name="$1"
    local path="$2"
    local ns
    ns=$(read_estimate "$path")
    if [ "$ns" != "0" ] && [ "$ns" != "null" ]; then
        printf "  %-40s %s\n" "$name" "$(format_ns "$ns")"
    else
        printf "  %-40s %s\n" "$name" "(no data)"
    fi
}

# Run CPU benchmarks
echo "--- Running CPU benchmarks ---"
echo ""
cargo bench --bench score 2>/dev/null
cargo bench --bench compress 2>/dev/null
cargo bench --bench store 2>/dev/null
echo ""

echo "--- CPU Results ---"
echo ""
echo "  Score:"
print_result "score 10 pages (32 vec each)" "score_latency/pages/10"
print_result "score 100 pages" "score_latency/pages/100"
print_result "score 1000 pages" "score_latency/pages/1000"
print_result "score 1 page (64 vec)" "score_single_page_64vec"
echo ""
echo "  Compression:"
print_result "key quantize 32 vec" "key_quantize/vectors/32"
print_result "key quantize 128 vec" "key_quantize/vectors/128"
print_result "key quantize 512 vec" "key_quantize/vectors/512"
print_result "value quantize 256 elem (4-bit)" "value_quantize/elements_4bit/256"
print_result "value quantize 4096 elem (4-bit)" "value_quantize/elements_4bit/4096"
print_result "sketch creation 64×128" "sketch_creation/dim/64x128"
print_result "sketch creation 128×256" "sketch_creation/dim/128x256"
print_result "sketch creation 128×512" "sketch_creation/dim/128x512"
echo ""
echo "  Store I/O:"
print_result "cold start (100 pages)" "cold_start_100_pages"
print_result "append 1 page" "append_single_page"
print_result "get_page lookup" "get_page_from_100"
echo ""

# Run GPU benchmarks if requested
if [ "$GPU" = true ]; then
    echo "--- GPU benchmarks ---"
    echo ""

    echo "[GPU] Benchmarks (auto dispatch, GPU_MIN_BATCH=5000)..."
    cargo bench --bench gpu_score --features gpu 2>/dev/null
    echo ""

    echo "[GPU] Benchmarks (forced GPU, QJL_GPU_MIN_BATCH=0)..."
    QJL_GPU_MIN_BATCH=0 cargo bench --bench gpu_score --features gpu 2>/dev/null
    echo ""

    echo "--- GPU Results (32 vec/page) ---"
    echo ""
    echo "  CPU baseline (sketch.score per page, never GPU):"
    echo "    d=64, s=128:"
    print_result "  10 pages" "cpu_per_page_d64/pages/10"
    print_result "  100 pages" "cpu_per_page_d64/pages/100"
    print_result "  1000 pages" "cpu_per_page_d64/pages/1000"
    print_result "  10000 pages" "cpu_per_page_d64/pages/10000"
    echo "    d=128, s=256:"
    print_result "  10 pages" "cpu_per_page_d128/pages/10"
    print_result "  100 pages" "cpu_per_page_d128/pages/100"
    print_result "  1000 pages" "cpu_per_page_d128/pages/1000"
    print_result "  10000 pages" "cpu_per_page_d128/pages/10000"
    echo ""
    echo "  score_all_pages (batched GPU if >= GPU_MIN_BATCH):"
    echo "    d=64, s=128:"
    print_result "  10 pages (320 vec)" "score_all_pages_d64/pages/10"
    print_result "  100 pages (3.2K vec)" "score_all_pages_d64/pages/100"
    print_result "  1000 pages (32K vec)" "score_all_pages_d64/pages/1000"
    print_result "  10000 pages (320K vec)" "score_all_pages_d64/pages/10000"
    echo "    d=128, s=256:"
    print_result "  10 pages (320 vec)" "score_all_pages_d128/pages/10"
    print_result "  100 pages (3.2K vec)" "score_all_pages_d128/pages/100"
    print_result "  1000 pages (32K vec)" "score_all_pages_d128/pages/1000"
    print_result "  10000 pages (320K vec)" "score_all_pages_d128/pages/10000"
    echo ""
fi

# Save report if requested
if [ "$SAVE" = true ]; then
    mkdir -p "$REPORT_DIR"

    # Copy criterion HTML reports
    if [ -d "$CRITERION_DIR" ]; then
        cp -r "$CRITERION_DIR" "$REPORT_DIR/criterion"
    fi

    # Write summary
    {
        echo "qjl-sketch benchmark report"
        echo "============================"
        echo "Date:    $(date)"
        echo "System:  $SYSTEM_INFO"
        echo "CPU:     $CPU_INFO"
        echo "Rust:    $(rustc --version)"
        echo "Version: $(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)"
        echo "GPU:     $GPU"
        echo ""
        echo "Criterion HTML reports: criterion/report/index.html"
    } > "$REPORT_DIR/summary.txt"

    echo "Report saved to $REPORT_DIR/"
    echo "Open $REPORT_DIR/criterion/report/index.html for interactive charts."
fi

echo "=== done ==="
