#!/bin/bash
# Chunked GDN coopmat benchmark
# Usage: ./scripts/bench-gdn-chunked.sh <model.gguf> [output_file]

set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf> [output_file]}"
OUT="${2:-gdn-chunked-results.md}"
LOG="${OUT%.md}.log"
BENCH="./build/bin/llama-bench"

if [ ! -f "$BENCH" ]; then
    echo "ERROR: llama-bench not found. Build first:"
    echo "  cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build build --target llama-bench -j\$(nproc)"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    exit 1
fi

echo "Checking model + GPU..."
PROBE=$($BENCH -m "$MODEL" -ngl 99 -fa 1 -n 0 -p 1 -v 2>&1) || {
    echo "ERROR: llama-bench failed to load model. Full output:"
    echo "$PROBE"
    echo "$PROBE" > "$LOG"
    exit 1
}

GPU_LINE=$(echo "$PROBE" | grep "ggml_vulkan: 0 =" | head -1 || echo "unknown")
GPU_NAME=$(echo "$GPU_LINE" | sed 's/.*0 = //' || echo "unknown")
BUILD=$(echo "$PROBE" | grep "^build:" || echo "unknown")
COOPMAT="no"
echo "$GPU_LINE" | grep -q "KHR_coopmat" && COOPMAT="yes (KHR_coopmat)"
GDN_MODE="not detected"
echo "$PROBE" | grep -q "chunked) enabled" && GDN_MODE="chunked (coopmat)"
echo "$PROBE" | grep -q "autoregressive) enabled" && [ "$GDN_MODE" = "not detected" ] && GDN_MODE="autoregressive"
echo "$PROBE" | grep -q "chunked) enabled" && echo "$PROBE" | grep -q "autoregressive) enabled" && GDN_MODE="both (auto + chunked)"

{
    echo "# Chunked GDN Coopmat Benchmark"
    echo ""
    echo "**GPU:** ${GPU_NAME}"
    echo "**Coopmat:** ${COOPMAT}"
    echo "**GDN mode:** ${GDN_MODE}"
    echo "**Model:** $(basename "$MODEL")"
    echo "**Date:** $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "**Build:** $BUILD"
    echo "**OS:** $(uname -srm)"
    echo "**RAM:** $(free -h | awk '/Mem:/{print $2}') total"
    echo ""
} > "$OUT"

if [ "$GDN_MODE" = "not detected" ]; then
    echo "WARNING: GDN not detected for this model. Results may not show GDN profiling data."
fi

echo "Running throughput benchmark (PP-512/1024/2048 + TG-128)..."
if ! RESULT=$($BENCH -m "$MODEL" -ngl 99 -fa 1 -n 128 -p 512,1024,2048 --output md 2>&1); then
    echo "ERROR: Benchmark failed. See $LOG for details."
    echo "$RESULT" > "$LOG"
    echo "" >> "$OUT"
    echo "## ERROR: Benchmark failed" >> "$OUT"
    echo '```' >> "$OUT"
    echo "$RESULT" | tail -30 >> "$OUT"
    echo '```' >> "$OUT"
    cat "$OUT"
    exit 1
fi

{
    echo "## Throughput (default ubatch)"
    echo ""
    echo "$RESULT" | grep -E "^\|"
    echo ""
} >> "$OUT"

echo "Running n_ubatch sweep (PP-2048)..."
{
    echo "## Throughput by n_ubatch (PP-2048)"
    echo ""
} >> "$OUT"

for UB in 256 512 1024 2048; do
    echo "  ubatch=$UB..."
    UB_RESULT=$($BENCH -m "$MODEL" -ngl 99 -fa 1 -n 0 -p 2048 -ub $UB --output md 2>&1) || true
    UB_LINE=$(echo "$UB_RESULT" | grep "pp2048" | head -1)
    if [ -n "$UB_LINE" ]; then
        if [ "$UB" = "256" ]; then
            echo "$UB_RESULT" | grep -E "^\| (model|---)" | head -2 >> "$OUT"
        fi
        echo "$UB_LINE" >> "$OUT"
    fi
done
echo "" >> "$OUT"

echo "Running GDN kernel profiling (PP-512)..."
PROF=$(GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=9999 $BENCH -m "$MODEL" -ngl 99 -fa 1 -n 0 -p 512 2>&1 | grep "GATED_DELTA" | head -5)

if [ -n "$PROF" ]; then
    {
        echo "## GDN Kernel Timing (PP-512)"
        echo ""
        echo '```'
        echo "$PROF"
        echo '```'
        echo ""
    } >> "$OUT"
else
    echo "*No GDN profiling data — model may not use GATED_DELTA_NET.*" >> "$OUT"
    echo "" >> "$OUT"
fi

echo ""
echo "Done. Results saved to: $OUT"
echo "---------------------------------------"
cat "$OUT"
