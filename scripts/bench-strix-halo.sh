#!/usr/bin/env bash
#
# bench-strix-halo.sh — Benchmark llama.cpp on AMD Strix Halo (Ryzen AI Max+ 395)
#
# Runs llama-bench across GPU-primary, CPU-primary, and hybrid configurations
# to find optimal settings for your specific SKU and model.
#
# Usage:
#   ./scripts/bench-strix-halo.sh <model.gguf> [output_dir]
#
# Prerequisites:
#   - TTM limits configured (see CLAUDE.md or skill docs)
#   - ROCm 6.1+ installed (for HIP backend)
#   - Built with: cmake -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_GRAPHS=ON
#
set -euo pipefail

MODEL="${1:?Usage: $0 <model.gguf> [output_dir]}"
OUTDIR="${2:-bench_results/$(date +%Y%m%d_%H%M%S)}"
BENCH="./build/bin/llama-bench"
REPS=3

if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model file not found: $MODEL" >&2
    exit 1
fi

if [[ ! -x "$BENCH" ]]; then
    echo "Error: llama-bench not found at $BENCH — build first" >&2
    exit 1
fi

mkdir -p "$OUTDIR"

echo "=== Strix Halo Benchmark Suite ==="
echo "Model:  $MODEL"
echo "Output: $OUTDIR"
echo "Reps:   $REPS"
echo ""

# Critical for prompt processing performance on AMD
export ROCBLAS_USE_HIPBLASLT=1

# Show detected devices
echo "--- Detected devices ---"
$BENCH --list-devices 2>&1 | tee "$OUTDIR/devices.txt"
echo ""

# ── Test 1: GPU-primary (full offload) ──────────────────────────────
echo "--- Test 1/4: GPU-primary (ngl=99) ---"
$BENCH -m "$MODEL" \
    -ngl 99 \
    --no-mmap \
    -fa 1 \
    -ctk q8_0 -ctv q8_0 \
    -t 2,4 \
    -b 2048 -ub 512 \
    -p 128,512,2048 -n 128 \
    -r "$REPS" \
    -o csv 2>/dev/null | tee "$OUTDIR/gpu_primary.csv"
echo ""

# ── Test 2: CPU-primary (no GPU) ────────────────────────────────────
echo "--- Test 2/4: CPU-primary (ngl=0) ---"
$BENCH -m "$MODEL" \
    -ngl 0 \
    -t 4,8,16 \
    -b 2048 -ub 512 \
    -p 128,512 -n 128 \
    -r "$REPS" \
    -o csv 2>/dev/null | tee "$OUTDIR/cpu_primary.csv"
echo ""

# ── Test 3: Hybrid (sweep ngl) ──────────────────────────────────────
echo "--- Test 3/4: Hybrid (ngl sweep) ---"
$BENCH -m "$MODEL" \
    -ngl 0,10,20,40,99 \
    --no-mmap \
    -fa 1 \
    -t 4 \
    -b 2048 -ub 512 \
    -p 512 -n 128 \
    -r "$REPS" \
    -o csv 2>/dev/null | tee "$OUTDIR/hybrid_sweep.csv"
echo ""

# ── Test 4: KV cache quantization comparison ────────────────────────
echo "--- Test 4/4: KV cache quantization ---"
for KV in f16 q8_0 q4_0; do
    echo "  KV type: $KV"
    $BENCH -m "$MODEL" \
        -ngl 99 \
        --no-mmap \
        -fa 1 \
        -ctk "$KV" -ctv "$KV" \
        -t 4 \
        -p 512,2048 -n 128 \
        -r "$REPS" \
        -o csv 2>/dev/null
done | tee "$OUTDIR/kv_cache.csv"
echo ""

# ── Summary ─────────────────────────────────────────────────────────
echo "=== Benchmark Complete ==="
echo "Results saved to: $OUTDIR/"
echo ""
echo "Quick analysis tips:"
echo "  - Compare pp (prompt processing) across GPU vs CPU"
echo "  - GPU pp512 should be 800+ t/s for 7B Q4 (with hipBLASLt)"
echo "  - GPU tg128 should be ~48 t/s for 7B Q4"
echo "  - If pp is low, verify ROCBLAS_USE_HIPBLASLT=1 is set"
echo "  - Find optimal ngl in hybrid_sweep.csv"
echo ""
echo "To view results as markdown:"
echo "  column -t -s, $OUTDIR/gpu_primary.csv"
