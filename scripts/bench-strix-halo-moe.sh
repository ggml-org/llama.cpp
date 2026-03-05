#!/usr/bin/env bash
#
# bench-strix-halo-moe.sh — MoE-specific benchmarks for AMD Strix Halo
#
# Targets Mixture-of-Experts models (Qwen3-30B-A3B, DeepSeek, etc.) with
# tests that capture MoE-specific performance characteristics: expert routing
# overhead, batch coalescing benefits, and sparse vs dense throughput.
#
# Usage:
#   ./scripts/bench-strix-halo-moe.sh <moe_model.gguf> [output_dir]
#
# Prerequisites:
#   - TTM limits configured (see CLAUDE.md or skill docs)
#   - ROCm 6.1+ installed (for HIP backend)
#   - Built with: cmake -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DGGML_HIP_GRAPHS=ON
#
set -euo pipefail

MODEL="${1:?Usage: $0 <moe_model.gguf> [output_dir]}"
OUTDIR="${2:-bench_results/moe_$(date +%Y%m%d_%H%M%S)}"
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

echo "=== Strix Halo MoE Benchmark Suite ==="
echo "Model:  $MODEL"
echo "Output: $OUTDIR"
echo "Reps:   $REPS"
echo ""

# Critical for prompt processing performance on AMD
export ROCBLAS_USE_HIPBLASLT=1

# ── Record model metadata ────────────────────────────────────────
echo "--- Model metadata ---"
{
    echo "model_path: $MODEL"
    echo "timestamp: $(date -Iseconds)"
    echo "hostname: $(hostname)"
    echo "rocm_version: $(cat /opt/rocm/.info/version 2>/dev/null || echo 'N/A')"
} | tee "$OUTDIR/metadata.txt"
echo ""

# Show detected devices
echo "--- Detected devices ---"
$BENCH --list-devices 2>&1 | tee "$OUTDIR/devices.txt"
echo ""

# ── Test 1: Token generation at varying batch sizes ──────────────
# Measures expert coalescing benefits — larger batches should amortize
# expert weight reads across tokens routed to the same expert.
echo "--- Test 1/5: Token generation batch size sweep (ngl=99) ---"
$BENCH -m "$MODEL" \
    -ngl 99 \
    --no-mmap \
    -fa 1 \
    -ctk q8_0 -ctv q8_0 \
    -t 4 \
    -p 0 -n 128 \
    -r "$REPS" \
    -o csv 2>/dev/null | tee "$OUTDIR/tg_batch_sweep.csv"
echo ""

# ── Test 2: Prompt processing vs token generation ratio ──────────
# MoE models have a distinctive pp/tg ratio because pp is compute-bound
# (all experts fire for each prompt token) while tg is bandwidth-bound
# (only top-K experts active). This ratio reveals routing overhead.
echo "--- Test 2/5: PP vs TG ratio (ngl=99) ---"
$BENCH -m "$MODEL" \
    -ngl 99 \
    --no-mmap \
    -fa 1 \
    -ctk q8_0 -ctv q8_0 \
    -t 4 \
    -p 128,512,1024,2048 -n 128 \
    -r "$REPS" \
    -o csv 2>/dev/null | tee "$OUTDIR/pp_tg_ratio.csv"
echo ""

# ── Test 3: Thread count sweep (bandwidth contention) ────────────
# MoE decode reads ~3B active params per token from shared memory bus.
# Too many CPU threads compete for bandwidth with the GPU. Find the
# sweet spot where GPU expert matmuls aren't starved.
echo "--- Test 3/5: Thread count sweep (ngl=99) ---"
$BENCH -m "$MODEL" \
    -ngl 99 \
    --no-mmap \
    -fa 1 \
    -ctk q8_0 -ctv q8_0 \
    -t 1,2,4,8 \
    -p 512 -n 128 \
    -r "$REPS" \
    -o csv 2>/dev/null | tee "$OUTDIR/thread_sweep.csv"
echo ""

# ── Test 4: KV cache quantization impact on MoE ──────────────────
# MoE models benefit more from KV cache quantization because the memory
# savings directly free bandwidth for expert weight reads.
echo "--- Test 4/5: KV cache quantization ---"
for KV in f16 q8_0 q4_0; do
    echo "  KV type: $KV"
    $BENCH -m "$MODEL" \
        -ngl 99 \
        --no-mmap \
        -fa 1 \
        -ctk "$KV" -ctv "$KV" \
        -t 4 \
        -p 512 -n 128 \
        -r "$REPS" \
        -o csv 2>/dev/null
done | tee "$OUTDIR/kv_cache_moe.csv"
echo ""

# ── Test 5: FORCE_MMQ vs hipBLAS for expert matmuls ──────────────
# The quantized matmul kernels (MMQ) can outperform hipBLAS for the
# smaller per-expert matrix sizes in MoE models. This test requires
# separate builds, so we check for an MMQ binary.
echo "--- Test 5/5: MMQ vs hipBLAS comparison ---"
MMQ_BENCH="./build-mmq/bin/llama-bench"
if [[ -x "$MMQ_BENCH" ]]; then
    echo "  Running hipBLAS path..."
    $BENCH -m "$MODEL" \
        -ngl 99 \
        --no-mmap \
        -fa 1 \
        -ctk q8_0 -ctv q8_0 \
        -t 4 \
        -p 512 -n 128 \
        -r "$REPS" \
        -o csv 2>/dev/null | tee "$OUTDIR/hipblas.csv"

    echo "  Running MMQ path..."
    $MMQ_BENCH -m "$MODEL" \
        -ngl 99 \
        --no-mmap \
        -fa 1 \
        -ctk q8_0 -ctv q8_0 \
        -t 4 \
        -p 512 -n 128 \
        -r "$REPS" \
        -o csv 2>/dev/null | tee "$OUTDIR/mmq.csv"
else
    echo "  Skipped — no MMQ build found at $MMQ_BENCH"
    echo "  To enable: cmake -DGGML_CUDA_FORCE_MMQ=ON -B build-mmq && cmake --build build-mmq"
fi
echo ""

# ── Summary ─────────────────────────────────────────────────────
echo "=== MoE Benchmark Complete ==="
echo "Results saved to: $OUTDIR/"
echo ""
echo "MoE-specific analysis tips:"
echo "  - PP/TG ratio for MoE should be much higher than dense models"
echo "    (PP activates all experts, TG only top-K)"
echo "  - Thread sweep: optimal is usually 2-4 threads with ngl=99"
echo "    (more threads steal bandwidth from expert weight reads)"
echo "  - KV cache q4_0 frees ~4x memory vs f16 for longer contexts"
echo "  - MMQ often beats hipBLAS for MoE due to smaller per-expert matrices"
echo ""
echo "Compare with dense model baseline:"
echo "  ./scripts/bench-strix-halo.sh <dense_model.gguf>"
