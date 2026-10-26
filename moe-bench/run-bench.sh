#!/usr/bin/env bash
# Reproduce the MoE expert cache measurements in moe-bench/RESULTS.md.
#
# Research material for the moe-gpu-optimize branch, not intended for upstream.
#
#   ./moe-bench/run-bench.sh <model.gguf> [out_dir]
#
# Needs a CUDA build in ./build (cmake -B build -DGGML_CUDA=ON && cmake --build build).
# Close other GPU and memory hungry applications first: the host expert working set
# of a 20 GiB model does not fit next to them on a 32 GiB machine.

set -euo pipefail

MODEL=${1:?usage: run-bench.sh <model.gguf> [out_dir]}
OUT=${2:-moe-bench/out}
BIN=${BIN:-./build/bin}
NCMOE=${NCMOE:-40}          # layers whose experts stay in host memory
THREADS=${THREADS:-8}
REPS=${REPS:-5}
NGEN=${NGEN:-64}
NPROMPT=${NPROMPT:-512}
SIZES=${SIZES:-384,528,645,1024,2048,3072}

mkdir -p "$OUT"
echo "model   : $MODEL"
echo "binaries: $BIN"
echo "out     : $OUT"
"$BIN/llama-bench" --version 2>&1 | head -2

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null | tee "$OUT/gpu.txt" || true

# ---------------------------------------------------------------------------
# 1. generation throughput over the cache size sweep, with and without the
#    sampled least-used eviction that this branch adds
# ---------------------------------------------------------------------------
for sample in 0 16; do
    echo "== tg$NGEN, LLAMA_MOE_CACHE_SAMPLE=$sample =="
    LLAMA_MOE_CACHE_SAMPLE=$sample "$BIN/llama-bench" \
        -m "$MODEL" -ngl 99 -ncmoe "$NCMOE" -mcmoe "$SIZES" \
        -n "$NGEN" -p 0 -r "$REPS" -t "$THREADS" \
        -o md 2>&1 | tee "$OUT/tg_sample$sample.md"
done

# ---------------------------------------------------------------------------
# 2. prompt processing, which bypasses the cache above 32 tokens per ubatch
# ---------------------------------------------------------------------------
echo "== pp$NPROMPT =="
"$BIN/llama-bench" -m "$MODEL" -ngl 99 -ncmoe "$NCMOE" -mcmoe "$SIZES" \
    -n 0 -p "$NPROMPT" -r "$REPS" -t "$THREADS" -o md 2>&1 | tee "$OUT/pp.md"

# ---------------------------------------------------------------------------
# 3. host expert page pinning: pageable mmap vs capped pinning vs load-mode none
# ---------------------------------------------------------------------------
echo "== pinning =="
{
    echo "### pageable mmap (default)"
    "$BIN/llama-bench" -m "$MODEL" -ngl 99 -ncmoe "$NCMOE" -mcmoe 1024 \
        -n "$NGEN" -p 0 -r "$REPS" -t "$THREADS" -o md 2>&1

    for mib in 2048 4096 8192; do
        echo "### mmap + pinned $mib MiB"
        GGML_CUDA_REGISTER_HOST=1 LLAMA_MOE_PIN_MIB=$mib \
        "$BIN/llama-bench" -m "$MODEL" -ngl 99 -ncmoe "$NCMOE" -mcmoe 1024 \
            -n "$NGEN" -p 0 -r "$REPS" -t "$THREADS" -o md 2>&1
    done

    echo "### --load-mode none (experts in a pinned host buffer, no mmap)"
    "$BIN/llama-bench" -m "$MODEL" -ngl 99 -ncmoe "$NCMOE" -mcmoe 1024 -lm none \
        -n "$NGEN" -p 0 -r "$REPS" -t "$THREADS" -o md 2>&1
} | tee "$OUT/pinning.md"

# ---------------------------------------------------------------------------
# 4. deterministic cache telemetry: hit rate and upload volume per cache size
#    these numbers are exact and do not depend on machine load
# ---------------------------------------------------------------------------
echo "== telemetry =="
: > "$OUT/telemetry.txt"
for mib in ${SIZES//,/ }; do
    for sample in 0 16; do
        line=$(LLAMA_MOE_CACHE_SAMPLE=$sample "$BIN/llama-completion" \
            -m "$MODEL" -ngl 99 -ncmoe "$NCMOE" --moe-cache-mib "$mib" \
            -c 2048 -n 256 -f moe-bench/prompt.txt --temp 0 --seed 1 \
            -t "$THREADS" --no-warmup -no-cnv -v < /dev/null 2>&1 \
            | grep -E "ubatch <= 8" | tail -1)
        echo "mcmoe=$mib sample=$sample $line" | tee -a "$OUT/telemetry.txt"
    done
done

echo
echo "done, results in $OUT"
