#!/usr/bin/env bash
# Reproduce the raw mirrored vs vocabulary-parallel output A/B on four RDNA2 GPUs.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-$ROOT/build}
ROCM_PATH=${ROCM_PATH:-/opt/rocm/core-7.14}
MODEL=${MODEL:-$HOME/models/Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf}
OUT_DIR=${OUT_DIR:-$HOME/llama-jobs/vocab-ab-$(date +%Y%m%d-%H%M%S)}
REPETITIONS=${REPETITIONS:-5}
TOKENS=${TOKENS:-128}

mkdir -p "$OUT_DIR"
[ -x "$BUILD_DIR/bin/llama-bench" ] || { echo "missing llama-bench: $BUILD_DIR/bin/llama-bench" >&2; exit 2; }
[ -f "$MODEL" ] || { echo "missing model: $MODEL" >&2; exit 2; }

export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-10.3.0}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export GGML_HIP_GRAPHS=${GGML_HIP_GRAPHS:-1}
export GGML_CUDA_ALLREDUCE=nccl

run_mode() {
    local mode=$1 sequence=$2
    unset GGML_TP_VOCAB_OUTPUT GGML_TP_SHARDED_OUTPUT
    if [ "$mode" = vocab ]; then
        export GGML_TP_VOCAB_OUTPUT=1
    fi
    echo "=== sequence=$sequence mode=$mode ===" | tee -a "$OUT_DIR/summary.log"
    "$BUILD_DIR/bin/llama-bench" \
        -m "$MODEL" -ngl 99 -sm tensor -ts 1/1/1/1 -fa on \
        -b 2048 -ub 256 -p 0 -n "$TOKENS" -r "$REPETITIONS" -o csv \
        2>"$OUT_DIR/${sequence}-${mode}.stderr.log" \
        | tee "$OUT_DIR/${sequence}-${mode}.csv" | tee -a "$OUT_DIR/summary.log"
}

# Reverse order controls for model-load, clock-ramp, and thermal ordering.
run_mode off   ab1
run_mode vocab ab1
run_mode vocab ba2
run_mode off   ba2

echo "A/B artifacts: $OUT_DIR"