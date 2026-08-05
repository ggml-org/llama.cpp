#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 LLAMA_BENCH MODEL OUT_DIR [llama-bench arguments...]" >&2
    exit 2
fi

bench=$1
model=$2
out_dir=$3
shift 3
extra_args=("$@")

if [[ ! -x "$bench" ]]; then
    echo "llama-bench is not executable: $bench" >&2
    exit 2
fi
if [[ ! -f "$model" ]]; then
    echo "model does not exist: $model" >&2
    exit 2
fi

batch=${PREFILL_BATCH:-8192}
ubatch=${PREFILL_UBATCH:-2048}
threads=${PREFILL_THREADS:-25}
repetitions=${PREFILL_REPETITIONS:-5}
tokens=${PREFILL_TOKENS:-512,2048,8192}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-cutlass-moe"
mkdir -p "$run_dir"

nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true

run_case() {
    local label=$1
    shift
    env "${blackwell_prefill_clean_env[@]}" \
        GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 "$@" "$bench" \
        -m "$model" \
        -p "$tokens" \
        -n 0 \
        -r "$repetitions" \
        -t "$threads" \
        -ngl 999 \
        -b "$batch" \
        -ub "$ubatch" \
        -fa on \
        -o jsonl \
        "${extra_args[@]}" \
        > "$run_dir/$label.jsonl" \
        2> "$run_dir/$label.stderr"
    if [[ "$label" == cutlass-* ]] && ! grep -q 'MoE MMQ: backend=cutlass' "$run_dir/$label.stderr"; then
        echo "$label did not execute the CUTLASS MoE backend" >&2
        exit 1
    fi
}

run_case disabled GGML_CUDA_MOE_MMQ_DISABLE=1
run_case native
run_case cutlass-gemm \
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass \
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 \
    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=none
run_case cutlass-w13 \
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass \
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 \
    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=w13
run_case cutlass-full \
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass \
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 \
    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full
run_case cutlass-ceiling \
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass \
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 \
    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full \
    LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1 \
    LLAMA_CUDA_FATTN_Q_ROPE=1 \
    GGML_CUDA_FATTN_SM120_CAUSAL_SCHEDULE=1 \
    GGML_CUDA_ADD_RMS_NORM_FUSION=1

printf '%s\n' "$run_dir"
