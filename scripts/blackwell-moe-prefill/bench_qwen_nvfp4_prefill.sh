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

tokens=${QWEN_PREFILL_TOKENS:-512,2048,8192}
batch=${QWEN_PREFILL_BATCH:-8192}
ubatch=${QWEN_PREFILL_UBATCH:-8192}
threads=${QWEN_PREFILL_THREADS:-25}
repetitions=${QWEN_PREFILL_REPETITIONS:-5}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-qwen-nvfp4-prefill"
mkdir -p "$run_dir"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
printf 'label\tbackend\tgraphs\ttokens\tbatch\tubatch\tthreads\trepetitions\n' > "$run_dir/manifest.tsv"

cutlass_environment=(
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass
    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0
    GGML_CUDA_MOE_MMQ_CUTLASS_NVFP4_PREFILL_LOG=1
)

run_case() {
    local label=$1
    local backend=$2
    local graphs=$3
    shift 3

    env "${blackwell_prefill_clean_env[@]}" \
        -u GGML_CUDA_DISABLE_GRAPHS \
        "$@" \
        "$bench" \
        -m "$model" \
        -p "$tokens" \
        -n 0 \
        -r "$repetitions" \
        -t "$threads" \
        -ngl 999 \
        -b "$batch" \
        -ub "$ubatch" \
        -fa on \
        -v \
        -o jsonl \
        "${extra_args[@]}" \
        > "$run_dir/$label.jsonl" \
        2> "$run_dir/$label.stderr"

    if [[ "$backend" == cutlass ]]; then
        if ! grep -q 'MoE CUTLASS NVFP4 prefill dispatch:' "$run_dir/$label.stderr"; then
            echo "$label did not execute the Qwen NVFP4 CUTLASS prefill path" >&2
            exit 1
        fi
    elif grep -q 'MoE CUTLASS NVFP4 prefill dispatch:' "$run_dir/$label.stderr"; then
        echo "$label unexpectedly executed the CUTLASS prefill path" >&2
        exit 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$backend" "$graphs" "$tokens" "$batch" "$ubatch" "$threads" "$repetitions" \
        >> "$run_dir/manifest.tsv"
}

run_case native-eager native off GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MOE_MMQ_DISABLE=1
run_case cutlass-eager cutlass off GGML_CUDA_DISABLE_GRAPHS=1 "${cutlass_environment[@]}"
run_case cutlass-graphs cutlass on "${cutlass_environment[@]}"

printf '%s\n' "$run_dir"
