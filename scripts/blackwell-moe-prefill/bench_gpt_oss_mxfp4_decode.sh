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
python_bin=${PYTHON:-python3}

if [[ ! -x "$bench" ]]; then
    echo "llama-bench is not executable: $bench" >&2
    exit 2
fi
if [[ ! -f "$model" ]]; then
    echo "model does not exist: $model" >&2
    exit 2
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "python interpreter does not exist: $python_bin" >&2
    exit 2
fi

tokens=${DECODE_TOKENS:-128}
batch=${DECODE_BATCH:-512}
ubatch=${DECODE_UBATCH:-512}
threads=${DECODE_THREADS:-25}
repetitions=${DECODE_REPETITIONS:-5}
expected_layers=${GPT_OSS_EXPERT_LAYERS:-36}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-gpt-oss-mxfp4-decode"
mkdir -p "$run_dir"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
printf 'label\tgraphs\tbackend\tinput_scale\toutput\ttokens\tbatch\tubatch\tthreads\trepetitions\n' \
    > "$run_dir/manifest.tsv"

cutlass_environment=(
    GGML_CUDA_DISABLE_GRAPHS=1
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE=1
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_LOG=1
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0
    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full
    GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N=32
    GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N=32
    GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB=1
    GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB=1
)

run_case() {
    local label=$1
    local backend=$2
    shift 2

    env "${blackwell_prefill_clean_env[@]}" \
        -u GGML_CUDA_DISABLE_GRAPHS \
        GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
        "$@" \
        "$bench" \
        -m "$model" \
        -p 0 \
        -n "$tokens" \
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
        if ! grep -q 'MoE MMQ CUTLASS decode dispatch:.*schedule=direct' "$run_dir/$label.stderr"; then
            echo "$label did not execute the direct-schedule CUTLASS decode path" >&2
            exit 1
        fi
        local matched_layers
        matched_layers=$(grep 'MoE MMQ CUTLASS decode dispatch:' "$run_dir/$label.stderr" |
            sed -nE 's/.* weight=([^ ]+) .*/\1/p' | sort -u | wc -l)
        if (( matched_layers < expected_layers )); then
            echo "$label matched only $matched_layers of $expected_layers expected expert layers" >&2
            exit 1
        fi
    elif grep -q 'MoE MMQ CUTLASS decode dispatch:' "$run_dir/$label.stderr"; then
        echo "$label unexpectedly executed the CUTLASS decode path" >&2
        exit 1
    fi

    printf '%s\toff\t%s\toff\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$backend" "$([[ "$backend" == cutlass ]] && echo bf16 || echo f32)" \
        "$tokens" "$batch" "$ubatch" "$threads" "$repetitions" \
        >> "$run_dir/manifest.tsv"
}

run_case native-eager native \
    GGML_CUDA_DISABLE_GRAPHS=1 \
    GGML_CUDA_MOE_MMQ_DISABLE=1
run_case cutlass-full cutlass "${cutlass_environment[@]}"

"$python_bin" "$script_dir/summarize_decode_bench.py" "$run_dir" > "$run_dir/summary.md"
printf '%s\n' "$run_dir"
