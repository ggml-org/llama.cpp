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
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-cutlass-decode"
mkdir -p "$run_dir"
cutlass_marker='MoE (MMQ CUTLASS|CUTLASS NVFP4) decode dispatch:'

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
    local graphs=$2
    local backend=$3
    local input_scale=$4
    local output=$5
    shift 5

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
        if [[ "$label" == cutlass-fused-bf16 ]]; then
            if ! grep -q 'MoE CUTLASS NVFP4 fused decode dispatch:' "$run_dir/$label.stderr"; then
                echo "$label did not execute the fused CUTLASS decode path" >&2
                exit 1
            fi
        elif ! grep -Eq "$cutlass_marker" "$run_dir/$label.stderr"; then
            echo "$label did not execute the CUTLASS decode path" >&2
            exit 1
        elif grep -q 'MoE CUTLASS NVFP4 decode dispatch:' "$run_dir/$label.stderr"; then
            if ! grep -Eq "MoE CUTLASS NVFP4 decode dispatch: tokens=1 .*K=2048 N=(512|1024).*output=$output" \
                    "$run_dir/$label.stderr"; then
                echo "$label did not execute an NVFP4 W13 shape" >&2
                exit 1
            fi
            if ! grep -q "MoE CUTLASS NVFP4 decode dispatch: tokens=1 .*K=512 N=2048.*output=$output" \
                    "$run_dir/$label.stderr"; then
                echo "$label did not execute the NVFP4 W2 shape" >&2
                exit 1
            fi
        fi
    elif grep -Eq "$cutlass_marker" "$run_dir/$label.stderr"; then
        echo "$label unexpectedly executed the CUTLASS decode path" >&2
        exit 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$graphs" "$backend" "$input_scale" "$output" \
        "$tokens" "$batch" "$ubatch" "$threads" "$repetitions" \
        >> "$run_dir/manifest.tsv"
}

run_case native-default on native off f32 GGML_CUDA_MOE_MMQ_DISABLE=1
run_case native-eager off native off f32 GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MOE_MMQ_DISABLE=1
run_case native-calibrated-eager off native on f32 \
    GGML_CUDA_DISABLE_GRAPHS=1 \
    GGML_CUDA_MOE_MMQ_DISABLE=1 \
    LLAMA_CUDA_MOE_NVFP4_INPUT_SCALE=1
run_case cutlass-dynamic-bf16 off cutlass off bf16 "${cutlass_environment[@]}"
run_case cutlass-fused-bf16 off cutlass off bf16 \
    "${cutlass_environment[@]}" \
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_FUSED=1
run_case cutlass-calibrated-bf16 off cutlass on bf16 \
    "${cutlass_environment[@]}" \
    LLAMA_CUDA_MOE_NVFP4_INPUT_SCALE=1
run_case cutlass-calibrated-f32 off cutlass on f32 \
    "${cutlass_environment[@]}" \
    LLAMA_CUDA_MOE_NVFP4_INPUT_SCALE=1 \
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_OUTPUT_F32=1

if [[ ${DECODE_FULL_MATRIX:-0} == 1 ]]; then
    run_case cutlass-dynamic-f32 off cutlass off f32 \
        "${cutlass_environment[@]}" \
        GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_OUTPUT_F32=1
fi

"$python_bin" "$script_dir/summarize_decode_bench.py" "$run_dir" > "$run_dir/summary.md"
printf '%s\n' "$run_dir"
