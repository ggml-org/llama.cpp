#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 LABEL LLAMA_BENCH MODEL OUT_DIR [llama-bench arguments...]" >&2
    exit 2
fi

label=$1
bench=$2
model=$3
out_dir=$4
shift 4

if [[ ! -x "$bench" ]]; then
    echo "llama-bench is not executable: $bench" >&2
    exit 2
fi
if [[ ! -f "$model" ]]; then
    echo "model does not exist: $model" >&2
    exit 2
fi

nsys_bin=${NSYS_BIN:-nsys}
if ! command -v "$nsys_bin" >/dev/null 2>&1; then
    echo "nsys does not exist: $nsys_bin" >&2
    exit 2
fi
tokens=${PREFILL_TOKENS:-8192}
batch=${PREFILL_BATCH:-8192}
ubatch=${PREFILL_UBATCH:-2048}
threads=${PREFILL_THREADS:-25}
repetitions=${PREFILL_REPETITIONS:-3}
warmup_args=(--no-warmup)
if [[ ${PREFILL_PROFILE_WARMUP:-0} != 0 ]]; then
    warmup_args=()
fi
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-$label"
report="$run_dir/profile"
bench_stdout="$run_dir/llama-bench.jsonl"
bench_stderr="$run_dir/llama-bench.stderr"

mkdir -p "$run_dir"
nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
"$nsys_bin" --version > "$run_dir/nsys-version.txt" 2>&1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
printf 'tokens\tbatch\tubatch\tthreads\trepetitions\twarmup\n' > "$run_dir/profile-config.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tokens" "$batch" "$ubatch" "$threads" "$repetitions" "${PREFILL_PROFILE_WARMUP:-0}" \
    >> "$run_dir/profile-config.tsv"
{
    printf 'GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MOE_PROFILE=1 '
    printf '%q ' "$bench" \
        -m "$model" -p "$tokens" -n 0 -r "$repetitions" \
        "${warmup_args[@]}" -t "$threads" -ngl 999 -b "$batch" -ub "$ubatch" -fa on -o jsonl "$@"
    printf '\n'
} > "$run_dir/benchmark-command.sh"

GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MOE_PROFILE=1 "$nsys_bin" profile \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --output="$report" \
    bash -c 'stdout=$1; stderr=$2; shift 2; exec "$@" > "$stdout" 2> "$stderr"' \
        bash "$bench_stdout" "$bench_stderr" "$bench" \
        -m "$model" \
        -p "$tokens" \
        -n 0 \
        -r "$repetitions" \
        "${warmup_args[@]}" \
        -t "$threads" \
        -ngl 999 \
        -b "$batch" \
        -ub "$ubatch" \
        -fa on \
        -o jsonl \
        "$@" \
        > "$run_dir/nsys.stdout" \
        2> "$run_dir/nsys.stderr"

if ! "$nsys_bin" stats \
        --quiet \
        --force-export=true \
        --report cuda_gpu_kern_sum \
        --format csv \
        "$report.nsys-rep" \
        > "$run_dir/cuda-kernels.csv" \
        2> "$run_dir/cuda-kernels.stderr"; then
    echo "warning: failed to export CUDA kernel statistics" >&2
fi

if ! "$nsys_bin" stats \
        --quiet \
        --force-export=true \
        --report nvtx_gpu_proj_sum \
        --format csv \
        "$report.nsys-rep" \
        > "$run_dir/nvtx-gpu.csv" \
        2> "$run_dir/nvtx-gpu.stderr"; then
    echo "warning: failed to export NVTX GPU projection statistics" >&2
fi

if ! "$nsys_bin" stats \
        --quiet \
        --force-export=true \
        --report cuda_api_sum \
        --format csv \
        "$report.nsys-rep" \
        > "$run_dir/cuda-api.csv" \
        2> "$run_dir/cuda-api.stderr"; then
    echo "warning: failed to export CUDA API statistics" >&2
fi

printf '%s\n' "$run_dir"
