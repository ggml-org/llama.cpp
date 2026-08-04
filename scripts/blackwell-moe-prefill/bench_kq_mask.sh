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

tokens=${PREFILL_TOKENS:-8192}
batch=${PREFILL_BATCH:-8192}
ubatch=${PREFILL_UBATCH:-8192}
threads=${PREFILL_THREADS:-25}
repetitions=${PREFILL_REPETITIONS:-3}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-kq-mask"
mkdir -p "$run_dir"

nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
printf 'label\tenvironment\n' > "$run_dir/manifest.tsv"

run_case() {
    local label=$1
    shift
    printf '%s\t%s\n' "$label" "$*" >> "$run_dir/manifest.tsv"
    env LLAMA_KQ_MASK_CONTIGUOUS_LOG=1 LLAMA_CUDA_FATTN_DIRECT_CAUSAL_LOG=1 "$@" "$bench" \
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
}

if [[ ${KQ_MASK_VALIDATE:-0} == 1 ]]; then
    run_case validate LLAMA_KQ_MASK_CONTIGUOUS_VALIDATE=1
fi
run_case baseline LLAMA_KQ_MASK_CONTIGUOUS_DISABLE=1
run_case contiguous
run_case direct-causal LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1

if ! grep -q 'KQ mask: contiguous=0' "$run_dir/baseline.stderr"; then
    echo "baseline did not use the generic KQ mask path" >&2
    exit 1
fi
if ! grep -q 'KQ mask: contiguous=1' "$run_dir/contiguous.stderr"; then
    echo "contiguous KQ mask fast path was not selected" >&2
    exit 1
fi
if ! grep -q 'FlashAttention: direct-causal=1, swa=0' "$run_dir/direct-causal.stderr"; then
    echo "direct causal FlashAttention base KQ mask path was not selected" >&2
    exit 1
fi
if ! grep -q 'FlashAttention: direct-causal=1, swa=1' "$run_dir/direct-causal.stderr"; then
    echo "direct causal FlashAttention SWA KQ mask path was not selected" >&2
    exit 1
fi

printf '%s\n' "$run_dir"
