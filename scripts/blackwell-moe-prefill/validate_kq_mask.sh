#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 LLAMA_DEBUG MODEL OUT_DIR [llama-debug arguments...]" >&2
    exit 2
fi

debug_bin=$1
model=$2
out_dir=$3
shift 3
extra_args=("$@")
python_bin=${PYTHON:-python3}

if [[ ! -x "$debug_bin" ]]; then
    echo "llama-debug is not executable: $debug_bin" >&2
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

tokens=${KQ_MASK_VALIDATE_TOKENS:-1024}
batch=${KQ_MASK_VALIDATE_BATCH:-$((tokens + 16))}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-kq-mask-validation"
reference_dir="$run_dir/reference"
prompt_file="$run_dir/prompt.txt"
mkdir -p "$reference_dir"

for ((i = 0; i < tokens; ++i)); do
    printf ' hello' >> "$prompt_file"
done

run_case() {
    local label=$1
    shift
    local logits_dir="$run_dir/$label"
    mkdir -p "$logits_dir"
    env "$@" "$debug_bin" \
        -m "$model" \
        -f "$prompt_file" \
        -n 0 \
        -ngl 999 \
        -b "$batch" \
        -ub "$batch" \
        -fa on \
        -lv 4 \
        --save-logits \
        --logits-output-dir "$logits_dir" \
        "${extra_args[@]}" \
        > "$run_dir/$label.stdout" \
        2> "$run_dir/$label.stderr"
}

run_case reference LLAMA_KQ_MASK_CONTIGUOUS_DISABLE=1
run_case contiguous LLAMA_KQ_MASK_CONTIGUOUS_VALIDATE=1 LLAMA_KQ_MASK_CONTIGUOUS_LOG=1 LLAMA_CUDA_FATTN_DIRECT_CAUSAL_LOG=1
run_case direct-causal LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1 LLAMA_KQ_MASK_CONTIGUOUS_LOG=1 LLAMA_CUDA_FATTN_DIRECT_CAUSAL_LOG=1

if ! grep -q 'KQ mask: contiguous=1' "$run_dir/contiguous.stdout" "$run_dir/contiguous.stderr"; then
    echo "contiguous KQ mask fast path was not selected" >&2
    exit 1
fi
if ! grep -q 'FlashAttention: direct-causal=1, swa=0' "$run_dir/direct-causal.stdout" "$run_dir/direct-causal.stderr"; then
    echo "direct causal FlashAttention base KQ mask path was not selected" >&2
    exit 1
fi
if ! grep -q 'FlashAttention: direct-causal=1, swa=1' "$run_dir/direct-causal.stdout" "$run_dir/direct-causal.stderr"; then
    echo "direct causal FlashAttention SWA KQ mask path was not selected" >&2
    exit 1
fi

reference_logits=$(find "$reference_dir" -maxdepth 1 -type f -name 'llamacpp-*.bin' ! -name '*tokens.bin' -print -quit)
if [[ -z "$reference_logits" ]]; then
    echo "llama-debug did not produce reference logits" >&2
    exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for label in contiguous direct-causal; do
    candidate_logits=$(find "$run_dir/$label" -maxdepth 1 -type f -name 'llamacpp-*.bin' ! -name '*tokens.bin' -print -quit)
    if [[ -z "$candidate_logits" ]]; then
        echo "llama-debug did not produce $label logits" >&2
        exit 1
    fi
    "$python_bin" "$script_dir/compare_logits.py" \
        "$reference_logits" \
        "$candidate_logits" \
        --rtol 0 \
        --atol 0 \
        --max-nmse 0 \
        > "$run_dir/$label-comparison.json"
done

printf '%s\n' "$run_dir"
