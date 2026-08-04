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

tokens=${ATTENTION_VALIDATE_TOKENS:-1024}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-attention-validation"
prompt_file="$run_dir/prompt.txt"
mkdir -p "$run_dir"
for ((i = 0; i < tokens; ++i)); do
    printf ' hello' >> "$prompt_file"
done
printf '\n' >> "$prompt_file"

cases=(
    'reference|'
    'direct-causal|LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1'
    'q-rope|LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1 LLAMA_CUDA_FATTN_Q_ROPE=1'
    'causal-tiles|LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1 LLAMA_CUDA_FATTN_Q_ROPE=1 GGML_CUDA_FATTN_CAUSAL_TILES=1'
    'sm120-causal|LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1 LLAMA_CUDA_FATTN_Q_ROPE=1 GGML_CUDA_FATTN_SM120_CAUSAL_SCHEDULE=1'
)

for entry in "${cases[@]}"; do
    label=${entry%%|*}
    environment=${entry#*|}
    read -r -a env_args <<< "$environment"
    mkdir -p "$run_dir/$label"
    env GGML_CUDA_DISABLE_GRAPHS=1 LLAMA_KQ_MASK_CONTIGUOUS_LOG=1 LLAMA_CUDA_FATTN_DIRECT_CAUSAL_LOG=1 GGML_CUDA_FATTN_LOG_CONFIG=1 \
        "${env_args[@]}" "$debug_bin" \
        -m "$model" \
        -f "$prompt_file" \
        -n 0 \
        -ngl 999 \
        -b "$tokens" \
        -ub "$tokens" \
        -fa on \
        --save-logits \
        --logits-output-dir "$run_dir/$label" \
        "${extra_args[@]}" \
        > "$run_dir/$label.stdout" \
        2> "$run_dir/$label.stderr"
done

for label in direct-causal q-rope causal-tiles sm120-causal; do
    if ! grep -q 'FlashAttention: direct-causal=1' "$run_dir/$label.stderr"; then
        echo "$label did not select direct causal attention" >&2
        exit 1
    fi
done
for label in q-rope causal-tiles sm120-causal; do
    if ! grep -q 'FlashAttention: q-rope=1' "$run_dir/$label.stderr"; then
        echo "$label did not select Q RoPE fusion" >&2
        exit 1
    fi
done
if ! grep -q 'FlashAttention: sm120-causal=1' "$run_dir/sm120-causal.stderr"; then
    echo "sm120-causal did not select the SM120 schedule" >&2
    exit 1
fi

find_logits() {
    find "$1" -maxdepth 1 -type f -name 'llamacpp-*.bin' ! -name '*tokens.bin' -print -quit
}

reference_logits=$(find_logits "$run_dir/reference")
if [[ -z "$reference_logits" ]]; then
    echo "llama-debug did not produce reference logits" >&2
    exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for label in direct-causal q-rope causal-tiles sm120-causal; do
    candidate_logits=$(find_logits "$run_dir/$label")
    if [[ -z "$candidate_logits" ]]; then
        echo "llama-debug did not produce $label logits" >&2
        exit 1
    fi
    "$python_bin" "$script_dir/compare_logits.py" \
        "$reference_logits" "$candidate_logits" \
        --rtol 0 \
        --atol 0 \
        --max-nmse 0 \
        > "$run_dir/$label-comparison.json"
done

printf '%s\n' "$run_dir"
