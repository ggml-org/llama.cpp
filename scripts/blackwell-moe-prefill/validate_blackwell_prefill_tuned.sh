#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 LLAMA_DEBUG MODEL OUT_DIR" >&2
    exit 2
fi

debug_bin=$1
model=$2
out_dir=$3
python_bin=${PYTHON:-python3}
tokens=${PREFILL_TUNED_VALIDATE_TOKENS:-8192}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"

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

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-blackwell-prefill-tuned-validation"
prompt_file="$run_dir/prompt.txt"
mkdir -p "$run_dir/reference" "$run_dir/tuned"
for ((i = 0; i < tokens; ++i)); do
    printf ' hello' >> "$prompt_file"
done
printf '\n' >> "$prompt_file"

common=(
    "$debug_bin"
    -m "$model"
    -f "$prompt_file"
    -n 0
    -ngl 999
    -b "$tokens"
    -ub "$tokens"
    -fa on
    -v
    --save-logits
)

env "${blackwell_prefill_clean_env[@]}" GGML_CUDA_DISABLE_GRAPHS=1 \
    GGML_CUDA_MOE_MMQ_DISABLE=1 \
    "${common[@]}" \
    --logits-output-dir "$run_dir/reference" \
    > "$run_dir/reference.stdout" \
    2> "$run_dir/reference.stderr"

env "${blackwell_prefill_clean_env[@]}" GGML_CUDA_DISABLE_GRAPHS=1 \
    GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
    LLAMA_CUDA_FATTN_DIRECT_CAUSAL_LOG=1 \
    GGML_CUDA_FATTN_LOG_CONFIG=1 \
    GGML_CUDA_ADD_RMS_NORM_LOG=1 \
    LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1 \
    GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma-inplace \
    GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 \
    GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 \
    GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue \
    GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted \
    GGML_CUDA_MOE_MMQ_W13_TILE_ROWS=64 \
    GGML_CUDA_MOE_MMQ_W2_OUTPUT_TILE_MAJOR=1 \
    GGML_CUDA_ADD_RMS_NORM_FUSION=1 \
    "${common[@]}" \
    --logits-output-dir "$run_dir/tuned" \
    > "$run_dir/tuned.stdout" \
    2> "$run_dir/tuned.stderr"

if ! grep -q 'FlashAttention: direct-causal=1' "$run_dir/tuned.stdout" "$run_dir/tuned.stderr" ||
        ! grep -q 'weights=tma-inplace' "$run_dir/tuned.stdout" "$run_dir/tuned.stderr" ||
        ! grep -q 'w13-epilogue=tma-epilogue' "$run_dir/tuned.stdout" "$run_dir/tuned.stderr" ||
        ! grep -q 'w2-epilogue=tma-weighted' "$run_dir/tuned.stdout" "$run_dir/tuned.stderr" ||
        ! grep -q 'CUDA add RMS norm fusion: enabled' "$run_dir/tuned.stdout" "$run_dir/tuned.stderr"; then
    echo "tuned validation did not select the complete strict path" >&2
    exit 1
fi

reference_logits=$(find "$run_dir/reference" -maxdepth 1 -type f -name 'llamacpp-*.bin' ! -name '*tokens.bin' -print -quit)
tuned_logits=$(find "$run_dir/tuned" -maxdepth 1 -type f -name 'llamacpp-*.bin' ! -name '*tokens.bin' -print -quit)
if [[ -z "$reference_logits" || -z "$tuned_logits" ]]; then
    echo "llama-debug did not produce logits" >&2
    exit 1
fi

"$python_bin" "$script_dir/compare_logits.py" \
    "$reference_logits" "$tuned_logits" \
    --rtol 0 \
    --atol 0 \
    --max-nmse 0 \
    > "$run_dir/comparison.json"

printf '%s\n' "$run_dir"
