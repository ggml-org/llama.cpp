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

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-moe-vllm-validation"
prompt_file="$run_dir/prompt.txt"
mkdir -p "$run_dir"
for ((i = 0; i < 1024; ++i)); do
    printf ' hello' >> "$prompt_file"
done
printf '\n' >> "$prompt_file"

common='GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 GGML_CUDA_MOE_MMQ_REPACK_ASYNC=1 GGML_CUDA_MOE_MMQ_REPACK_CACHE_ENTRIES=8 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue'
tma_inplace='GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma-inplace GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted'
cases=(
    'reference|GGML_CUDA_MOE_MMQ_DISABLE=1'
    "fp4-full-k|$common GGML_CUDA_MOE_MMQ_TMA_TAIL_DISABLE=1 GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted"
    "fp4-weighted|$common GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted"
    "fp4-atomic|$common GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-atomic"
    "mxfp8-weighted|$common GGML_CUDA_MOE_MMQ_ACTIVATION_FORMAT=mxfp8 GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted"
    "mxfp8-atomic|$common GGML_CUDA_MOE_MMQ_ACTIVATION_FORMAT=mxfp8 GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-atomic"
    "tma-inplace|$tma_inplace"
)

for entry in "${cases[@]}"; do
    label=${entry%%|*}
    environment=${entry#*|}
    read -r -a env_args <<< "$environment"
    mkdir -p "$run_dir/$label"
    env GGML_CUDA_DISABLE_GRAPHS=1 "${env_args[@]}" "$debug_bin" \
        -m "$model" \
        -f "$prompt_file" \
        -n 0 \
        -ngl 999 \
        -b 2048 \
        -ub 2048 \
        --save-logits \
        --logits-output-dir "$run_dir/$label" \
        "${extra_args[@]}" \
        > "$run_dir/$label.stdout" \
        2> "$run_dir/$label.stderr"
done

find_logits() {
    find "$1" -maxdepth 1 -type f -name 'llamacpp-*.bin' ! -name '*tokens.bin' -print -quit
}

reference_logits=$(find_logits "$run_dir/reference")
if [[ -z "$reference_logits" ]]; then
    echo "llama-debug did not produce reference logits" >&2
    exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for label in fp4-full-k fp4-weighted tma-inplace; do
    candidate_logits=$(find_logits "$run_dir/$label")
    "$python_bin" "$script_dir/compare_logits.py" \
        "$reference_logits" "$candidate_logits" \
        --rtol 0 \
        --atol 0 \
        --max-nmse 0 \
        > "$run_dir/$label-comparison.json"
done

candidate_logits=$(find_logits "$run_dir/fp4-atomic")
"$python_bin" "$script_dir/compare_logits.py" \
    "$reference_logits" "$candidate_logits" \
    --rtol "${MOE_ATOMIC_RTOL:-1e-5}" \
    --atol "${MOE_ATOMIC_ATOL:-1e-5}" \
    --max-nmse "${MOE_ATOMIC_MAX_NMSE:-1e-10}" \
    > "$run_dir/fp4-atomic-comparison.json"

for label in mxfp8-weighted mxfp8-atomic; do
    candidate_logits=$(find_logits "$run_dir/$label")
    "$python_bin" "$script_dir/compare_logits.py" \
        "$reference_logits" "$candidate_logits" \
        --rtol "${MOE_MXFP8_RTOL:-0.05}" \
        --atol "${MOE_MXFP8_ATOL:-0.1}" \
        --max-nmse "${MOE_MXFP8_MAX_NMSE:-0.001}" \
        > "$run_dir/$label-comparison.json"
done

printf '%s\n' "$run_dir"
