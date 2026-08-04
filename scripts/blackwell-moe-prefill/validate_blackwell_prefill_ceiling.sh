#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 TEST_BACKEND_OPS LLAMA_DEBUG MODEL OUT_DIR [llama-debug arguments...]" >&2
    exit 2
fi

test_backend_ops=$1
debug_bin=$2
model=$3
out_dir=$4
shift 4
extra_args=("$@")
python_bin=${PYTHON:-python3}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"

for executable in "$test_backend_ops" "$debug_bin"; do
    if [[ ! -x "$executable" ]]; then
        echo "executable does not exist: $executable" >&2
        exit 2
    fi
done
if [[ ! -f "$model" ]]; then
    echo "model does not exist: $model" >&2
    exit 2
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "python interpreter does not exist: $python_bin" >&2
    exit 2
fi

tokens=${PREFILL_CEILING_VALIDATE_TOKENS:-1024}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-blackwell-prefill-validation"
prompt_file="$run_dir/prompt.txt"
mkdir -p "$run_dir"
for ((i = 0; i < tokens; ++i)); do
    printf ' hello' >> "$prompt_file"
done
printf '\n' >> "$prompt_file"

env "${blackwell_prefill_clean_env[@]}" GGML_CUDA_DISABLE_GRAPHS=1 \
    GGML_CUDA_ADD_RMS_NORM_TEST=1 GGML_CUDA_ADD_RMS_NORM_FUSION=1 GGML_CUDA_ADD_RMS_NORM_LOG=1 \
    "$test_backend_ops" test -b CUDA0 -o ADD_RMS_NORM_MUL \
    > "$run_dir/add-rms-norm-correctness.txt" \
    2> "$run_dir/add-rms-norm-correctness.stderr"

if ! grep -q 'ADD_RMS_NORM_MUL' "$run_dir/add-rms-norm-correctness.txt"; then
    echo "no add RMS norm fusion case matched" >&2
    exit 1
fi

direct='LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1'
ceiling_attention="$direct LLAMA_CUDA_FATTN_Q_ROPE=1 GGML_CUDA_FATTN_SM120_CAUSAL_SCHEDULE=1"
tma_inplace='GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma-inplace GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted'
tuned_moe="$tma_inplace GGML_CUDA_MOE_MMQ_W13_TILE_ROWS=64 GGML_CUDA_MOE_MMQ_W2_OUTPUT_TILE_MAJOR=1"
cases=(
    'reference|reference|GGML_CUDA_MOE_MMQ_DISABLE=1'
    "direct-causal|bitwise|GGML_CUDA_MOE_MMQ_DISABLE=1 $direct"
    "tma-inplace|bitwise|$tma_inplace"
    "strict-tuned|bitwise|$direct $tuned_moe GGML_CUDA_ADD_RMS_NORM_FUSION=1"
    "full-ceiling-tuned|metrics-only|$ceiling_attention $tuned_moe GGML_CUDA_ADD_RMS_NORM_FUSION=1"
)

printf 'label\tvalidation\tenvironment\n' > "$run_dir/manifest.tsv"

for entry in "${cases[@]}"; do
    label=${entry%%|*}
    remainder=${entry#*|}
    validation=${remainder%%|*}
    environment=${remainder#*|}
    read -r -a env_args <<< "$environment"
    mkdir -p "$run_dir/$label"
    env "${blackwell_prefill_clean_env[@]}" GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 LLAMA_KQ_MASK_CONTIGUOUS_LOG=1 LLAMA_CUDA_FATTN_DIRECT_CAUSAL_LOG=1 \
        GGML_CUDA_FATTN_LOG_CONFIG=1 GGML_CUDA_ADD_RMS_NORM_LOG=1 "${env_args[@]}" "$debug_bin" \
        -m "$model" \
        -f "$prompt_file" \
        -n 0 \
        -ngl 999 \
        -b "$tokens" \
        -ub "$tokens" \
        -fa on \
        -v \
        --save-logits \
        --logits-output-dir "$run_dir/$label" \
        "${extra_args[@]}" \
        > "$run_dir/$label.stdout" \
        2> "$run_dir/$label.stderr"
    printf '%s\t%s\t%s\n' "$label" "$validation" "$environment" >> "$run_dir/manifest.tsv"
done

for label in tma-inplace strict-tuned full-ceiling-tuned; do
    if ! grep -q 'weights=tma-inplace' "$run_dir/$label.stdout" "$run_dir/$label.stderr" ||
            ! grep -q 'w13-epilogue=tma-epilogue' "$run_dir/$label.stdout" "$run_dir/$label.stderr" ||
            ! grep -q 'w2-epilogue=tma-weighted' "$run_dir/$label.stdout" "$run_dir/$label.stderr"; then
        echo "$label did not select the complete in-place TMA MoE path" >&2
        exit 1
    fi
done
for label in direct-causal strict-tuned full-ceiling-tuned; do
    if ! grep -q 'FlashAttention: direct-causal=1' "$run_dir/$label.stdout" "$run_dir/$label.stderr"; then
        echo "$label did not select direct causal Attention" >&2
        exit 1
    fi
done
for label in strict-tuned full-ceiling-tuned; do
    if ! grep -q 'CUDA add RMS norm fusion: enabled' "$run_dir/$label.stdout" "$run_dir/$label.stderr"; then
        echo "$label did not select add RMS norm fusion" >&2
        exit 1
    fi
done
if ! grep -q 'FlashAttention: q-rope=1' "$run_dir/full-ceiling-tuned.stdout" "$run_dir/full-ceiling-tuned.stderr" ||
        ! grep -q 'FlashAttention: sm120-causal=1' "$run_dir/full-ceiling-tuned.stdout" "$run_dir/full-ceiling-tuned.stderr"; then
    echo "full-ceiling-tuned did not select the ceiling Attention path" >&2
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

for label in direct-causal tma-inplace strict-tuned; do
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

ceiling_logits=$(find_logits "$run_dir/full-ceiling-tuned")
if [[ -z "$ceiling_logits" ]]; then
    echo "llama-debug did not produce full-ceiling-tuned logits" >&2
    exit 1
fi
"$python_bin" "$script_dir/compare_logits.py" \
    "$reference_logits" "$ceiling_logits" \
    --metrics-only \
    > "$run_dir/full-ceiling-tuned-comparison.json"

printf '%s\n' "$run_dir"
