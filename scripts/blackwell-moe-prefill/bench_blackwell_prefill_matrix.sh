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

python_bin=${PYTHON:-python3}
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "python interpreter does not exist: $python_bin" >&2
    exit 2
fi

threads=${PREFILL_THREADS:-25}
repetitions=${PREFILL_REPETITIONS:-3}
ubatches=${PREFILL_MATRIX_UBATCHES:-2048,8192}
ubatches=${ubatches// /,}
selected=,${PREFILL_MATRIX_CASES:-all},
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-blackwell-prefill-matrix"
mkdir -p "$run_dir"

nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
printf 'label\tvalidation\tenvironment\tresult\n' > "$run_dir/manifest.tsv"

direct='LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1'
q_rope="$direct LLAMA_CUDA_FATTN_Q_ROPE=1"
sm120="$direct GGML_CUDA_FATTN_SM120_CAUSAL_SCHEDULE=1"
tma_inplace='GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma-inplace GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted'
cases=(
    'baseline|bitwise|GGML_CUDA_MOE_MMQ_DISABLE=1'
    'sweet-spot|bitwise|GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=canonical'
    "direct-causal|bitwise|GGML_CUDA_MOE_MMQ_DISABLE=1 $direct"
    "direct-causal-sweet-spot|bitwise|GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=canonical $direct"
    "tma-inplace|bitwise|$tma_inplace"
    "direct-tma-inplace|bitwise|$direct $tma_inplace"
    "direct-tma-inplace-tuned|bitwise|$direct $tma_inplace GGML_CUDA_MOE_MMQ_W13_TILE_ROWS=64 GGML_CUDA_MOE_MMQ_W2_OUTPUT_TILE_MAJOR=1"
    "direct-tma-inplace-tuned-norm|bitwise|$direct $tma_inplace GGML_CUDA_MOE_MMQ_W13_TILE_ROWS=64 GGML_CUDA_MOE_MMQ_W2_OUTPUT_TILE_MAJOR=1 GGML_CUDA_ADD_RMS_NORM_FUSION=1"
    "combined-bitwise|bitwise|$direct $tma_inplace GGML_CUDA_ADD_RMS_NORM_FUSION=1"
    "q-rope-ceiling|non-bitwise|$q_rope $tma_inplace GGML_CUDA_ADD_RMS_NORM_FUSION=1"
    "sm120-ceiling|non-bitwise|$sm120 $tma_inplace GGML_CUDA_ADD_RMS_NORM_FUSION=1"
    "sm120-ceiling-tuned|non-bitwise|$sm120 $tma_inplace GGML_CUDA_MOE_MMQ_W13_TILE_ROWS=64 GGML_CUDA_MOE_MMQ_W2_OUTPUT_TILE_MAJOR=1 GGML_CUDA_ADD_RMS_NORM_FUSION=1"
    "full-ceiling|non-bitwise|$q_rope GGML_CUDA_FATTN_SM120_CAUSAL_SCHEDULE=1 $tma_inplace GGML_CUDA_ADD_RMS_NORM_FUSION=1"
    "full-ceiling-tuned|non-bitwise|$q_rope GGML_CUDA_FATTN_SM120_CAUSAL_SCHEDULE=1 $tma_inplace GGML_CUDA_MOE_MMQ_W13_TILE_ROWS=64 GGML_CUDA_MOE_MMQ_W2_OUTPUT_TILE_MAJOR=1 GGML_CUDA_ADD_RMS_NORM_FUSION=1"
)

run_selected() {
    local label=$1
    [[ "$selected" == ',all,' || "$selected" == *",$label,"* ]]
}

for entry in "${cases[@]}"; do
    label=${entry%%|*}
    remainder=${entry#*|}
    validation=${remainder%%|*}
    environment=${remainder#*|}
    if ! run_selected "$label"; then
        continue
    fi
    read -r -a env_args <<< "$environment"
    result="$label-pp8192.jsonl"
    env "${blackwell_prefill_clean_env[@]}" GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
        LLAMA_KQ_MASK_CONTIGUOUS_LOG=1 LLAMA_CUDA_FATTN_DIRECT_CAUSAL_LOG=1 GGML_CUDA_FATTN_LOG_CONFIG=1 GGML_CUDA_ADD_RMS_NORM_LOG=1 \
        "${env_args[@]}" "$bench" \
        -m "$model" \
        -p 8192 \
        -n 0 \
        -r "$repetitions" \
        -t "$threads" \
        -ngl 999 \
        -b 8192 \
        -ub "$ubatches" \
        -fa on \
        -o jsonl \
        --progress \
        "${extra_args[@]}" \
        > "$run_dir/$result" \
        2> "$run_dir/$label.stderr"
    printf '%s\t%s\t%s\t%s\n' \
        "$label" "$validation" "$environment" "$result" >> "$run_dir/manifest.tsv"
done

"$python_bin" "$script_dir/summarize_prefill_matrix.py" "$run_dir" > "$run_dir/summary.md"
printf '%s\n' "$run_dir"
