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

threads=${PREFILL_THREADS:-25}
repetitions=${PREFILL_REPETITIONS:-3}
read -r -a prompts <<< "${MOE_CEILING_PROMPTS:-512 2048 8192}"
read -r -a ubatches <<< "${MOE_CEILING_UBATCHES:-2048 8192}"
selected=,${MOE_CEILING_CASES:-all},
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-moe-vllm-ceiling"
mkdir -p "$run_dir"

nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
printf 'label\tprompt\tubatch\tenvironment\tresult\n' > "$run_dir/manifest.tsv"

common='LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1 GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1'
tma_inplace='LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1 GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma-inplace GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1'
cases=(
    'disabled|GGML_CUDA_MOE_MMQ_DISABLE=1'
    "tma-full-k|$common GGML_CUDA_MOE_MMQ_TMA_TAIL_DISABLE=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused-quant GGML_CUDA_MOE_MMQ_W2_EPILOGUE=fused"
    "tma-fp4|$common GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused-quant GGML_CUDA_MOE_MMQ_W2_EPILOGUE=fused"
    "async-repack|$common GGML_CUDA_MOE_MMQ_REPACK_ASYNC=1 GGML_CUDA_MOE_MMQ_REPACK_CACHE_ENTRIES=8 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused-quant GGML_CUDA_MOE_MMQ_W2_EPILOGUE=fused"
    "w13-epilogue|$common GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue GGML_CUDA_MOE_MMQ_W2_EPILOGUE=fused"
    "w2-weighted|$common GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused-quant GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted"
    "fp4-all-full-k|$common GGML_CUDA_MOE_MMQ_TMA_TAIL_DISABLE=1 GGML_CUDA_MOE_MMQ_REPACK_ASYNC=1 GGML_CUDA_MOE_MMQ_REPACK_CACHE_ENTRIES=8 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted"
    "fp4-all|$common GGML_CUDA_MOE_MMQ_REPACK_ASYNC=1 GGML_CUDA_MOE_MMQ_REPACK_CACHE_ENTRIES=8 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted"
    "fp4-atomic|$common GGML_CUDA_MOE_MMQ_REPACK_ASYNC=1 GGML_CUDA_MOE_MMQ_REPACK_CACHE_ENTRIES=8 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-atomic"
    "mxfp8-weighted|$common GGML_CUDA_MOE_MMQ_REPACK_ASYNC=1 GGML_CUDA_MOE_MMQ_REPACK_CACHE_ENTRIES=8 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue GGML_CUDA_MOE_MMQ_ACTIVATION_FORMAT=mxfp8 GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted"
    "mxfp8-atomic|$common GGML_CUDA_MOE_MMQ_REPACK_ASYNC=1 GGML_CUDA_MOE_MMQ_REPACK_CACHE_ENTRIES=8 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue GGML_CUDA_MOE_MMQ_ACTIVATION_FORMAT=mxfp8 GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-atomic"
    "tma-inplace-fp4|$tma_inplace GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused-quant GGML_CUDA_MOE_MMQ_W2_EPILOGUE=fused"
    "tma-inplace-full|$tma_inplace GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted"
)

run_selected() {
    local label=$1
    [[ "$selected" == ',all,' || "$selected" == *",$label,"* ]]
}

for entry in "${cases[@]}"; do
    label=${entry%%|*}
    environment=${entry#*|}
    if ! run_selected "$label"; then
        continue
    fi
    read -r -a env_args <<< "$environment"

    for prompt in "${prompts[@]}"; do
        for ubatch in "${ubatches[@]}"; do
            if ((ubatch > prompt)); then
                continue
            fi
            result="$label-pp$prompt-ub$ubatch.jsonl"
            env GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 "${env_args[@]}" "$bench" \
                -m "$model" \
                -p "$prompt" \
                -n 0 \
                -r "$repetitions" \
                -t "$threads" \
                -ngl 999 \
                -b 8192 \
                -ub "$ubatch" \
                -fa on \
                -o jsonl \
                "${extra_args[@]}" \
                > "$run_dir/$result" \
                2> "$run_dir/$label-pp$prompt-ub$ubatch.stderr"
            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$label" "$prompt" "$ubatch" "$environment" "$result" >> "$run_dir/manifest.tsv"
        done
    done
done

printf '%s\n' "$run_dir"
