#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 TEST_BACKEND_OPS LLAMA_BENCH MODEL OUT_DIR [llama-bench arguments...]" >&2
    exit 2
fi

test_backend_ops=$1
bench=$2
model=$3
out_dir=$4
shift 4
extra_args=("$@")

for executable in "$test_backend_ops" "$bench"; do
    if [[ ! -x "$executable" ]]; then
        echo "not executable: $executable" >&2
        exit 2
    fi
done
if [[ ! -f "$model" ]]; then
    echo "model does not exist: $model" >&2
    exit 2
fi

batch=${PREFILL_BATCH:-8192}
ubatch=${PREFILL_UBATCH:-2048}
threads=${PREFILL_THREADS:-25}
repetitions=${PREFILL_REPETITIONS:-3}
tokens=${PREFILL_TOKENS:-512,2048,8192}
selected=,${MOE_COMPONENT_CASES:-all},
python_bin=${PYTHON:-python3}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-moe-components"
mkdir -p "$run_dir"

nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
printf 'label\tenvironment\n' > "$run_dir/manifest.tsv"

cases=(
    'disabled|GGML_CUDA_MOE_MMQ_DISABLE=1'
    'staged-no-plan|GGML_CUDA_MOE_MMQ_SHARED_PLAN_DISABLE=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=staged GGML_CUDA_MOE_MMQ_W2_EPILOGUE=staged'
    'plan-staged|GGML_CUDA_MOE_MMQ_PERSISTENT_DISABLE=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=staged GGML_CUDA_MOE_MMQ_W2_EPILOGUE=staged'
    'w13-epilogue-fused|GGML_CUDA_MOE_MMQ_PERSISTENT_DISABLE=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused GGML_CUDA_MOE_MMQ_W2_EPILOGUE=staged'
    'w13-epilogue-quant|GGML_CUDA_MOE_MMQ_PERSISTENT_DISABLE=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused-quant GGML_CUDA_MOE_MMQ_W2_EPILOGUE=staged'
    'w2-epilogue-fused|GGML_CUDA_MOE_MMQ_PERSISTENT_DISABLE=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused GGML_CUDA_MOE_MMQ_W2_EPILOGUE=fused'
    'w13-persistent|GGML_CUDA_MOE_MMQ_W2_PERSISTENT_DISABLE=1'
    'w2-persistent|GGML_CUDA_MOE_MMQ_W13_PERSISTENT_DISABLE=1'
    'persistent|'
    'persistent-w13-epilogue-quant|GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused-quant'
    'persistent-tma|GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1'
    'persistent-tma-warp|GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1'
    'persistent-tma-full-k|GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 GGML_CUDA_MOE_MMQ_TMA_TAIL_DISABLE=1'
    'cuda-ceiling|LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1 GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused-quant'
    'async-repack|GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 GGML_CUDA_MOE_MMQ_REPACK_ASYNC=1 GGML_CUDA_MOE_MMQ_REPACK_CACHE_ENTRIES=8 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused-quant'
    'w13-tma-epilogue|GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue'
    'w2-tma-weighted|GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused-quant GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted'
    'w2-tma-atomic|GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused-quant GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-atomic'
    'mxfp8-tma-weighted|GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue GGML_CUDA_MOE_MMQ_ACTIVATION_FORMAT=mxfp8 GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted'
    'mxfp8-tma-atomic|GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue GGML_CUDA_MOE_MMQ_ACTIVATION_FORMAT=mxfp8 GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-atomic'
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

    env_args=()
    if [[ -n $environment ]]; then
        read -r -a env_args <<< "$environment"
    fi
    printf '%s\t%s\n' "$label" "$environment" >> "$run_dir/manifest.tsv"

    if [[ ${MOE_COMPONENT_VALIDATE:-0} == 1 ]]; then
        env GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MOE_MMQ_TEST=1 GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 "${env_args[@]}" \
            "$test_backend_ops" test -b CUDA0 -o MOE_MMQ \
            > "$run_dir/$label-correctness.txt" \
            2> "$run_dir/$label-correctness.stderr"
    fi

    env GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 "${env_args[@]}" "$bench" \
        -m "$model" \
        -p "$tokens" \
        -n 0 \
        -r "$repetitions" \
        -t "$threads" \
        -ngl 999 \
        -b "$batch" \
        -ub "$ubatch" \
        -fa on \
        -o jsonl \
        "${extra_args[@]}" \
        > "$run_dir/$label.jsonl" \
        2> "$run_dir/$label.stderr"
done

"$python_bin" "$script_dir/summarize_components.py" "$run_dir" > "$run_dir/summary.md"
printf '%s\n' "$run_dir"
