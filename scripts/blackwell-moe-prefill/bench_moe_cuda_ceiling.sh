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
        echo "executable does not exist: $executable" >&2
        exit 2
    fi
done
if [[ ! -f "$model" ]]; then
    echo "model does not exist: $model" >&2
    exit 2
fi

threads=${PREFILL_THREADS:-25}
repetitions=${PREFILL_REPETITIONS:-3}
tokens=${MOE_CEILING_TOKENS:-2048|8192}
read -r -a ubatches <<< "${MOE_CEILING_UBATCHES:-2048 8192}"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-moe-cuda-ceiling"
mkdir -p "$run_dir"

nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
printf 'case\tcorrectness\tmicrobenchmark\n' > "$run_dir/micro-manifest.tsv"
printf 'case\tubatch\tresult\n' > "$run_dir/model-manifest.tsv"

run_micro() {
    local label=$1
    shift

    env GGML_CUDA_MOE_MMQ_TEST=1 \
        GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
        "$@" \
        "$test_backend_ops" test -b CUDA0 -o MOE_MMQ \
        > "$run_dir/$label-correctness.txt" \
        2> "$run_dir/$label-correctness.stderr"

    env GGML_CUDA_MOE_MMQ_DISTRIBUTION_TEST=1 \
        GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
        "$@" \
        "$test_backend_ops" perf -b CUDA0 -o MOE_MMQ \
        -p "n_token=($tokens).*n_embd=2880" \
        > "$run_dir/$label-perf.txt" \
        2> "$run_dir/$label-perf.stderr"

    if ! grep -q 'MOE_MMQ' "$run_dir/$label-correctness.txt" ||
            ! grep -q 'MOE_MMQ' "$run_dir/$label-perf.txt"; then
        echo "no MoE MMQ case matched for $label" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\n' \
        "$label" "$label-correctness.txt" "$label-perf.txt" >> "$run_dir/micro-manifest.tsv"
}

run_model() {
    local label=$1
    local ubatch=$2
    shift 2

    env LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1 \
        GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
        "$@" \
        "$bench" \
        -m "$model" \
        -p 8192 \
        -n 0 \
        -r "$repetitions" \
        -t "$threads" \
        -ngl 999 \
        -b 8192 \
        -ub "$ubatch" \
        -fa on \
        -o jsonl \
        "${extra_args[@]}" \
        > "$run_dir/$label-ub$ubatch.jsonl" \
        2> "$run_dir/$label-ub$ubatch.stderr"

    printf '%s\t%s\t%s\n' \
        "$label" "$ubatch" "$label-ub$ubatch.jsonl" >> "$run_dir/model-manifest.tsv"
}

run_micro canonical GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=canonical
run_micro tma-cooperative \
    GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma \
    GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1
run_micro tma-warp-specialized \
    GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma \
    GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 \
    GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1

for ubatch in "${ubatches[@]}"; do
    run_model canonical "$ubatch" \
        GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=canonical \
        GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused-quant
    run_model tma-cooperative "$ubatch" \
        GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma \
        GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 \
        GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused-quant
    run_model tma-warp-specialized "$ubatch" \
        GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma \
        GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 \
        GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 \
        GGML_CUDA_MOE_MMQ_W13_EPILOGUE=fused-quant
done

printf '%s\n' "$run_dir"
