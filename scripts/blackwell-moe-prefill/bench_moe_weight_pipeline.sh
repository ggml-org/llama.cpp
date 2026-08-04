#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 TEST_BACKEND_OPS OUT_DIR" >&2
    exit 2
fi

test_backend_ops=$1
out_dir=$2
if [[ ! -x "$test_backend_ops" ]]; then
    echo "test-backend-ops is not executable: $test_backend_ops" >&2
    exit 2
fi

tokens=${MOE_WEIGHT_PIPELINE_TOKENS:-2048|8192}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-moe-weight-pipeline"
mkdir -p "$run_dir"

nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
printf 'case\tcorrectness\tperformance\n' > "$run_dir/manifest.tsv"

run_case() {
    local label=$1
    shift

    if [[ ${MOE_WEIGHT_PIPELINE_VALIDATE:-1} == 1 ]]; then
        env GGML_CUDA_MOE_MMQ_TEST=1 \
            GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
            "$@" \
            "$test_backend_ops" test -b CUDA0 -o MOE_MMQ \
            > "$run_dir/$label-correctness.txt" \
            2> "$run_dir/$label-correctness.stderr"
    fi

    env GGML_CUDA_MOE_MMQ_DISTRIBUTION_TEST=1 \
        GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
        "$@" \
        "$test_backend_ops" perf -b CUDA0 -o MOE_MMQ \
        -p "n_token=($tokens).*n_embd=2880" \
        > "$run_dir/$label-perf.txt" \
        2> "$run_dir/$label-perf.stderr"

    printf '%s\t%s\t%s\n' \
        "$label" "$label-correctness.txt" "$label-perf.txt" >> "$run_dir/manifest.tsv"
}

read -r -a cases <<< "${MOE_WEIGHT_PIPELINE_CASES:-canonical split split-activation-async split-weight-pipeline}"
for case_name in "${cases[@]}"; do
    case "$case_name" in
        canonical)
            run_case canonical GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=canonical
            ;;
        split)
            run_case split GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=split
            ;;
        split-activation-async)
            run_case split-activation-async \
                GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=split \
                GGML_CUDA_MOE_MMQ_CP_ASYNC=1
            ;;
        split-weight-pipeline)
            run_case split-weight-pipeline \
                GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=split \
                GGML_CUDA_MOE_MMQ_WEIGHT_PIPELINE=1
            ;;
        *)
            echo "invalid weight pipeline case: $case_name" >&2
            exit 2
            ;;
    esac
done

printf '%s\n' "$run_dir"
