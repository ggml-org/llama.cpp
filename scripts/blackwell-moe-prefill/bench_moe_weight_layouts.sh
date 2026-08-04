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

read -r -a layouts <<< "${MOE_WEIGHT_LAYOUTS:-canonical interleaved split}"
tokens=${MOE_WEIGHT_TOKENS:-2048|8192}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-moe-weight-layouts"
mkdir -p "$run_dir"

nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
printf 'layout\tcorrectness\tperformance\n' > "$run_dir/manifest.tsv"

for layout in "${layouts[@]}"; do
    case "$layout" in
        canonical|interleaved|split) ;;
        *) echo "invalid weight layout: $layout" >&2; exit 2 ;;
    esac

    if [[ ${MOE_WEIGHT_VALIDATE:-1} == 1 ]]; then
        env GGML_CUDA_MOE_MMQ_TEST=1 \
            GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT="$layout" \
            GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
            "$test_backend_ops" test -b CUDA0 -o MOE_MMQ \
            > "$run_dir/$layout-correctness.txt" \
            2> "$run_dir/$layout-correctness.stderr"
    fi

    env GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT="$layout" \
        GGML_CUDA_MOE_MMQ_DISTRIBUTION_TEST=1 \
        GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
        "$test_backend_ops" perf -b CUDA0 -o MOE_MMQ \
        -p "n_token=($tokens).*n_embd=2880" \
        > "$run_dir/$layout-perf.txt" \
        2> "$run_dir/$layout-perf.stderr"

    printf '%s\t%s\t%s\n' "$layout" "$layout-correctness.txt" "$layout-perf.txt" >> "$run_dir/manifest.tsv"
done

printf '%s\n' "$run_dir"
