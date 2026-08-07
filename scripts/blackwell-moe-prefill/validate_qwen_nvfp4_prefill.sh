#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 TEST_BACKEND_OPS OUT_DIR [test-backend-ops arguments...]" >&2
    exit 2
fi

test_bin=$1
out_dir=$2
shift 2
extra_args=("$@")

if [[ ! -x "$test_bin" ]]; then
    echo "test-backend-ops is not executable: $test_bin" >&2
    exit 2
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-qwen-nvfp4-prefill-validation"
mkdir -p "$run_dir"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
printf 'label\tgraphs\n' > "$run_dir/manifest.tsv"

common_environment=(
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass
    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0
    GGML_CUDA_MOE_MMQ_CUTLASS_NVFP4_PREFILL_LOG=1
    GGML_CUDA_MOE_MMQ_CUTLASS_NVFP4_PREFILL_TEST=1
)

run_test() {
    local label=$1
    local graphs=$2
    shift 2

    env "${blackwell_prefill_clean_env[@]}" \
        -u GGML_CUDA_DISABLE_GRAPHS \
        "${common_environment[@]}" \
        "$@" \
        "$test_bin" test -b CUDA0 -o MOE_NVFP4_BLOCK "${extra_args[@]}" \
        > "$run_dir/$label.stdout" \
        2> "$run_dir/$label.stderr"

    if ! grep -q 'MoE CUTLASS NVFP4 prefill dispatch:' "$run_dir/$label.stderr"; then
        echo "$label did not execute the Qwen NVFP4 CUTLASS prefill path" >&2
        exit 1
    fi
    printf '%s\t%s\n' "$label" "$graphs" >> "$run_dir/manifest.tsv"
}

run_test eager off GGML_CUDA_DISABLE_GRAPHS=1
run_test graphs on

printf '%s\n' "$run_dir"
