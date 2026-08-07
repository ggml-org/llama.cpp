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
run_dir="$out_dir/$stamp-cutlass-nvfp4-decode-validation"
mkdir -p "$run_dir"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
printf 'label\ttest\toutput\n' > "$run_dir/manifest.tsv"

common_environment=(
    GGML_CUDA_DISABLE_GRAPHS=1
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE=1
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_LOG=1
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0
)

run_test() {
    local label=$1
    local test_name=$2
    local output=$3
    shift 3

    env "${blackwell_prefill_clean_env[@]}" \
        -u GGML_CUDA_DISABLE_GRAPHS \
        "${common_environment[@]}" \
        "$@" \
        "$test_bin" test -b CUDA0 -o "$test_name" "${extra_args[@]}" \
        > "$run_dir/$label.stdout" \
        2> "$run_dir/$label.stderr"

    local tokens_list=(1 2 4 8)
    local shapes=('K=2048 N=512' 'K=2048 N=1024' 'K=512 N=2048')
    if [[ "$test_name" == MOE_NVFP4_BLOCK ]]; then
        tokens_list=(1 8)
        shapes=('K=2048 N=512' 'K=512 N=2048')
    fi
    for tokens in "${tokens_list[@]}"; do
        for shape in "${shapes[@]}"; do
            if ! grep -q "MoE CUTLASS NVFP4 decode dispatch: tokens=$tokens .* $shape output=$output" \
                    "$run_dir/$label.stderr"; then
                echo "$label did not execute tokens=$tokens with $shape and output=$output" >&2
                exit 1
            fi
        done
    done
    printf '%s\t%s\t%s\n' "$label" "$test_name" "$output" >> "$run_dir/manifest.tsv"
}

run_test matmul-bf16 MOE_NVFP4_DECODE bf16 \
    GGML_CUDA_MOE_MMQ_CUTLASS_NVFP4_DECODE_TEST=1
run_test matmul-f32 MOE_NVFP4_DECODE f32 \
    GGML_CUDA_MOE_MMQ_CUTLASS_NVFP4_DECODE_TEST=1 \
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_OUTPUT_F32=1
run_test block-bf16 MOE_NVFP4_BLOCK bf16 \
    GGML_CUDA_MOE_MMQ_CUTLASS_NVFP4_DECODE_BLOCK_TEST=1
run_test block-f32 MOE_NVFP4_BLOCK f32 \
    GGML_CUDA_MOE_MMQ_CUTLASS_NVFP4_DECODE_BLOCK_TEST=1 \
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_OUTPUT_F32=1

env "${blackwell_prefill_clean_env[@]}" \
    -u GGML_CUDA_DISABLE_GRAPHS \
    "${common_environment[@]}" \
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_FUSED=1 \
    GGML_CUDA_MOE_MMQ_CUTLASS_NVFP4_DECODE_FUSED_TEST=1 \
    "$test_bin" test -b CUDA0 -o MOE_NVFP4_BLOCK "${extra_args[@]}" \
    > "$run_dir/block-fused.stdout" \
    2> "$run_dir/block-fused.stderr"
if ! grep -q 'MoE CUTLASS NVFP4 fused decode dispatch:' "$run_dir/block-fused.stderr"; then
    echo "block-fused did not execute the fused CUTLASS decode path" >&2
    exit 1
fi
printf '%s\t%s\t%s\n' block-fused MOE_NVFP4_BLOCK bf16 >> "$run_dir/manifest.tsv"

printf '%s\n' "$run_dir"
