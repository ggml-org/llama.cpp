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
run_dir="$out_dir/$stamp-cutlass-decode-validation"
mkdir -p "$run_dir"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"

env "${blackwell_prefill_clean_env[@]}" \
    -u GGML_CUDA_DISABLE_GRAPHS \
    GGML_CUDA_DISABLE_GRAPHS=1 \
    GGML_CUDA_MOE_MMQ_TEST=1 \
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_TEST=1 \
    GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass \
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE=1 \
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_LOG=1 \
    GGML_CUDA_MOE_MMQ_CUTLASS_VALIDATE_SUPPORT=1 \
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 \
    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full \
    GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N=32 \
    GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N=32 \
    GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB=1 \
    GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB=1 \
    "$test_bin" test -b CUDA0 -o MOE_MMQ "${extra_args[@]}" \
    > "$run_dir/test.stdout" \
    2> "$run_dir/test.stderr"

if ! grep -q 'MoE MMQ CUTLASS decode dispatch:.*schedule=direct' "$run_dir/test.stderr"; then
    echo "the validation did not execute the CUTLASS decode path" >&2
    exit 1
fi
if ! grep -q 'cutlass-decode=1' "$run_dir/test.stderr"; then
    echo "the validation did not enable CUTLASS decode" >&2
    exit 1
fi

printf '%s\n' "$run_dir"
