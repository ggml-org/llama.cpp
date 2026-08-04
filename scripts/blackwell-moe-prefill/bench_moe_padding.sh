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

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-moe-padding"
mkdir -p "$run_dir"
nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"

if [[ ${MOE_PADDING_VALIDATE:-0} == 1 ]]; then
    env GGML_CUDA_MOE_MMQ_TEST=1 GGML_CUDA_MOE_MMQ_PADDED_TEST=1 \
        "$test_backend_ops" test -b CUDA0 -o MOE_MMQ \
        > "$run_dir/correctness.txt" \
        2> "$run_dir/correctness.stderr"
fi

env GGML_CUDA_MOE_MMQ_PADDED_TEST=1 \
    "$test_backend_ops" perf -b CUDA0 -o MUL_MAT_ID \
    -p 'type_a=mxfp4.*n_mats=128.*n_used=4.*m=(2880|2944|5760|5888).*n=(2048|8192).*k=(2880|2944)' \
    > "$run_dir/mul-mat-id.txt" \
    2> "$run_dir/mul-mat-id.stderr"

env GGML_CUDA_MOE_MMQ_PADDED_TEST=1 \
    "$test_backend_ops" perf -b CUDA0 -o MOE_MMQ \
    -p 'n_token=(2048|8192).*n_embd=(2880|2944)' \
    > "$run_dir/moe-mmq.txt" \
    2> "$run_dir/moe-mmq.stderr"

printf '%s\n' "$run_dir"
