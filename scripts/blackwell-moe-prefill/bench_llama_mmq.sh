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
run_dir="$out_dir/$stamp-llama-mmq"
mkdir -p "$run_dir"
nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"

run_correctness() {
    local label=$1
    shift
    env GGML_CUDA_MOE_MMQ_TEST=1 "$@" "$test_backend_ops" \
        test \
        -b CUDA0 \
        -o MOE_MMQ \
        > "$run_dir/$label-correctness.txt" \
        2> "$run_dir/$label-correctness.stderr"
}

run_correctness baseline GGML_CUDA_MOE_MMQ_DISABLE=1
run_correctness fused-generic GGML_CUDA_MOE_MMQ_PERSISTENT_DISABLE=1
run_correctness persistent

"$test_backend_ops" perf \
    -b CUDA0 \
    -o MUL_MAT_ID \
    -p 'type_a=mxfp4.*type_b=f32.*n_mats=128.*n_used=4.*b=(0|1).*m=(2880|5760).*n=(512|1024|2048|4096|8192).*k=2880' \
    > "$run_dir/mul-mat-id.txt" \
    2> "$run_dir/mul-mat-id.stderr"

"$test_backend_ops" perf \
    -b CUDA0 \
    -o MOE_MMQ \
    > "$run_dir/moe-mmq.txt" \
    2> "$run_dir/moe-mmq.stderr"

if ! grep -q 'MUL_MAT_ID' "$run_dir/mul-mat-id.txt"; then
    echo "no GPT-OSS MUL_MAT_ID cases matched" >&2
    exit 1
fi
if ! grep -q 'MOE_MMQ' "$run_dir/moe-mmq.txt"; then
    echo "no GPT-OSS fused MoE cases matched" >&2
    exit 1
fi
for label in baseline fused-generic persistent; do
    if ! grep -q 'MOE_MMQ' "$run_dir/$label-correctness.txt"; then
        echo "no GPT-OSS correctness case matched for $label" >&2
        exit 1
    fi
done

printf '%s\n' "$run_dir"
