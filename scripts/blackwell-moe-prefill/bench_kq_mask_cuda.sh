#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 TEST_BACKEND_OPS OUT_DIR [test-backend-ops arguments...]" >&2
    exit 2
fi

test_backend_ops=$1
out_dir=$2
shift 2
extra_args=("$@")

if [[ ! -x "$test_backend_ops" ]]; then
    echo "test-backend-ops is not executable: $test_backend_ops" >&2
    exit 2
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-kq-mask-cuda"
mkdir -p "$run_dir"

nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true

env GGML_CUDA_KQ_MASK_TEST=1 "$test_backend_ops" \
    perf \
    -b CUDA0 \
    -o FILL,DIAG_MASK_INF \
    "${extra_args[@]}" \
    > "$run_dir/perf.txt" \
    2> "$run_dir/perf.stderr"

if ! grep -q 'FILL' "$run_dir/perf.txt"; then
    echo "no CUDA FILL case was measured" >&2
    exit 1
fi
if ! grep -q 'DIAG_MASK_INF' "$run_dir/perf.txt"; then
    echo "no CUDA DIAG_MASK_INF case was measured" >&2
    exit 1
fi

printf '%s\n' "$run_dir"
