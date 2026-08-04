#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 LLAMA_CPP_ROOT OUT_DIR" >&2
    exit 2
fi

repo_root=$(realpath "$1")
out_dir=$2
source_file="$repo_root/scripts/blackwell-moe-prefill/moe_weight_repack.cu"
if [[ ! -f "$source_file" ]]; then
    echo "source file does not exist: $source_file" >&2
    exit 2
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-moe-repack"
mkdir -p "$run_dir"
nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"

nvcc=${CUDACXX:-nvcc}
"$nvcc" \
    -O3 \
    -std=c++17 \
    -arch=sm_120a \
    -I "$repo_root" \
    "$source_file" \
    -o "$run_dir/moe-weight-repack"

"$run_dir/moe-weight-repack" \
    --experts "${MOE_REPACK_EXPERTS:-128}" \
    --iterations "${MOE_REPACK_ITERATIONS:-10}" \
    > "$run_dir/results.jsonl"

printf '%s\n' "$run_dir"
