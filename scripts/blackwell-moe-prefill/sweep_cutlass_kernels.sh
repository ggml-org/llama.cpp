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
run_dir="$out_dir/$stamp-cutlass-kernel-sweep"
mkdir -p "$run_dir"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"

for w13_swap in 0 1; do
    for w13_tile in 32 64 128; do
        for w2_swap in 0 1; do
            for w2_tile in 32 64 128; do
                label="w13-n-$w13_tile-swap-$w13_swap-w2-n-$w2_tile-swap-$w2_swap"
                env "${blackwell_prefill_clean_env[@]}" \
                    GGML_CUDA_DISABLE_GRAPHS=1 \
                    GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
                    GGML_CUDA_MOE_MMQ_BACKEND=cutlass \
                    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full \
                    GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N="$w13_tile" \
                    GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N="$w2_tile" \
                    GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB="$w13_swap" \
                    GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB="$w2_swap" \
                    "$test_bin" perf -b CUDA0 -o MOE_MMQ "${extra_args[@]}" \
                    > "$run_dir/$label.stdout" \
                    2> "$run_dir/$label.stderr"
                if ! grep -q 'MoE MMQ: backend=cutlass' "$run_dir/$label.stderr"; then
                    echo "$label did not execute the CUTLASS MoE backend" >&2
                    exit 1
                fi
            done
        done
    done
done

printf '%s\n' "$run_dir"
