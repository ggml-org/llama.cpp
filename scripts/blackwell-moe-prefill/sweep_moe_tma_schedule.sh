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

read -r -a stages <<< "${MOE_TMA_STAGES:-w13 w2}"
read -r -a tiles <<< "${MOE_TMA_TILE_ROWS:-32 64 128}"
read -r -a multipliers <<< "${MOE_TMA_CTA_MULTIPLIERS:-1 2}"
read -r -a orders <<< "${MOE_TMA_WORK_ORDERS:-token output}"
tokens=${MOE_TMA_SCHEDULE_TOKENS:-8192}
layout=${MOE_TMA_SCHEDULE_LAYOUT:-tma-inplace}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-moe-tma-schedule"
mkdir -p "$run_dir"

nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
printf 'stage\ttile_rows\tcta_multiplier\twork_order\tresult\n' > "$run_dir/manifest.tsv"

for stage in "${stages[@]}"; do
    case "$stage" in
        w13) prefix=GGML_CUDA_MOE_MMQ_W13 ;;
        w2)  prefix=GGML_CUDA_MOE_MMQ_W2 ;;
        *) echo "invalid MoE stage: $stage" >&2; exit 2 ;;
    esac

    for tile in "${tiles[@]}"; do
        for multiplier in "${multipliers[@]}"; do
            for order in "${orders[@]}"; do
                label="$stage-tile$tile-cta$multiplier-$order"
                env_args=(
                    "${prefix}_TILE_ROWS=$tile"
                    "${prefix}_CTA_MULTIPLIER=$multiplier"
                )
                if [[ $order == output ]]; then
                    env_args+=("${prefix}_OUTPUT_TILE_MAJOR=1")
                fi

                env GGML_CUDA_DISABLE_GRAPHS=1 \
                    GGML_CUDA_MOE_MMQ_DISTRIBUTION_TEST=1 \
                    GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
                    GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT="$layout" \
                    GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 \
                    GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 \
                    GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue \
                    GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted \
                    "${env_args[@]}" \
                    "$test_backend_ops" perf -b CUDA0 -o MOE_MMQ \
                    -p "n_token=($tokens).*n_embd=2880" \
                    > "$run_dir/$label.txt" \
                    2> "$run_dir/$label.stderr"

                printf '%s\t%s\t%s\t%s\t%s.txt\n' \
                    "$stage" "$tile" "$multiplier" "$order" "$label" >> "$run_dir/manifest.tsv"
            done
        done
    done
done

printf '%s\n' "$run_dir"
