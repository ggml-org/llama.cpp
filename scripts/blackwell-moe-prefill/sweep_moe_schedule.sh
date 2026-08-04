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

read -r -a tiles <<< "${MOE_TILE_ROWS:-32 64 128}"
read -r -a multipliers <<< "${MOE_CTA_MULTIPLIERS:-1 2}"
read -r -a orders <<< "${MOE_WORK_ORDERS:-token output}"
tokens=${MOE_SCHEDULE_TOKENS:-2048|8192}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-moe-schedule"
mkdir -p "$run_dir"
nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
printf 'stage\ttile_rows\tcta_multiplier\twork_order\tfile\n' > "$run_dir/manifest.tsv"

for stage in w13 w2; do
    if [[ $stage == w13 ]]; then
        other_disable=GGML_CUDA_MOE_MMQ_W2_PERSISTENT_DISABLE=1
        prefix=GGML_CUDA_MOE_MMQ_W13
    else
        other_disable=GGML_CUDA_MOE_MMQ_W13_PERSISTENT_DISABLE=1
        prefix=GGML_CUDA_MOE_MMQ_W2
    fi

    for tile in "${tiles[@]}"; do
        for multiplier in "${multipliers[@]}"; do
            for order in "${orders[@]}"; do
                label="$stage-tile$tile-cta$multiplier-$order"
                env_args=(
                    "$other_disable"
                    "${prefix}_TILE_ROWS=$tile"
                    "${prefix}_CTA_MULTIPLIER=$multiplier"
                )
                if [[ $order == output ]]; then
                    env_args+=("${prefix}_OUTPUT_TILE_MAJOR=1")
                fi

                printf '%s\t%s\t%s\t%s\t%s.txt\n' \
                    "$stage" "$tile" "$multiplier" "$order" "$label" >> "$run_dir/manifest.tsv"
                env GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 "${env_args[@]}" \
                    "$test_backend_ops" perf -b CUDA0 -o MOE_MMQ \
                    -p "n_token=($tokens).*n_embd=2880" \
                    > "$run_dir/$label.txt" \
                    2> "$run_dir/$label.stderr"
            done
        done
    done
done

printf '%s\n' "$run_dir"
