#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 TEST_BACKEND_OPS OUT_DIR" >&2
    exit 2
fi

test_backend_ops=$1
out_dir=$2
ncu_bin=${NCU_BIN:-ncu}

if [[ ! -x "$test_backend_ops" ]]; then
    echo "test-backend-ops is not executable: $test_backend_ops" >&2
    exit 2
fi
if ! command -v "$ncu_bin" >/dev/null 2>&1; then
    echo "Nsight Compute was not found: $ncu_bin" >&2
    exit 2
fi

read -r -a modes <<< "${NCU_MOE_TMA_MODES:-cooperative warp-specialized}"
read -r -a stages <<< "${NCU_MOE_TMA_STAGES:-w13 w2}"
read -r -a token_counts <<< "${NCU_MOE_TMA_TOKENS:-2048 8192}"
read -r -a distributions <<< "${NCU_MOE_TMA_DISTRIBUTIONS:-uniform skewed}"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-moe-tma-ncu"
mkdir -p "$run_dir"

nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
"$ncu_bin" --version > "$run_dir/ncu-version.txt" 2>&1
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
printf 'mode\tstage\ttokens\tdistribution\treport\tcsv\n' > "$run_dir/manifest.tsv"

sections=(
    SpeedOfLight
    ComputeWorkloadAnalysis
    MemoryWorkloadAnalysis
    Occupancy
    LaunchStats
    SchedulerStats
    WarpStateStats
    InstructionStats
)
section_args=()
for section in "${sections[@]}"; do
    section_args+=(--section "$section")
done

metric_args=()
if [[ -n ${NCU_EXTRA_METRICS:-} ]]; then
    metric_args+=(--metrics "$NCU_EXTRA_METRICS")
fi

for mode in "${modes[@]}"; do
    case "$mode" in
        cooperative) mode_args=() ;;
        warp-specialized) mode_args=(GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1) ;;
        *) echo "invalid TMA mode: $mode" >&2; exit 2 ;;
    esac

    for stage in "${stages[@]}"; do
        case "$stage" in
            w13) launch_skip=0 ;;
            w2) launch_skip=1 ;;
            *) echo "invalid MoE stage: $stage" >&2; exit 2 ;;
        esac

        for tokens in "${token_counts[@]}"; do
            for distribution in "${distributions[@]}"; do
                case "$distribution" in
                    uniform) skewed=0 ;;
                    skewed) skewed=1 ;;
                    *) echo "invalid expert distribution: $distribution" >&2; exit 2 ;;
                esac

                label="$mode-$stage-pp$tokens-$distribution"
                report="$run_dir/$label"
                csv="$run_dir/$label.csv"

                env GGML_CUDA_DISABLE_GRAPHS=1 \
                    GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma \
                    GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 \
                    GGML_CUDA_MOE_MMQ_DISTRIBUTION_TEST=1 \
                    GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
                    "${mode_args[@]}" \
                    "$ncu_bin" \
                    --target-processes all \
                    --replay-mode kernel \
                    --cache-control none \
                    --clock-control none \
                    --kernel-name 'regex:moe_tma_persistent.*' \
                    --launch-skip-before-match "$launch_skip" \
                    --launch-count 1 \
                    --apply-rules yes \
                    --page raw \
                    --csv \
                    --log-file "$csv" \
                    --export "$report" \
                    "${section_args[@]}" \
                    "${metric_args[@]}" \
                    "$test_backend_ops" perf -b CUDA0 -o MOE_MMQ \
                    -p "n_token=$tokens.*skewed_ids=$skewed.*n_embd=2880" \
                    > "$run_dir/$label.stdout" \
                    2> "$run_dir/$label.stderr"

                printf '%s\t%s\t%s\t%s\t%s.ncu-rep\t%s\n' \
                    "$mode" "$stage" "$tokens" "$distribution" "$report" "$csv" >> "$run_dir/manifest.tsv"
            done
        done
    done
done

python_bin=${PYTHON:-python3}
"$python_bin" "$script_dir/summarize_ncu.py" "$run_dir" > "$run_dir/summary.md"
printf '%s\n' "$run_dir"
