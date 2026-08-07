#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 LLAMA_BENCH MODEL OUT_DIR [llama-bench arguments...]" >&2
    exit 2
fi

bench=$1
model=$2
out_dir=$3
shift 3
extra_args=("$@")

if [[ ! -x "$bench" ]]; then
    echo "llama-bench is not executable: $bench" >&2
    exit 2
fi
if [[ ! -f "$model" ]]; then
    echo "model does not exist: $model" >&2
    exit 2
fi

python_bin=${PYTHON:-python3}
nsys_bin=${NSYS_BIN:-nsys}
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "python interpreter does not exist: $python_bin" >&2
    exit 2
fi
if ! command -v "$nsys_bin" >/dev/null 2>&1; then
    echo "nsys does not exist: $nsys_bin" >&2
    exit 2
fi

tokens=${DECODE_NSYS_TOKENS:-32}
batch=${DECODE_NSYS_BATCH:-512}
ubatch=${DECODE_NSYS_UBATCH:-512}
threads=${DECODE_NSYS_THREADS:-25}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-cutlass-decode-nsys"
mkdir -p "$run_dir/cases"
cutlass_marker='MoE (MMQ CUTLASS|CUTLASS NVFP4) decode dispatch:'

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
"$nsys_bin" --version > "$run_dir/nsys-version.txt" 2>&1
printf 'label\tbackend\tinput_scale\toutput\ttokens\twarmup_tokens\trun_dir\tenvironment\n' \
    > "$run_dir/manifest.tsv"

cutlass_environment=(
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE=1
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_LOG=1
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0
    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full
    GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N=32
    GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N=32
    GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB=1
    GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB=1
)
if [[ -n ${DECODE_NSYS_CASES:-} ]]; then
    read -r -a cases <<< "$DECODE_NSYS_CASES"
else
    cases=(
        native-eager
        cutlass-dynamic-bf16
        cutlass-fused-bf16
        cutlass-calibrated-bf16
        cutlass-calibrated-f32
    )
    if [[ ${DECODE_NSYS_FULL_MATRIX:-0} == 1 ]]; then
        cases+=(
            native-calibrated-eager
            cutlass-dynamic-f32
        )
    fi
fi

for label in "${cases[@]}"; do
    backend=cutlass
    input_scale=off
    output=bf16
    env_args=("${cutlass_environment[@]}")
    case "$label" in
        native-eager)
            backend=native
            output=f32
            env_args=(GGML_CUDA_MOE_MMQ_DISABLE=1)
            ;;
        native-calibrated-eager)
            backend=native
            input_scale=on
            output=f32
            env_args=(GGML_CUDA_MOE_MMQ_DISABLE=1 LLAMA_CUDA_MOE_NVFP4_INPUT_SCALE=1)
            ;;
        cutlass-dynamic-f32)
            output=f32
            env_args+=(GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_OUTPUT_F32=1)
            ;;
        cutlass-fused-bf16)
            env_args+=(GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_FUSED=1)
            ;;
        cutlass-calibrated-bf16)
            input_scale=on
            env_args+=(LLAMA_CUDA_MOE_NVFP4_INPUT_SCALE=1)
            ;;
        cutlass-calibrated-f32)
            input_scale=on
            output=f32
            env_args+=(LLAMA_CUDA_MOE_NVFP4_INPUT_SCALE=1 GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_OUTPUT_F32=1)
            ;;
    esac
    environment=$(printf '%s ' "${env_args[@]}")
    environment=${environment% }
    case_dir="$run_dir/cases/$label"
    report="$case_dir/profile"
    mkdir -p "$case_dir"

    {
        printf 'env '
        printf '%q ' "${env_args[@]}" GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MOE_PROFILE=1 \
            "$bench" -m "$model" -p 0 -n "$tokens" -r 1 -t "$threads" -ngl 999 \
            -b "$batch" -ub "$ubatch" -fa on -v -o jsonl "${extra_args[@]}"
        printf '\n'
    } > "$case_dir/benchmark-command.sh"

    env "${blackwell_prefill_clean_env[@]}" \
        -u GGML_CUDA_DISABLE_GRAPHS \
        "${env_args[@]}" \
        GGML_CUDA_DISABLE_GRAPHS=1 \
        GGML_CUDA_MOE_PROFILE=1 \
        GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
        "$nsys_bin" profile \
            --force-overwrite=true \
            --trace=cuda,nvtx \
            --sample=none \
            --cpuctxsw=none \
            --output="$report" \
            bash -c 'stdout=$1; stderr=$2; shift 2; exec "$@" > "$stdout" 2> "$stderr"' \
                bash "$case_dir/llama-bench.jsonl" "$case_dir/llama-bench.stderr" \
                "$bench" -m "$model" -p 0 -n "$tokens" -r 1 -t "$threads" -ngl 999 \
                -b "$batch" -ub "$ubatch" -fa on -v -o jsonl "${extra_args[@]}" \
            > "$case_dir/nsys.stdout" \
            2> "$case_dir/nsys.stderr"

    if [[ ! -f "$report.nsys-rep" ]]; then
        echo "$label did not produce an Nsys report" >&2
        exit 1
    fi
    if [[ "$backend" == cutlass ]]; then
        if [[ "$label" == cutlass-fused-bf16 ]]; then
            if ! grep -q 'MoE CUTLASS NVFP4 fused decode dispatch:' "$case_dir/llama-bench.stderr"; then
                echo "$label did not execute the fused CUTLASS decode path" >&2
                exit 1
            fi
        elif ! grep -Eq "$cutlass_marker" "$case_dir/llama-bench.stderr"; then
            echo "$label did not execute the CUTLASS decode path" >&2
            exit 1
        elif [[ ${DECODE_NSYS_REQUIRE_DIRECT:-0} == 1 ]] &&
                ! grep -q 'MoE MMQ CUTLASS decode dispatch:.*schedule=direct' "$case_dir/llama-bench.stderr"; then
            echo "$label did not execute the direct-schedule CUTLASS decode path" >&2
            exit 1
        elif grep -q 'MoE CUTLASS NVFP4 decode dispatch:' "$case_dir/llama-bench.stderr"; then
            if ! grep -Eq "MoE CUTLASS NVFP4 decode dispatch: tokens=1 .*K=2048 N=(512|1024).*output=$output" \
                    "$case_dir/llama-bench.stderr"; then
                echo "$label did not execute an NVFP4 W13 shape" >&2
                exit 1
            fi
            if ! grep -q "MoE CUTLASS NVFP4 decode dispatch: tokens=1 .*K=512 N=2048.*output=$output" \
                    "$case_dir/llama-bench.stderr"; then
                echo "$label did not execute the NVFP4 W2 shape" >&2
                exit 1
            fi
        fi
        if [[ ${DECODE_NSYS_EXPECTED_LAYERS:-0} -gt 0 ]]; then
            matched_layers=$(grep 'MoE MMQ CUTLASS decode dispatch:' "$case_dir/llama-bench.stderr" |
                sed -nE 's/.* weight=([^ ]+) .*/\1/p' | sort -u | wc -l)
            if (( matched_layers < DECODE_NSYS_EXPECTED_LAYERS )); then
                echo "$label matched only $matched_layers of $DECODE_NSYS_EXPECTED_LAYERS expected expert layers" >&2
                exit 1
            fi
        fi
    elif grep -Eq "$cutlass_marker" "$case_dir/llama-bench.stderr"; then
        echo "$label unexpectedly executed the CUTLASS decode path" >&2
        exit 1
    fi

    for spec in \
        'cuda_gpu_kern_sum:cuda-kernels.csv' \
        'nvtx_gpu_proj_sum:nvtx-gpu.csv' \
        'cuda_api_sum:cuda-api.csv'; do
        report_name=${spec%%:*}
        output_name=${spec#*:}
        if ! "$nsys_bin" stats --quiet --force-export=true --report "$report_name" --format csv \
                "$report.nsys-rep" > "$case_dir/$output_name" 2> "$case_dir/$output_name.stderr"; then
            echo "failed to export $report_name for $label" >&2
            exit 1
        fi
    done

    relative_dir=${case_dir#"$run_dir/"}
    printf '%s\t%s\t%s\t%s\t%s\t1\t%s\t%s\n' \
        "$label" "$backend" "$input_scale" "$output" "$tokens" "$relative_dir" "$environment" \
        >> "$run_dir/manifest.tsv"
done

"$python_bin" "$script_dir/summarize_decode_nsys.py" "$run_dir" > "$run_dir/summary.md"
printf '%s\n' "$run_dir"
