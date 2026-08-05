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

python_bin=${PYTHON:-}
if [[ -z "$python_bin" ]]; then
    if [[ -x /root/miniconda3/bin/python ]]; then
        python_bin=/root/miniconda3/bin/python
    else
        python_bin=python3
    fi
fi
nsys_bin=${NSYS_BIN:-nsys}
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "python interpreter does not exist: $python_bin" >&2
    exit 2
fi
if ! command -v "$nsys_bin" >/dev/null 2>&1; then
    echo "nsys does not exist: $nsys_bin" >&2
    exit 2
fi

tokens=${PREFILL_CUTLASS_NSYS_TOKENS:-8192}
batch=${PREFILL_CUTLASS_NSYS_BATCH:-8192}
ubatch=${PREFILL_CUTLASS_NSYS_UBATCH:-8192}
threads=${PREFILL_CUTLASS_NSYS_THREADS:-25}
selected=,${PREFILL_CUTLASS_NSYS_CASES:-native-tuned,cutlass-support-legacy,cutlass-support-prefix,cutlass-support-prefix-quant,cutlass-support-full},
selected=${selected// /,}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-cutlass-moe-nsys"
mkdir -p "$run_dir/cases"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
"$nsys_bin" --version > "$run_dir/nsys-version.txt" 2>&1
printf 'label\tvalidation\tenvironment\trun_dir\tbackend\tpdl\tw13_tile\tw13_swap\tw2_tile\tw2_swap\n' \
    > "$run_dir/manifest.tsv"

common='LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1 GGML_CUDA_ADD_RMS_NORM_FUSION=1'
native="$common GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma-inplace GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted GGML_CUDA_MOE_MMQ_W13_TILE_ROWS=64 GGML_CUDA_MOE_MMQ_W2_OUTPUT_TILE_MAJOR=1"
cutlass="$common GGML_CUDA_MOE_MMQ_BACKEND=cutlass GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full"
cutlass_tuned="$cutlass GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N=32 GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB=1 GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N=64 GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB=1"
cases=(
    "native-tuned|bitwise|native|-|-|-|-|-|$native"
    "cutlass-support-legacy|metrics-only|cutlass|0|32|1|64|1|$cutlass_tuned GGML_CUDA_MOE_MMQ_CUTLASS_PREFIX_DISABLE=1 GGML_CUDA_MOE_MMQ_CUTLASS_CTA_QUANT_DISABLE=1 GGML_CUDA_MOE_MMQ_CUTLASS_CTA_ACTIVATION_DISABLE=1"
    "cutlass-support-prefix|metrics-only|cutlass|0|32|1|64|1|$cutlass_tuned GGML_CUDA_MOE_MMQ_CUTLASS_CTA_QUANT_DISABLE=1 GGML_CUDA_MOE_MMQ_CUTLASS_CTA_ACTIVATION_DISABLE=1"
    "cutlass-support-prefix-quant|metrics-only|cutlass|0|32|1|64|1|$cutlass_tuned GGML_CUDA_MOE_MMQ_CUTLASS_CTA_ACTIVATION_DISABLE=1"
    "cutlass-support-full|metrics-only|cutlass|0|32|1|64|1|$cutlass_tuned"
    "cutlass-activation-r1|metrics-only|cutlass|0|32|1|64|1|$cutlass_tuned GGML_CUDA_MOE_MMQ_CUTLASS_ACTIVATION_ROWS=1"
    "cutlass-activation-r4|metrics-only|cutlass|0|32|1|64|1|$cutlass_tuned GGML_CUDA_MOE_MMQ_CUTLASS_ACTIVATION_ROWS=4"
    "cutlass-activation-r8|metrics-only|cutlass|0|32|1|64|1|$cutlass_tuned GGML_CUDA_MOE_MMQ_CUTLASS_ACTIVATION_ROWS=8"
    "cutlass-n32-64|metrics-only|cutlass|0|32|1|64|1|$cutlass GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N=32 GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB=1 GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N=64 GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB=1"
    "cutlass-n32|metrics-only|cutlass|0|32|1|32|1|$cutlass GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N=32 GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB=1 GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N=32 GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB=1"
    "cutlass-n64|metrics-only|cutlass|0|64|1|64|1|$cutlass GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N=64 GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB=1 GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N=64 GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB=1"
    "cutlass-n128|metrics-only|cutlass|0|128|1|128|1|$cutlass GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N=128 GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB=1 GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N=128 GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB=1"
    "cutlass-n128-no-swap|metrics-only|cutlass|0|128|0|128|0|$cutlass GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N=128 GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB=0 GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N=128 GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB=0"
)

run_selected() {
    local label=$1
    [[ "$selected" == ',all,' || "$selected" == *",$label,"* ]]
}

require_log() {
    local case_dir=$1
    local pattern=$2
    local message=$3
    if ! grep -q "$pattern" "$case_dir/llama-bench.stderr"; then
        echo "$message" >&2
        exit 1
    fi
}

selected_count=0
for entry in "${cases[@]}"; do
    IFS='|' read -r label validation backend pdl w13_tile w13_swap w2_tile w2_swap environment <<< "$entry"
    if ! run_selected "$label"; then
        continue
    fi
    selected_count=$((selected_count + 1))
    read -r -a env_args <<< "$environment"
    profile_output=$(env "${blackwell_prefill_clean_env[@]}" "${env_args[@]}" \
        NSYS_BIN="$nsys_bin" \
        PYTHON="$python_bin" \
        PREFILL_TOKENS="$tokens" \
        PREFILL_BATCH="$batch" \
        PREFILL_UBATCH="$ubatch" \
        PREFILL_THREADS="$threads" \
        PREFILL_REPETITIONS=1 \
        PREFILL_PROFILE_WARMUP=1 \
        GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
        LLAMA_CUDA_FATTN_DIRECT_CAUSAL_LOG=1 \
        GGML_CUDA_ADD_RMS_NORM_LOG=1 \
        bash "$script_dir/profile_llama.sh" \
        "$label" "$bench" "$model" "$run_dir/cases" -v "${extra_args[@]}")
    case_dir=${profile_output##*$'\n'}

    if [[ ! -f "$case_dir/profile.nsys-rep" ]]; then
        echo "$label did not produce an Nsys report" >&2
        exit 1
    fi
    if ! grep -q 'ffn_moe' "$case_dir/nvtx-gpu.csv"; then
        echo "$label did not record MoE NVTX ranges; rebuild with GGML_CUDA_MOE_PROFILE=ON" >&2
        exit 1
    fi

    if [[ "$backend" == cutlass ]]; then
        require_log "$case_dir" 'backend=cutlass' "$label did not select the CUTLASS backend"
        require_log "$case_dir" 'cutlass-fusion=full' "$label did not select full CUTLASS fusion"
        require_log "$case_dir" "cutlass-pdl=$pdl" "$label did not select PDL=$pdl"
        require_log "$case_dir" \
            "cutlass={w13-tile-n=$w13_tile,w13-swap=$w13_swap,w2-tile-n=$w2_tile,w2-swap=$w2_swap}" \
            "$label did not select the requested CUTLASS kernels"
        case "$label" in
            cutlass-support-legacy)
                require_log "$case_dir" 'cutlass-prefix=0 cutlass-cta-quant=0 cutlass-cta-activation=0' \
                    "$label did not disable the new support kernels"
                ;;
            cutlass-support-prefix)
                require_log "$case_dir" 'cutlass-prefix=1 cutlass-cta-quant=0 cutlass-cta-activation=0' \
                    "$label did not isolate the prefix scheduler"
                ;;
            cutlass-support-prefix-quant)
                require_log "$case_dir" 'cutlass-prefix=1 cutlass-cta-quant=1 cutlass-cta-activation=0' \
                    "$label did not isolate prefix scheduling and CTA quantization"
                ;;
            cutlass-support-full)
                require_log "$case_dir" 'cutlass-prefix=1 cutlass-cta-quant=1 cutlass-cta-activation=1' \
                    "$label did not enable the complete support pipeline"
                ;;
        esac
        if ! grep -q 'ffn_moe.cutlass_w13' "$case_dir/nvtx-gpu.csv" ||
           ! grep -q 'ffn_moe.cutlass_w2' "$case_dir/nvtx-gpu.csv"; then
            echo "$label did not record CUTLASS W13 and W2 ranges" >&2
            exit 1
        fi
    else
        require_log "$case_dir" 'backend=native' "$label did not select the native backend"
        require_log "$case_dir" 'weights=tma-inplace' "$label did not select in-place TMA weights"
    fi

    relative_dir=${case_dir#"$run_dir/"}
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$validation" "$environment" "$relative_dir" "$backend" "$pdl" \
        "$w13_tile" "$w13_swap" "$w2_tile" "$w2_swap" >> "$run_dir/manifest.tsv"
done

if [[ $selected_count -eq 0 ]]; then
    echo "PREFILL_CUTLASS_NSYS_CASES did not select a known case" >&2
    exit 2
fi

"$python_bin" "$script_dir/summarize_cutlass_nsys.py" \
    "$run_dir" --csv "$run_dir/cutlass-stages.csv" > "$run_dir/summary.md"
"$python_bin" "$script_dir/summarize_nsys.py" "$run_dir" > "$run_dir/kernel-details.md"
printf '%s\n' "$run_dir"
