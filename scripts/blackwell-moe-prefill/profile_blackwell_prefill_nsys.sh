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

tokens=${PREFILL_NSYS_TOKENS:-8192}
batch=${PREFILL_NSYS_BATCH:-8192}
ubatch=${PREFILL_NSYS_UBATCH:-8192}
threads=${PREFILL_NSYS_THREADS:-25}
repetitions=${PREFILL_NSYS_REPETITIONS:-1}
selected=,${PREFILL_NSYS_CASES:-baseline,strict-tuned,full-ceiling-tuned},
selected=${selected// /,}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-blackwell-prefill-nsys"
mkdir -p "$run_dir/cases"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
"$nsys_bin" --version > "$run_dir/nsys-version.txt" 2>&1
printf 'label\tvalidation\tenvironment\trun_dir\n' > "$run_dir/manifest.tsv"

direct='LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1'
tma_inplace='GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma-inplace GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1 GGML_CUDA_MOE_MMQ_W13_EPILOGUE=tma-epilogue GGML_CUDA_MOE_MMQ_W2_EPILOGUE=tma-weighted'
tuned="$direct $tma_inplace GGML_CUDA_MOE_MMQ_W13_TILE_ROWS=64 GGML_CUDA_MOE_MMQ_W2_OUTPUT_TILE_MAJOR=1 GGML_CUDA_ADD_RMS_NORM_FUSION=1"
ceiling="$tuned LLAMA_CUDA_FATTN_Q_ROPE=1 GGML_CUDA_FATTN_SM120_CAUSAL_SCHEDULE=1"
cutlass_gemm='GGML_CUDA_MOE_MMQ_BACKEND=cutlass GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=none'
cutlass_full="$direct GGML_CUDA_MOE_MMQ_BACKEND=cutlass GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full GGML_CUDA_ADD_RMS_NORM_FUSION=1 LLAMA_CUDA_FATTN_Q_ROPE=1 GGML_CUDA_FATTN_SM120_CAUSAL_SCHEDULE=1"
cases=(
    'baseline|bitwise|GGML_CUDA_MOE_MMQ_DISABLE=1'
    "strict-tuned|bitwise|$tuned"
    "full-ceiling-tuned|metrics-only|$ceiling"
    "cutlass-gemm|metrics-only|$cutlass_gemm"
    "cutlass-full|metrics-only|$cutlass_full"
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
    label=${entry%%|*}
    remainder=${entry#*|}
    validation=${remainder%%|*}
    environment=${remainder#*|}
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
        PREFILL_REPETITIONS="$repetitions" \
        PREFILL_PROFILE_WARMUP=0 \
        GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
        LLAMA_CUDA_FATTN_DIRECT_CAUSAL_LOG=1 \
        GGML_CUDA_FATTN_LOG_CONFIG=1 \
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

    case "$label" in
        strict-tuned|full-ceiling-tuned)
            require_log "$case_dir" 'FlashAttention: direct-causal=1' "$label did not select direct causal Attention"
            require_log "$case_dir" 'weights=tma-inplace' "$label did not select in-place TMA MoE"
            require_log "$case_dir" 'w13-epilogue=tma-epilogue' "$label did not select the W13 epilogue"
            require_log "$case_dir" 'w2-epilogue=tma-weighted' "$label did not select the W2 epilogue"
            require_log "$case_dir" 'CUDA add RMS norm fusion: enabled' "$label did not select add RMS norm fusion"
            ;;
        cutlass-gemm|cutlass-full)
            require_log "$case_dir" 'backend=cutlass' "$label did not select CUTLASS MoE"
            ;;
    esac
    if [[ "$label" == full-ceiling-tuned ]]; then
        require_log "$case_dir" 'FlashAttention: q-rope=1' "$label did not select Q RoPE fusion"
        require_log "$case_dir" 'FlashAttention: sm120-causal=1' "$label did not select the SM120 causal schedule"
    fi
    if [[ "$label" == cutlass-full ]]; then
        require_log "$case_dir" 'cutlass-fusion=full' "$label did not select full CUTLASS fusion"
        require_log "$case_dir" 'FlashAttention: direct-causal=1' "$label did not select direct causal Attention"
        require_log "$case_dir" 'FlashAttention: q-rope=1' "$label did not select Q RoPE fusion"
        require_log "$case_dir" 'FlashAttention: sm120-causal=1' "$label did not select the SM120 causal schedule"
        require_log "$case_dir" 'CUDA add RMS norm fusion: enabled' "$label did not select add RMS norm fusion"
    fi

    relative_dir=${case_dir#"$run_dir/"}
    printf '%s\t%s\t%s\t%s\n' \
        "$label" "$validation" "$environment" "$relative_dir" >> "$run_dir/manifest.tsv"
done

if [[ $selected_count -eq 0 ]]; then
    echo "PREFILL_NSYS_CASES did not select a known case" >&2
    exit 2
fi

"$python_bin" "$script_dir/summarize_nsys.py" "$run_dir" > "$run_dir/summary.md"
printf '%s\n' "$run_dir"
