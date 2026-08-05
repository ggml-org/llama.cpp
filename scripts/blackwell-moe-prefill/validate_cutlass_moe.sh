#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 LLAMA_DEBUG MODEL OUT_DIR [llama-debug arguments...]" >&2
    exit 2
fi

debug_bin=$1
model=$2
out_dir=$3
shift 3
extra_args=("$@")
python_bin=${PYTHON:-python3}

if [[ ! -x "$debug_bin" ]]; then
    echo "llama-debug is not executable: $debug_bin" >&2
    exit 2
fi
if [[ ! -f "$model" ]]; then
    echo "model does not exist: $model" >&2
    exit 2
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "python interpreter does not exist: $python_bin" >&2
    exit 2
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-cutlass-moe-validation"
prompt_file="$run_dir/prompt.txt"
mkdir -p "$run_dir/reference"
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"

for ((i = 0; i < 1024; ++i)); do
    printf ' hello' >> "$prompt_file"
done
printf '\n' >> "$prompt_file"

run_case() {
    local label=$1
    shift
    mkdir -p "$run_dir/$label"
    env "${blackwell_prefill_clean_env[@]}" \
        GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 "$@" "$debug_bin" \
        -m "$model" \
        -f "$prompt_file" \
        -n 0 \
        -ngl 999 \
        -b 2048 \
        -ub 2048 \
        --save-logits \
        --logits-output-dir "$run_dir/$label" \
        "${extra_args[@]}" \
        > "$run_dir/$label.stdout" \
        2> "$run_dir/$label.stderr"
    if [[ "$label" == cutlass-* ]] && ! grep -q 'MoE MMQ: backend=cutlass' "$run_dir/$label.stderr"; then
        echo "$label did not execute the CUTLASS MoE backend" >&2
        exit 1
    fi
}

run_case reference GGML_CUDA_MOE_MMQ_DISABLE=1
run_case cutlass-gemm \
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass \
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 \
    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=none
run_case cutlass-w13 \
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass \
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 \
    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=w13
run_case cutlass-full \
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass \
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 \
    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full
run_case cutlass-ceiling \
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass \
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 \
    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full \
    LLAMA_CUDA_FATTN_DIRECT_CAUSAL=1 \
    LLAMA_CUDA_FATTN_Q_ROPE=1 \
    GGML_CUDA_FATTN_SM120_CAUSAL_SCHEDULE=1 \
    GGML_CUDA_ADD_RMS_NORM_FUSION=1

reference_logits=$(find "$run_dir/reference" -maxdepth 1 -type f -name 'llamacpp-*.bin' ! -name '*tokens.bin' -print -quit)
if [[ -z "$reference_logits" ]]; then
    echo "llama-debug did not produce reference logits" >&2
    exit 1
fi

for label in cutlass-gemm cutlass-w13 cutlass-full cutlass-ceiling; do
    candidate_logits=$(find "$run_dir/$label" -maxdepth 1 -type f -name 'llamacpp-*.bin' ! -name '*tokens.bin' -print -quit)
    if [[ -z "$candidate_logits" ]]; then
        echo "llama-debug did not produce $label logits" >&2
        exit 1
    fi
    "$python_bin" "$script_dir/compare_logits.py" \
        "$reference_logits" \
        "$candidate_logits" \
        --metrics-only \
        > "$run_dir/$label-comparison.json"
done

printf '%s\n' "$run_dir"
