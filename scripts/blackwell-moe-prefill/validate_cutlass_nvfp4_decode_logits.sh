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
prompt=${DECODE_LOGITS_PROMPT-Hello}
vllm_logits=${VLLM_LOGITS_FILE-}

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
if [[ -n "$vllm_logits" && ! -f "$vllm_logits" ]]; then
    echo "VLLM_LOGITS_FILE does not exist: $vllm_logits" >&2
    exit 2
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-cutlass-nvfp4-decode-logits"
prompt_file="$run_dir/prompt.txt"
mkdir -p "$run_dir"
printf '%s' "$prompt" > "$prompt_file"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
printf 'label\tbackend\tinput_scale\toutput\n' > "$run_dir/manifest.tsv"

cutlass_environment=(
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE=1
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_LOG=1
    GGML_CUDA_MOE_MMQ_LOG_CONFIG=1
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0
)

run_case() {
    local label=$1
    local backend=$2
    local input_scale=$3
    local output=$4
    shift 4
    local case_dir="$run_dir/$label"
    mkdir -p "$case_dir"

    env "${blackwell_prefill_clean_env[@]}" \
        -u GGML_CUDA_DISABLE_GRAPHS \
        GGML_CUDA_DISABLE_GRAPHS=1 \
        "$@" \
        "$debug_bin" \
        -m "$model" \
        -f "$prompt_file" \
        -ngl 999 \
        -b 512 \
        -ub 512 \
        -fa on \
        -lv 4 \
        --save-logits \
        --logits-output-dir "$case_dir" \
        "${extra_args[@]}" \
        > "$run_dir/$label.stdout" \
        2> "$run_dir/$label.stderr"

    local prompt_info
    prompt_info=$(find "$case_dir" -maxdepth 1 -type f -name 'llamacpp-*-prompt.txt' -print -quit)
    if [[ -z "$prompt_info" ]] || ! grep -q '^n_tokens: 1$' "$prompt_info"; then
        echo "$label did not execute a single-token prompt" >&2
        exit 1
    fi

    if [[ "$backend" == cutlass ]]; then
        if [[ "$label" == cutlass-fused-bf16 ]]; then
            if ! grep -q 'MoE CUTLASS NVFP4 fused decode dispatch:' "$run_dir/$label.stderr"; then
                echo "$label did not execute the fused CUTLASS decode path" >&2
                exit 1
            fi
        else
            if ! grep -Eq "MoE CUTLASS NVFP4 decode dispatch: tokens=1 .*K=2048 N=(512|1024).*output=$output" \
                    "$run_dir/$label.stderr"; then
                echo "$label did not execute the expected CUTLASS W13 path" >&2
                exit 1
            fi
            if ! grep -q "MoE CUTLASS NVFP4 decode dispatch: tokens=1 .*K=512 N=2048.*output=$output" \
                    "$run_dir/$label.stderr"; then
                echo "$label did not execute the expected CUTLASS W2 path" >&2
                exit 1
            fi
        fi
    elif grep -q 'MoE CUTLASS NVFP4 decode dispatch:' "$run_dir/$label.stderr"; then
        echo "$label unexpectedly executed the CUTLASS path" >&2
        exit 1
    fi

    printf '%s\t%s\t%s\t%s\n' "$label" "$backend" "$input_scale" "$output" >> "$run_dir/manifest.tsv"
}

run_case native native off f32 GGML_CUDA_MOE_MMQ_DISABLE=1
run_case native-calibrated native on f32 \
    GGML_CUDA_MOE_MMQ_DISABLE=1 \
    LLAMA_CUDA_MOE_NVFP4_INPUT_SCALE=1
run_case cutlass-dynamic-bf16 cutlass off bf16 "${cutlass_environment[@]}"
run_case cutlass-dynamic-f32 cutlass off f32 \
    "${cutlass_environment[@]}" \
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_OUTPUT_F32=1
run_case cutlass-fused-bf16 cutlass off bf16 \
    "${cutlass_environment[@]}" \
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_FUSED=1
run_case cutlass-calibrated-bf16 cutlass on bf16 \
    "${cutlass_environment[@]}" \
    LLAMA_CUDA_MOE_NVFP4_INPUT_SCALE=1
run_case cutlass-calibrated-f32 cutlass on f32 \
    "${cutlass_environment[@]}" \
    LLAMA_CUDA_MOE_NVFP4_INPUT_SCALE=1 \
    GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_OUTPUT_F32=1

find_logits() {
    find "$1" -maxdepth 1 -type f -name 'llamacpp-*.bin' ! -name '*tokens.bin' -print -quit
}

compare_pair() {
    local reference_label=$1
    local candidate_label=$2
    local reference_file candidate_file output_file
    reference_file=$(find_logits "$run_dir/$reference_label")
    candidate_file=$(find_logits "$run_dir/$candidate_label")
    output_file="$run_dir/${reference_label}-vs-${candidate_label}.json"
    if [[ -z "$reference_file" || -z "$candidate_file" ]]; then
        echo "missing logits for $reference_label or $candidate_label" >&2
        exit 1
    fi
    "$python_bin" "$script_dir/compare_logits.py" \
        "$reference_file" "$candidate_file" --metrics-only > "$output_file"
}

for candidate in \
    native-calibrated \
    cutlass-dynamic-bf16 \
    cutlass-dynamic-f32 \
    cutlass-fused-bf16 \
    cutlass-calibrated-bf16 \
    cutlass-calibrated-f32; do
    compare_pair native "$candidate"
done
compare_pair cutlass-dynamic-bf16 cutlass-dynamic-f32
compare_pair cutlass-dynamic-bf16 cutlass-fused-bf16
compare_pair cutlass-calibrated-bf16 cutlass-calibrated-f32
compare_pair cutlass-dynamic-bf16 cutlass-calibrated-bf16
compare_pair cutlass-dynamic-f32 cutlass-calibrated-f32

if [[ -n "$vllm_logits" ]]; then
    for candidate in \
        native \
        native-calibrated \
        cutlass-dynamic-bf16 \
        cutlass-dynamic-f32 \
        cutlass-fused-bf16 \
        cutlass-calibrated-bf16 \
        cutlass-calibrated-f32; do
        candidate_file=$(find_logits "$run_dir/$candidate")
        "$python_bin" "$script_dir/compare_logits.py" \
            "$vllm_logits" "$candidate_file" --metrics-only > "$run_dir/vllm-vs-${candidate}.json"
    done
fi

"$python_bin" "$script_dir/summarize_decode_accuracy.py" "$run_dir" > "$run_dir/summary.md"
printf '%s\n' "$run_dir"
