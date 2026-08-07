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
expected_layers=${GPT_OSS_EXPERT_LAYERS:-36}

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
run_dir="$out_dir/$stamp-gpt-oss-mxfp4-decode-logits"
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
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0
    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full
    GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N=32
    GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N=32
    GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB=1
    GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB=1
)

run_case() {
    local label=$1
    local backend=$2
    shift 2
    local case_dir="$run_dir/$label"
    mkdir -p "$case_dir"

    env "${blackwell_prefill_clean_env[@]}" \
        -u GGML_CUDA_DISABLE_GRAPHS \
        GGML_CUDA_DISABLE_GRAPHS=1 \
        GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
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
        if ! grep -q 'MoE MMQ CUTLASS decode dispatch:.*schedule=direct' "$run_dir/$label.stderr"; then
            echo "$label did not execute the direct-schedule CUTLASS decode path" >&2
            exit 1
        fi
        local matched_layers
        matched_layers=$(grep 'MoE MMQ CUTLASS decode dispatch:' "$run_dir/$label.stderr" |
            sed -nE 's/.* weight=([^ ]+) .*/\1/p' | sort -u | wc -l)
        if (( matched_layers < expected_layers )); then
            echo "$label matched only $matched_layers of $expected_layers expected expert layers" >&2
            exit 1
        fi
    elif grep -q 'MoE MMQ CUTLASS decode dispatch:' "$run_dir/$label.stderr"; then
        echo "$label unexpectedly executed the CUTLASS decode path" >&2
        exit 1
    fi
    printf '%s\t%s\toff\t%s\n' \
        "$label" "$backend" "$([[ "$backend" == cutlass ]] && echo bf16 || echo f32)" \
        >> "$run_dir/manifest.tsv"
}

run_case native native GGML_CUDA_MOE_MMQ_DISABLE=1
run_case cutlass cutlass "${cutlass_environment[@]}"

find_logits() {
    find "$1" -maxdepth 1 -type f -name 'llamacpp-*.bin' ! -name '*tokens.bin' -print -quit
}

native_logits=$(find_logits "$run_dir/native")
cutlass_logits=$(find_logits "$run_dir/cutlass")
if [[ -z "$native_logits" || -z "$cutlass_logits" ]]; then
    echo "missing native or CUTLASS logits" >&2
    exit 1
fi
"$python_bin" "$script_dir/compare_logits.py" \
    "$native_logits" "$cutlass_logits" --metrics-only > "$run_dir/native-vs-cutlass.json"
"$python_bin" "$script_dir/summarize_decode_accuracy.py" "$run_dir" > "$run_dir/summary.md"
printf '%s\n' "$run_dir"
