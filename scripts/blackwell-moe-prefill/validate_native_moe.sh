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
run_dir="$out_dir/$stamp-native-moe-validation"
reference_dir="$run_dir/reference"
prompt_file="$run_dir/prompt.txt"
mkdir -p "$reference_dir"

for ((i = 0; i < 1024; ++i)); do
    printf ' hello' >> "$prompt_file"
done
printf '\n' >> "$prompt_file"

run_case() {
    local label=$1
    shift
    local logits_dir="$run_dir/$label"
    mkdir -p "$logits_dir"
    env "$@" "$debug_bin" \
        -m "$model" \
        -f "$prompt_file" \
        -n 0 \
        -ngl 999 \
        -b 2048 \
        -ub 2048 \
        --save-logits \
        --logits-output-dir "$logits_dir" \
        "${extra_args[@]}" \
        > "$run_dir/$label.stdout" \
        2> "$run_dir/$label.stderr"
}

run_case reference GGML_CUDA_MOE_MMQ_DISABLE=1
run_case fused-generic GGML_CUDA_MOE_MMQ_PERSISTENT_DISABLE=1
run_case persistent
run_case tma-cooperative \
    GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma \
    GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1
run_case tma-warp-specialized \
    GGML_CUDA_MOE_MMQ_WEIGHT_LAYOUT=tma \
    GGML_CUDA_MOE_MMQ_TMA_REQUIRE=1 \
    GGML_CUDA_MOE_MMQ_TMA_WARP_SPECIALIZED=1

reference_logits=$(find "$reference_dir" -maxdepth 1 -type f -name 'llamacpp-*.bin' ! -name '*tokens.bin' -print -quit)
if [[ -z "$reference_logits" ]]; then
    echo "llama-debug did not produce reference logits" >&2
    exit 1
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for label in fused-generic persistent tma-cooperative tma-warp-specialized; do
    candidate_logits=$(find "$run_dir/$label" -maxdepth 1 -type f -name 'llamacpp-*.bin' ! -name '*tokens.bin' -print -quit)
    if [[ -z "$candidate_logits" ]]; then
        echo "llama-debug did not produce $label logits" >&2
        exit 1
    fi
    "$python_bin" "$script_dir/compare_logits.py" \
        "$reference_logits" \
        "$candidate_logits" \
        --rtol 0 \
        --atol 0 \
        --max-nmse 0 \
        > "$run_dir/$label-comparison.json"
done

printf '%s\n' "$run_dir"
