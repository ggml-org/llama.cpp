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
run_dir="$out_dir/$stamp-qwen-nvfp4-prefill-logits"
prompt_file="$run_dir/prompt.txt"
mkdir -p "$run_dir"
for ((i = 0; i < 1024; ++i)); do
    printf ' hello' >> "$prompt_file"
done
printf '\n' >> "$prompt_file"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
printf 'label\tbackend\n' > "$run_dir/manifest.tsv"

run_case() {
    local label=$1
    local backend=$2
    shift 2
    local logits_dir="$run_dir/$label"
    mkdir -p "$logits_dir"

    env "${blackwell_prefill_clean_env[@]}" \
        GGML_CUDA_DISABLE_GRAPHS=1 \
        "$@" \
        "$debug_bin" \
        -m "$model" \
        -f "$prompt_file" \
        -n 0 \
        -ngl 999 \
        -b 2048 \
        -ub 2048 \
        -fa on \
        --save-logits \
        --logits-output-dir "$logits_dir" \
        "${extra_args[@]}" \
        > "$run_dir/$label.stdout" \
        2> "$run_dir/$label.stderr"

    if [[ "$backend" == cutlass ]] &&
       ! grep -q 'MoE CUTLASS NVFP4 prefill dispatch:' "$run_dir/$label.stderr"; then
        echo "$label did not execute the Qwen NVFP4 CUTLASS prefill path" >&2
        exit 1
    fi
    if [[ "$backend" == native ]] &&
       grep -q 'MoE CUTLASS NVFP4 prefill dispatch:' "$run_dir/$label.stderr"; then
        echo "$label unexpectedly executed the Qwen NVFP4 CUTLASS prefill path" >&2
        exit 1
    fi
    printf '%s\t%s\n' "$label" "$backend" >> "$run_dir/manifest.tsv"
}

run_case native native GGML_CUDA_MOE_MMQ_DISABLE=1
run_case cutlass cutlass \
    GGML_CUDA_MOE_MMQ_BACKEND=cutlass \
    GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full \
    GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 \
    GGML_CUDA_MOE_MMQ_CUTLASS_NVFP4_PREFILL_LOG=1

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

printf '%s\n' "$run_dir"
