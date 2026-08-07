#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 LLAMA_BENCH GPT_OSS_MODEL QWEN_MODEL OUT_DIR [llama-bench arguments...]" >&2
    exit 2
fi

bench=$1
gpt_model=$2
qwen_model=$3
out_dir=$4
shift 4
extra_args=("$@")

if [[ ! -x "$bench" ]]; then
    echo "llama-bench is not executable: $bench" >&2
    exit 2
fi
if [[ ! -f "$gpt_model" ]]; then
    echo "GPT-OSS model does not exist: $gpt_model" >&2
    exit 2
fi
if [[ ! -f "$qwen_model" ]]; then
    echo "Qwen model does not exist: $qwen_model" >&2
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

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-moe-cutlass-ab-nsys"
mkdir -p "$run_dir"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
printf 'workload\trun_dir\n' > "$run_dir/manifest.tsv"

gpt_output=$(PYTHON="$python_bin" NSYS_BIN="$nsys_bin" \
    bash "$script_dir/profile_gpt_oss_mxfp4_decode_nsys.sh" \
    "$bench" "$gpt_model" "$run_dir" "${extra_args[@]}")
gpt_dir=${gpt_output##*$'\n'}
printf 'gpt-oss-mxfp4-decode\t%s\n' "${gpt_dir#"$run_dir/"}" >> "$run_dir/manifest.tsv"

qwen_output=$(PYTHON="$python_bin" NSYS_BIN="$nsys_bin" \
    bash "$script_dir/profile_qwen_nvfp4_prefill_nsys.sh" \
    "$bench" "$qwen_model" "$run_dir" "${extra_args[@]}")
qwen_dir=${qwen_output##*$'\n'}
printf 'qwen-nvfp4-prefill\t%s\n' "${qwen_dir#"$run_dir/"}" >> "$run_dir/manifest.tsv"

printf '%s\n' "$run_dir"
