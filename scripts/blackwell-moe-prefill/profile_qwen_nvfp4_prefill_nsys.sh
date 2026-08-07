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

tokens=${QWEN_PREFILL_NSYS_TOKENS:-8192}
batch=${QWEN_PREFILL_NSYS_BATCH:-8192}
ubatch=${QWEN_PREFILL_NSYS_UBATCH:-8192}
threads=${QWEN_PREFILL_NSYS_THREADS:-25}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-qwen-nvfp4-prefill-nsys"
mkdir -p "$run_dir/cases"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
"$nsys_bin" --version > "$run_dir/nsys-version.txt" 2>&1
printf 'label\tbackend\tenvironment\trun_dir\n' > "$run_dir/manifest.tsv"

native_environment="GGML_CUDA_MOE_MMQ_DISABLE=1"
cutlass_environment="GGML_CUDA_MOE_MMQ_BACKEND=cutlass \
GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full \
GGML_CUDA_MOE_MMQ_CUTLASS_PDL=0 \
GGML_CUDA_MOE_MMQ_CUTLASS_NVFP4_PREFILL_LOG=1"
cases=(
    "native|native|$native_environment"
    "cutlass|cutlass|$cutlass_environment"
)

for entry in "${cases[@]}"; do
    IFS='|' read -r label backend environment <<< "$entry"
    read -r -a env_args <<< "$environment"
    profile_output=$(env "${blackwell_prefill_clean_env[@]}" \
        "${env_args[@]}" \
        GGML_CUDA_DISABLE_GRAPHS=1 \
        GGML_CUDA_MOE_PROFILE=1 \
        NSYS_BIN="$nsys_bin" \
        PYTHON="$python_bin" \
        PREFILL_TOKENS="$tokens" \
        PREFILL_BATCH="$batch" \
        PREFILL_UBATCH="$ubatch" \
        PREFILL_THREADS="$threads" \
        PREFILL_REPETITIONS=1 \
        PREFILL_PROFILE_WARMUP=1 \
        bash "$script_dir/profile_llama.sh" \
        "$label" "$bench" "$model" "$run_dir/cases" -v "${extra_args[@]}")
    case_dir=${profile_output##*$'\n'}

    if [[ ! -f "$case_dir/profile.nsys-rep" ]]; then
        echo "$label did not produce an Nsys report" >&2
        exit 1
    fi
    if [[ "$backend" == cutlass ]] &&
       ! grep -q 'MoE CUTLASS NVFP4 prefill dispatch:' "$case_dir/llama-bench.stderr"; then
        echo "$label did not execute the Qwen NVFP4 CUTLASS prefill path" >&2
        exit 1
    fi
    if [[ "$backend" == native ]] &&
       grep -q 'MoE CUTLASS NVFP4 prefill dispatch:' "$case_dir/llama-bench.stderr"; then
        echo "$label unexpectedly executed the Qwen NVFP4 CUTLASS prefill path" >&2
        exit 1
    fi
    if [[ "$backend" == cutlass ]] &&
       { ! grep -q 'ffn_moe.cutlass_nvfp4_w13' "$case_dir/nvtx-gpu.csv" ||
         ! grep -q 'ffn_moe.cutlass_nvfp4_w2' "$case_dir/nvtx-gpu.csv"; }; then
        echo "$label did not record the CUTLASS W13 and W2 ranges" >&2
        exit 1
    fi

    relative_dir=${case_dir#"$run_dir/"}
    printf '%s\t%s\t%s\t%s\n' "$label" "$backend" "$environment" "$relative_dir" >> "$run_dir/manifest.tsv"
done

"$python_bin" "$script_dir/summarize_nsys.py" "$run_dir" > "$run_dir/summary.md"
printf '%s\n' "$run_dir"
