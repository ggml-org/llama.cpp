#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 TEST_BACKEND_OPS LLAMA_DEBUG LLAMA_BENCH MODEL OUT_DIR" >&2
    exit 2
fi

test_backend_ops=$1
debug_bin=$2
bench=$3
model=$4
out_dir=$5

for executable in "$test_backend_ops" "$debug_bin" "$bench"; do
    if [[ ! -x "$executable" ]]; then
        echo "executable does not exist: $executable" >&2
        exit 2
    fi
done
if [[ ! -f "$model" ]]; then
    echo "model does not exist: $model" >&2
    exit 2
fi

python_bin=${PYTHON:-python3}
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "python interpreter does not exist: $python_bin" >&2
    exit 2
fi

threads=${FINAL_PREFILL_THREADS:-25}
repetitions=${FINAL_PREFILL_REPETITIONS:-3}
validate_tokens=${FINAL_PREFILL_VALIDATE_TOKENS:-8192}
ceiling_validate_tokens=${FINAL_PREFILL_CEILING_VALIDATE_TOKENS:-1024}
matrix_cases=${FINAL_PREFILL_MATRIX_CASES:-baseline,sweet-spot,direct-tma-inplace-tuned-norm,full-ceiling-tuned}
matrix_ubatches=${FINAL_PREFILL_MATRIX_UBATCHES:-2048,8192}
run_nsys=${FINAL_PREFILL_RUN_NSYS:-1}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-blackwell-prefill-final"
mkdir -p "$run_dir"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"
git -C "$script_dir/../.." rev-parse HEAD > "$run_dir/llama-commit.txt" 2>/dev/null || true
nvidia-smi -q > "$run_dir/nvidia-smi-q.txt"
printf 'step\tresult\n' > "$run_dir/manifest.tsv"

env "${blackwell_prefill_clean_env[@]}" GGML_CUDA_DISABLE_GRAPHS=1 \
    GGML_CUDA_ADD_RMS_NORM_TEST=1 \
    GGML_CUDA_ADD_RMS_NORM_FUSION=1 \
    "$test_backend_ops" test -b CUDA0 -o ADD_RMS_NORM_MUL \
    > "$run_dir/add-rms-norm-test.stdout" \
    2> "$run_dir/add-rms-norm-test.stderr"
if ! grep -q 'ADD_RMS_NORM_MUL' "$run_dir/add-rms-norm-test.stdout" "$run_dir/add-rms-norm-test.stderr"; then
    echo "no add RMS norm backend case matched" >&2
    exit 1
fi
printf 'add-rms-norm-test\t%s\n' 'add-rms-norm-test.stdout' >> "$run_dir/manifest.tsv"

env "${blackwell_prefill_clean_env[@]}" GGML_CUDA_DISABLE_GRAPHS=1 \
    GGML_CUDA_MOE_MMQ_TEST=1 \
    "$test_backend_ops" test -b CUDA0 -o MOE_MMQ \
    > "$run_dir/moe-mmq-test.stdout" \
    2> "$run_dir/moe-mmq-test.stderr"
if ! grep -q 'MOE_MMQ' "$run_dir/moe-mmq-test.stdout" "$run_dir/moe-mmq-test.stderr"; then
    echo "no MoE MMQ backend case matched" >&2
    exit 1
fi
printf 'moe-mmq-test\t%s\n' 'moe-mmq-test.stdout' >> "$run_dir/manifest.tsv"

tuned_output=$(env \
    PYTHON="$python_bin" \
    PREFILL_TUNED_VALIDATE_TOKENS="$validate_tokens" \
    bash "$script_dir/validate_blackwell_prefill_tuned.sh" \
    "$debug_bin" "$model" "$run_dir")
tuned_dir=${tuned_output##*$'\n'}
printf 'strict-logits\t%s\n' "${tuned_dir#"$run_dir/"}" >> "$run_dir/manifest.tsv"

ceiling_output=$(env \
    PYTHON="$python_bin" \
    PREFILL_CEILING_VALIDATE_TOKENS="$ceiling_validate_tokens" \
    bash "$script_dir/validate_blackwell_prefill_ceiling.sh" \
    "$test_backend_ops" "$debug_bin" "$model" "$run_dir")
ceiling_dir=${ceiling_output##*$'\n'}
printf 'ceiling-logits\t%s\n' "${ceiling_dir#"$run_dir/"}" >> "$run_dir/manifest.tsv"

matrix_output=$(env \
    PYTHON="$python_bin" \
    PREFILL_THREADS="$threads" \
    PREFILL_REPETITIONS="$repetitions" \
    PREFILL_MATRIX_CASES="$matrix_cases" \
    PREFILL_MATRIX_UBATCHES="$matrix_ubatches" \
    bash "$script_dir/bench_blackwell_prefill_matrix.sh" \
    "$bench" "$model" "$run_dir")
matrix_dir=${matrix_output##*$'\n'}
printf 'performance-matrix\t%s\n' "${matrix_dir#"$run_dir/"}" >> "$run_dir/manifest.tsv"

nsys_dir=
if [[ "$run_nsys" != 0 ]]; then
    nsys_output=$(env \
        PYTHON="$python_bin" \
        PREFILL_NSYS_THREADS="$threads" \
        PREFILL_NSYS_CASES=baseline,strict-tuned,full-ceiling-tuned \
        bash "$script_dir/profile_blackwell_prefill_nsys.sh" \
        "$bench" "$model" "$run_dir")
    nsys_dir=${nsys_output##*$'\n'}
    printf 'nsys\t%s\n' "${nsys_dir#"$run_dir/"}" >> "$run_dir/manifest.tsv"
fi

{
    printf '# Blackwell prefill final run\n\n'
    printf -- '- Backend correctness: `add-rms-norm-test.stdout`, `moe-mmq-test.stdout`\n'
    printf -- '- Strict logits: `%s`\n' "${tuned_dir#"$run_dir/"}"
    printf -- '- Ceiling logits metrics: `%s`\n' "${ceiling_dir#"$run_dir/"}"
    printf -- '- Performance matrix: `%s/summary.md`\n' "${matrix_dir#"$run_dir/"}"
    if [[ -n "$nsys_dir" ]]; then
        printf -- '- Nsys decomposition: `%s/summary.md`\n' "${nsys_dir#"$run_dir/"}"
    fi
} > "$run_dir/summary.md"

printf '%s\n' "$run_dir"
