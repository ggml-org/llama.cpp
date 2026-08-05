#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 TEST_BACKEND_OPS OUT_DIR [test-backend-ops arguments...]" >&2
    exit 2
fi

test_bin=$1
out_dir=$2
shift 2
extra_args=("$@")

if [[ ! -x "$test_bin" ]]; then
    echo "test-backend-ops is not executable: $test_bin" >&2
    exit 2
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_dir="$out_dir/$stamp-cutlass-support-validation"
mkdir -p "$run_dir"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/blackwell_prefill_env.sh"

common="GGML_CUDA_MOE_MMQ_BACKEND=cutlass GGML_CUDA_MOE_MMQ_CUTLASS_FUSION=full GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N=32 GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB=1 GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N=64 GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB=1"
cases=(
    "legacy|0|0|0|GGML_CUDA_MOE_MMQ_CUTLASS_PREFIX_DISABLE=1 GGML_CUDA_MOE_MMQ_CUTLASS_CTA_QUANT_DISABLE=1 GGML_CUDA_MOE_MMQ_CUTLASS_CTA_ACTIVATION_DISABLE=1"
    "prefix|1|0|0|GGML_CUDA_MOE_MMQ_CUTLASS_CTA_QUANT_DISABLE=1 GGML_CUDA_MOE_MMQ_CUTLASS_CTA_ACTIVATION_DISABLE=1"
    "prefix-quant|1|1|0|GGML_CUDA_MOE_MMQ_CUTLASS_CTA_ACTIVATION_DISABLE=1"
    "prefix-activation|1|0|1|GGML_CUDA_MOE_MMQ_CUTLASS_CTA_QUANT_DISABLE=1"
    "full-r1|1|1|1|GGML_CUDA_MOE_MMQ_CUTLASS_ACTIVATION_ROWS=1"
    "full|1|1|1|"
    "full-r8|1|1|1|GGML_CUDA_MOE_MMQ_CUTLASS_ACTIVATION_ROWS=8"
)

read -r -a common_env <<< "$common"
for entry in "${cases[@]}"; do
    IFS='|' read -r label prefix quant activation environment <<< "$entry"
    read -r -a case_env <<< "$environment"
    env "${blackwell_prefill_clean_env[@]}" "${common_env[@]}" "${case_env[@]}" \
        GGML_CUDA_DISABLE_GRAPHS=1 \
        GGML_CUDA_MOE_MMQ_TEST=1 \
        GGML_CUDA_MOE_MMQ_LOG_CONFIG=1 \
        GGML_CUDA_MOE_MMQ_CUTLASS_VALIDATE_SUPPORT=1 \
        "$test_bin" test -b CUDA0 -o MOE_MMQ "${extra_args[@]}" \
        > "$run_dir/$label.stdout" \
        2> "$run_dir/$label.stderr"

    pattern="cutlass-prefix=$prefix cutlass-cta-quant=$quant cutlass-cta-activation=$activation"
    if ! grep -q "$pattern" "$run_dir/$label.stderr"; then
        echo "$label did not select the requested support kernels" >&2
        exit 1
    fi
done

printf '%s\n' "$run_dir"
