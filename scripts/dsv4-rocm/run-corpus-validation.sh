#!/usr/bin/env bash
# Attested paired DSV4 validation on a fixed natural-text prompt.
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=${DSV4_MODEL:-/home/edwin/models/DeepSeek-V4-Flash-0731-GGUF/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf}
SERVER=${DSV4_SERVER:-$ROOT_DIR/build/bin/llama-server}
CORPUS=${DSV4_PROMPT_FILE:-$ROOT_DIR/scripts/dsv4-rocm/corpus/technical-proxy.txt}
OUTPUT_ROOT=${DSV4_OUTPUT_ROOT:-$HOME/llama-jobs/dsv4-corpus-validation}
BASE_PORT=${DSV4_PORT:-18180}
CTX_SIZE=${DSV4_CTX_SIZE:-8192}
N_PREDICT=${DSV4_N_PREDICT:-4}
CACHE_REUSE=${DSV4_CACHE_REUSE:-16}
TENSOR_SPLIT=${DSV4_TENSOR_SPLIT:-1,1,1,1}
REFERENCE_SPLIT=${DSV4_REFERENCE_SPLIT:-layer}
FLASH_ATTN=${DSV4_FLASH_ATTN:-on}
STARTUP_RETRIES=${DSV4_STARTUP_RETRIES:-600}
REQUEST_TIMEOUT=${DSV4_REQUEST_TIMEOUT:-600}
HASH_MODE=${DSV4_HASH_MODE:-full}
ALLOW_BUSY=${DSV4_ALLOW_BUSY_GPUS:-0}
BASE_MMQ_J=${DSV4_BASE_MMQ_J-}
CANDIDATE_MMQ_J=${DSV4_CANDIDATE_MMQ_J-16}
BASE_HC_MIXES=${DSV4_BASE_HC_MIXES-}
CANDIDATE_HC_MIXES=${DSV4_CANDIDATE_HC_MIXES-}

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

for pair in "DSV4_PORT:$BASE_PORT" "DSV4_CTX_SIZE:$CTX_SIZE" "DSV4_N_PREDICT:$N_PREDICT" \
            "DSV4_CACHE_REUSE:$CACHE_REUSE" "DSV4_STARTUP_RETRIES:$STARTUP_RETRIES" \
            "DSV4_REQUEST_TIMEOUT:$REQUEST_TIMEOUT"; do
    name=${pair%%:*}
    value=${pair#*:}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer"
done
[[ "$ALLOW_BUSY" == 0 || "$ALLOW_BUSY" == 1 ]] || fail "DSV4_ALLOW_BUSY_GPUS must be 0 or 1"
for pair in "DSV4_BASE_MMQ_J:$BASE_MMQ_J" "DSV4_CANDIDATE_MMQ_J:$CANDIDATE_MMQ_J"; do
    name=${pair%%:*}
    value=${pair#*:}
    case "$value" in
        ""|8|16|24|32|40|48|56|64|72|80|88|96|104|112|120|128) ;;
        *) fail "$name must be empty or a multiple of 8 in [8, 128]" ;;
    esac
done
for pair in "DSV4_BASE_HC_MIXES:$BASE_HC_MIXES" "DSV4_CANDIDATE_HC_MIXES:$CANDIDATE_HC_MIXES"; do
    name=${pair%%:*}
    value=${pair#*:}
    [[ -z "$value" || "$value" == 0 || "$value" == 1 ]] || fail "$name must be empty, 0, or 1"
done
[[ "$HASH_MODE" == full ]] || fail "attested corpus validation requires DSV4_HASH_MODE=full"
[[ "$FLASH_ATTN" == on || "$FLASH_ATTN" == off || "$FLASH_ATTN" == auto ]] || fail "invalid DSV4_FLASH_ATTN"
[[ -f "$MODEL" ]] || fail "model not found: $MODEL"
[[ -x "$SERVER" ]] || fail "server not executable: $SERVER"
[[ -f "$CORPUS" ]] || fail "corpus not found: $CORPUS"
for tool in awk flock git ldd python3 readlink rocm-smi sha256sum tee; do
    command -v "$tool" >/dev/null || fail "$tool is required"
done

MODEL=$(readlink -f "$MODEL")
SERVER=$(readlink -f "$SERVER")
CORPUS=$(readlink -f "$CORPUS")
LIBRARY_PATH=${DSV4_LIBRARY_PATH:-$(dirname "$SERVER")}
export LD_LIBRARY_PATH="$LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ -n $(git -C "$ROOT_DIR" status --porcelain=v1) ]]; then
    fail "source tree must be clean for attested validation"
fi

check_gpus_idle() {
    local phase=$1 output rc busy
    set +e
    output=$(rocm-smi --showpids 2>&1)
    rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
        printf 'rocm-smi --showpids failed during %s (exit %s):\n%s\n' "$phase" "$rc" "$output" >&2
        [[ "$ALLOW_BUSY" == 1 ]] || fail "cannot prove GPUs are idle; refusing to continue"
        return
    fi
    busy=$(printf '%s\n' "$output" | awk '$1 ~ /^[0-9]+$/ { print $0 }')
    if [[ -n "$busy" ]]; then
        printf 'ROCm reports active GPU processes during %s:\n%s\n' "$phase" "$busy" >&2
        [[ "$ALLOW_BUSY" == 1 ]] || fail "refusing to validate on busy GPUs"
    fi
}

mkdir -p "$OUTPUT_ROOT" "$HOME/llama-jobs"
if [[ -z ${LLAMA_JOB_DIR:-} ]]; then
    exec 9>"$HOME/llama-jobs/gpu.lock"
    flock -n 9 || fail "GPU job lock is held"
fi
check_gpus_idle "initial safety check"

commit=$(git -C "$ROOT_DIR" rev-parse --short=12 HEAD)
run_id="$(date -u +%Y%m%dT%H%M%S.%NZ)-attested-${commit}-$RANDOM"
run_dir="$OUTPUT_ROOT/$run_id"
mkdir "$run_dir"
printf 'run_dir=%s\n' "$run_dir"

export DSV4_MODEL="$MODEL"
export DSV4_SERVER="$SERVER"
export DSV4_PROMPT_FILE="$CORPUS"
export DSV4_CTX_SIZE="$CTX_SIZE"
export DSV4_N_PREDICT="$N_PREDICT"
export DSV4_CACHE_REUSE="$CACHE_REUSE"
export DSV4_TENSOR_SPLIT="$TENSOR_SPLIT"
export DSV4_REFERENCE_SPLIT="$REFERENCE_SPLIT"
export DSV4_FLASH_ATTN="$FLASH_ATTN"
export DSV4_STARTUP_RETRIES="$STARTUP_RETRIES"
export DSV4_REQUEST_TIMEOUT="$REQUEST_TIMEOUT"
export DSV4_HASH_MODE="$HASH_MODE"
export DSV4_BASE_MMQ_J="$BASE_MMQ_J"
export DSV4_CANDIDATE_MMQ_J="$CANDIDATE_MMQ_J"
export DSV4_BASE_HC_MIXES="$BASE_HC_MIXES"
export DSV4_CANDIDATE_HC_MIXES="$CANDIDATE_HC_MIXES"
export GGML_CUDA_ALLREDUCE=${GGML_CUDA_ALLREDUCE:-nccl}
export GGML_CUDA_P2P=${GGML_CUDA_P2P:-1}
export GGML_HIP_GRAPHS=${GGML_HIP_GRAPHS:-1}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-10.3.0}
# The per-arm DSV4_* controls above are authoritative. Prevent inherited direct
# overrides from appearing as contradictory evidence in the parent manifest.
unset GGML_HIP_RDNA2_MMQ_J GGML_HIP_RDNA2_HC_MIXES

{
    printf 'export DSV4_MODEL=%q\n' "$DSV4_MODEL"
    printf 'export DSV4_SERVER=%q\n' "$DSV4_SERVER"
    printf 'export DSV4_PROMPT_FILE=%q\n' "$DSV4_PROMPT_FILE"
    printf 'export DSV4_CTX_SIZE=%q\n' "$DSV4_CTX_SIZE"
    printf 'export DSV4_N_PREDICT=%q\n' "$DSV4_N_PREDICT"
    printf 'export DSV4_CACHE_REUSE=%q\n' "$DSV4_CACHE_REUSE"
    printf 'export DSV4_TENSOR_SPLIT=%q\n' "$DSV4_TENSOR_SPLIT"
    printf 'export DSV4_REFERENCE_SPLIT=%q\n' "$DSV4_REFERENCE_SPLIT"
    printf 'export DSV4_FLASH_ATTN=%q\n' "$DSV4_FLASH_ATTN"
    printf 'export DSV4_STARTUP_RETRIES=%q\n' "$DSV4_STARTUP_RETRIES"
    printf 'export DSV4_REQUEST_TIMEOUT=%q\n' "$DSV4_REQUEST_TIMEOUT"
    printf 'export DSV4_HASH_MODE=%q\n' "$DSV4_HASH_MODE"
    printf 'export DSV4_BASE_MMQ_J=%q\n' "$DSV4_BASE_MMQ_J"
    printf 'export DSV4_CANDIDATE_MMQ_J=%q\n' "$DSV4_CANDIDATE_MMQ_J"
    printf 'export DSV4_BASE_HC_MIXES=%q\n' "$DSV4_BASE_HC_MIXES"
    printf 'export DSV4_CANDIDATE_HC_MIXES=%q\n' "$DSV4_CANDIDATE_HC_MIXES"
    printf 'export DSV4_LIBRARY_PATH=%q\n' "$LIBRARY_PATH"
    printf 'export LD_LIBRARY_PATH=%q\n' "$LD_LIBRARY_PATH"
    printf 'export GGML_CUDA_ALLREDUCE=%q\n' "$GGML_CUDA_ALLREDUCE"
    printf 'export GGML_CUDA_P2P=%q\n' "$GGML_CUDA_P2P"
    printf 'export GGML_HIP_GRAPHS=%q\n' "$GGML_HIP_GRAPHS"
    printf 'export HSA_NO_SCRATCH_RECLAIM=%q\n' "$HSA_NO_SCRATCH_RECLAIM"
    printf 'export HSA_OVERRIDE_GFX_VERSION=%q\n' "$HSA_OVERRIDE_GFX_VERSION"
} > "$run_dir/effective-settings.sh"
chmod +x "$run_dir/effective-settings.sh"
sha256sum "$CORPUS" > "$run_dir/corpus.sha256"
cp "$CORPUS" "$run_dir/corpus.txt"

"$ROOT_DIR/scripts/dsv4-rocm/manifest.sh" "$run_dir" "$SERVER" "$MODEL"

variant_controls() {
    local variant=$1
    if [[ "$variant" == base ]]; then
        printf '%s\n' "$BASE_MMQ_J" "$BASE_HC_MIXES"
    else
        printf '%s\n' "$CANDIDATE_MMQ_J" "$CANDIDATE_HC_MIXES"
    fi
}

write_command() {
    local output=$1 variant=$2 port=$3 mmq_j hc_mixes
    local -a controls
    mapfile -t controls < <(variant_controls "$variant")
    mmq_j=${controls[0]}
    hc_mixes=${controls[1]}
    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\nsource %q\n' "$run_dir/effective-settings.sh"
        printf 'env -u GGML_HIP_RDNA2_MMQ_J -u GGML_HIP_RDNA2_HC_MIXES '
        [[ -z "$mmq_j" ]] || printf 'GGML_HIP_RDNA2_MMQ_J=%q ' "$mmq_j"
        [[ -z "$hc_mixes" ]] || printf 'GGML_HIP_RDNA2_HC_MIXES=%q ' "$hc_mixes"
        printf 'DSV4_OUTPUT_DIR=%q DSV4_PORT=%q ' "$run_dir/$variant" "$port"
        printf '%q\n' "$ROOT_DIR/tests/test-dsv4-validation.sh"
    } > "$output"
    chmod +x "$output"
}
write_command "$run_dir/base-command.sh" base "$BASE_PORT"
write_command "$run_dir/candidate-command.sh" candidate "$((BASE_PORT + 2))"
printf '%q\n' "$ROOT_DIR/scripts/dsv4-rocm/compare-validation.py" > "$run_dir/comparator-path.txt"

run_variant() {
    local variant=$1 port=$2 rc mmq_j hc_mixes
    local -a command controls
    mapfile -t controls < <(variant_controls "$variant")
    mmq_j=${controls[0]}
    hc_mixes=${controls[1]}
    command=(env -u GGML_HIP_RDNA2_MMQ_J -u GGML_HIP_RDNA2_HC_MIXES)
    [[ -z "$mmq_j" ]] || command+=("GGML_HIP_RDNA2_MMQ_J=$mmq_j")
    [[ -z "$hc_mixes" ]] || command+=("GGML_HIP_RDNA2_HC_MIXES=$hc_mixes")
    command+=(DSV4_OUTPUT_DIR="$run_dir/$variant" DSV4_PORT="$port" "$ROOT_DIR/tests/test-dsv4-validation.sh")

    check_gpus_idle "pre-$variant safety check"
    set +e
    "${command[@]}" 2>&1 | tee "$run_dir/$variant-validation.log"
    rc=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$rc" > "$run_dir/$variant.rc"
    [[ "$rc" -eq 0 ]] || fail "$variant validation failed with exit $rc"
}

run_variant base "$BASE_PORT"
run_variant candidate "$((BASE_PORT + 2))"
"$ROOT_DIR/scripts/dsv4-rocm/compare-validation.py" \
    "$run_dir/base" "$run_dir/candidate" --json "$run_dir/comparison.json" \
    | tee "$run_dir/comparison.tsv"
sha256sum "$run_dir"/base/{reference,tensor}/*.json "$run_dir"/candidate/{reference,tensor}/*.json \
    > "$run_dir/responses.sha256"
printf 'complete=1\nfinished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" > "$run_dir/status.txt"
printf 'Artifacts: %s\n' "$run_dir"