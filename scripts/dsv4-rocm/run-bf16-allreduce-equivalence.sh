#!/usr/bin/env bash
# Short deterministic FP32-vs-BF16 hidden-AllReduce correctness gate.
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=${DSV4_MODEL:-/home/edwin/models/DeepSeek-V4-Flash-0731-GGUF/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf}
BINARY=${DSV4_BF16_EQ_BINARY:-$ROOT_DIR/build/bin/test-dsv4-bf16-allreduce-equivalence}
OUTPUT_ROOT=${DSV4_BF16_EQ_OUTPUT_ROOT:-$HOME/llama-jobs/dsv4-rocm-bf16-equivalence}
DEPTH=${DSV4_BF16_EQ_DEPTH:-2048}
N_GEN=${DSV4_BF16_EQ_N_GEN:-4}
SEED=${DSV4_BF16_EQ_SEED:-12345}
TIMEOUT_S=${DSV4_BF16_EQ_TIMEOUT:-1200}
THREADS=${DSV4_THREADS:-12}
BATCH=${DSV4_BATCH:-512}
UBATCH=${DSV4_UBATCH:-256}
TENSOR_SPLIT=${DSV4_TENSOR_SPLIT:-1,1,1,1}
DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage: scripts/dsv4-rocm/run-bf16-allreduce-equivalence.sh [--dry-run]

Runs one explicit FP32-control process and one guarded-BF16 process at exactly
2K context with four deterministic teacher-forced target tokens. Each process
captures raw full-vocabulary F32 logits and a per-context AllReduce audit. The
comparator requires identical argmax tokens, finite values, every logit within
0.05 + 0.01*scale, RMSE <= 0.02, exactly 344 eligible hidden reductions, zero
candidate dispatches in control, 344 in candidate, and a positive force-FP32
count. This is a short pre-performance gate; it does not accept an optimization.
USAGE
}

fail() { echo "ERROR: $*" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown argument: $1" ;;
    esac
done

[[ "$DEPTH" == 2048 ]] || fail "short gate requires DSV4_BF16_EQ_DEPTH=2048"
[[ "$N_GEN" == 4 ]] || fail "short gate requires DSV4_BF16_EQ_N_GEN=4"
[[ "$SEED" == 12345 ]] || fail "short gate requires DSV4_BF16_EQ_SEED=12345"
[[ "$BATCH" == 512 && "$UBATCH" == 256 ]] || fail "short gate requires batch/ubatch 512/256"
[[ "$TENSOR_SPLIT" == 1,1,1,1 ]] || fail "short gate requires tensor split 1,1,1,1"
[[ "$THREADS" == 12 ]] || fail "short gate requires 12 threads"
[[ "$TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || fail "timeout must be a positive integer"

for name in $(compgen -e); do
    case "$name" in
        NCCL_*|RCCL_*|GGML_CUDA_DISABLE_GRAPHS|GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE|GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE_AUDIT)
            fail "refusing inherited $name; the gate uses process-scoped settings" ;;
    esac
done

[[ -f "$MODEL" ]] || fail "model not found: $MODEL"
[[ -x "$BINARY" ]] || fail "test binary not executable: $BINARY"
for tool in date flock fuser git python3 readlink rocm-smi setsid sha256sum timeout; do
    command -v "$tool" >/dev/null || fail "$tool is required"
done
MODEL=$(readlink -f "$MODEL")
BINARY=$(readlink -f "$BINARY")
LIBRARY_PATH=${DSV4_LIBRARY_PATH:-$(dirname "$BINARY")}

command=(
    "$BINARY"
    --model "$MODEL"
    --n-gpu-layers 999
    --split-mode tensor
    --tensor-split "$TENSOR_SPLIT"
    --batch-size "$BATCH"
    --ubatch-size "$UBATCH"
    --cache-type-k f16
    --cache-type-v f16
    --flash-attn on
    --threads "$THREADS"
    --threads-batch "$THREADS"
)

printf 'Planned command:'; printf ' %q' "${command[@]}"; printf '\n'
printf 'Contract: depth=%s n_gen=%s seed=%s control=0 candidate=1\n' "$DEPTH" "$N_GEN" "$SEED"
if [[ "$DRY_RUN" == 1 ]]; then
    echo "Dry run only; no ROCm query, lock, model load, or GPU process was started."
    exit 0
fi

[[ -z $(git -C "$ROOT_DIR" status --short) ]] || fail "repository must be clean"
if fuser /dev/kfd >/tmp/dsv4-bf16-kfd.$$ 2>/dev/null; then
    cat /tmp/dsv4-bf16-kfd.$$ >&2
    rm -f /tmp/dsv4-bf16-kfd.$$
    fail "/dev/kfd is busy"
fi
rm -f /tmp/dsv4-bf16-kfd.$$

mkdir -p "$OUTPUT_ROOT" "$HOME/llama-jobs"
exec 9>"$HOME/llama-jobs/gpu.lock"
flock -n 9 || fail "GPU job lock is held"

check_gpus_idle() {
    local phase=$1 output
    output=$(rocm-smi --showpids 2>&1) || fail "cannot prove GPUs idle during $phase: $output"
    if printf '%s\n' "$output" | awk '$1 ~ /^[0-9]+$/ { found=1 } END { exit !found }'; then
        printf '%s\n' "$output" >&2
        fail "ROCm reports an active process during $phase"
    fi
}
check_gpus_idle initial

commit=$(git -C "$ROOT_DIR" rev-parse --short=12 HEAD)
run_id="$(date -u +%Y%m%dT%H%M%S.%NZ)-bf16-hidden-short-correctness-${commit}-$RANDOM"
run_dir="$OUTPUT_ROOT/$run_id"
mkdir "$run_dir"
printf 'run_dir=%s\n' "$run_dir"

{
    printf 'schema_version=1\n'
    printf 'purpose=short_bf16_hidden_allreduce_correctness\n'
    printf 'optimization_accepted=0\n'
    printf 'depth=%s\n' "$DEPTH"
    printf 'n_gen=%s\n' "$N_GEN"
    printf 'seed=%s\n' "$SEED"
    printf 'batch=%s\n' "$BATCH"
    printf 'ubatch=%s\n' "$UBATCH"
    printf 'tensor_split=%s\n' "$TENSOR_SPLIT"
    printf 'abs_tolerance=0.05\nrelative_tolerance=0.01\nmaximum_rmse=0.02\n'
    printf 'git_head=%s\n' "$(git -C "$ROOT_DIR" rev-parse HEAD)"
    printf 'binary=%s\nmodel=%s\n' "$BINARY" "$MODEL"
    printf 'binary_sha256=%s\n' "$(sha256sum "$BINARY" | awk '{print $1}')"
} > "$run_dir/contract.txt"
DSV4_HASH_MODE=metadata "$ROOT_DIR/scripts/dsv4-rocm/manifest.sh" "$run_dir" "$BINARY" "$MODEL"

run_arm() {
    local arm=$1 value=$2 rc
    local arm_dir="$run_dir/$arm"
    mkdir "$arm_dir"
    check_gpus_idle "before $arm"
    {
        printf 'env '
        printf 'GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE=%q ' "$value"
        printf 'GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE_AUDIT=%q ' "$arm_dir/audit.jsonl"
        printf 'DSV4_BF16_EQ_OUTPUT_DIR=%q DSV4_BF16_EQ_DEPTH=%q DSV4_BF16_EQ_N_GEN=%q DSV4_BF16_EQ_SEED=%q ' \
            "$arm_dir" "$DEPTH" "$N_GEN" "$SEED"
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "$arm_dir/command.sh"
    chmod +x "$arm_dir/command.sh"
    printf 'started_at_ns=%s\n' "$(date +%s%N)" > "$arm_dir/status.txt"
    set +e
    env \
        LD_LIBRARY_PATH="$LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        HSA_OVERRIDE_GFX_VERSION=10.3.0 \
        HSA_NO_SCRATCH_RECLAIM=1 \
        GGML_HIP_GRAPHS=1 \
        GGML_CUDA_ALLREDUCE=nccl \
        GGML_CUDA_P2P=1 \
        GGML_HIP_RDNA2_MMQ_J=16 \
        GGML_HIP_RDNA2_HC_MIXES=1 \
        GGML_HIP_RDNA2_LID_SUBWAVE=4 \
        GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE="$value" \
        GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE_AUDIT="$arm_dir/audit.jsonl" \
        DSV4_BF16_EQ_OUTPUT_DIR="$arm_dir" \
        DSV4_BF16_EQ_DEPTH="$DEPTH" \
        DSV4_BF16_EQ_N_GEN="$N_GEN" \
        DSV4_BF16_EQ_SEED="$SEED" \
        setsid timeout --signal=TERM --kill-after=5s "${TIMEOUT_S}s" "${command[@]}" \
        > "$arm_dir/result.json" 2> "$arm_dir/bench.log"
    rc=$?
    set -e
    printf 'process_exit_code=%s\nfinished_at_ns=%s\n' "$rc" "$(date +%s%N)" >> "$arm_dir/status.txt"
    [[ $rc -eq 0 ]] || fail "$arm failed with exit $rc; see $arm_dir/bench.log"
    [[ -s "$arm_dir/result.json" && -s "$arm_dir/logits.f32" && -s "$arm_dir/audit.jsonl" ]] || \
        fail "$arm did not produce complete result/logit/audit artifacts"
    if [[ "$value" == 1 ]]; then
        grep -q 'using guarded RDNA2 BF16 hidden AllReduce' "$arm_dir/bench.log" || \
            fail "candidate dispatch attestation log is missing"
    fi
    check_gpus_idle "after $arm"
}

run_arm control 0
run_arm candidate 1

set +e
"$ROOT_DIR/scripts/dsv4-rocm/compare-bf16-allreduce-equivalence.py" \
    "$run_dir/control" "$run_dir/candidate" --json "$run_dir/comparison.json" \
    | tee "$run_dir/comparison.txt"
compare_rc=${PIPESTATUS[0]}
set -e
printf 'comparison_exit_code=%s\noptimization_accepted=0\n' "$compare_rc" > "$run_dir/final-status.txt"
rocm-smi --showuse --showmemuse --showpower --showclocks --showtemp > "$run_dir/rocm-smi-final.txt" 2>&1 || true
printf 'artifact=%s\n' "$run_dir"
exit "$compare_rc"