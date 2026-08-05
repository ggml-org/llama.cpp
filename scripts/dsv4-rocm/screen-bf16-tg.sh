#!/usr/bin/env bash
# Fast matched 0/2K/8K TG triage for guarded BF16 hidden AllReduce.
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUTPUT_ROOT=${DSV4_BF16_SCREEN_OUTPUT_ROOT:-$HOME/llama-jobs/dsv4-rocm-bf16-screen}
CORRECTNESS_ROOT=${DSV4_BF16_EQ_OUTPUT_ROOT:-$HOME/llama-jobs/dsv4-rocm-bf16-equivalence}
CORRECTNESS_DIR=${DSV4_BF16_EQ_RESULT:-}
DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage: scripts/dsv4-rocm/screen-bf16-tg.sh [--dry-run]

After a passing short correctness artifact, runs matched FP32 control and BF16
candidate arms at only 0/2K/8K: tg8, six raw repetitions, first discarded and
five retained. A >2% regression at any depth or <4% gain at 8K is a NO-GO.
A pass is only PROMISING_SHORT_SCREEN and never accepts the optimization. No
16K/32K/64K work is launched by this script.
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

for name in $(compgen -e); do
    case "$name" in
        NCCL_*|RCCL_*|GGML_CUDA_DISABLE_GRAPHS|GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE|GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE_AUDIT)
            fail "refusing inherited $name; each arm uses process-scoped settings" ;;
    esac
done
for tool in awk date find flock fuser git python3 rocm-smi tee; do
    command -v "$tool" >/dev/null || fail "$tool is required"
done
[[ -x "$ROOT_DIR/scripts/dsv4-rocm/run-tg.sh" ]] || fail "run-tg.sh is not executable"
[[ -x "$ROOT_DIR/scripts/dsv4-rocm/compare-bf16-tg.py" ]] || fail "comparator is not executable"

if [[ -z "$CORRECTNESS_DIR" && -d "$CORRECTNESS_ROOT" ]]; then
    CORRECTNESS_DIR=$(find "$CORRECTNESS_ROOT" -mindepth 1 -maxdepth 1 -type d -print | sort | tail -1)
fi
[[ -n "$CORRECTNESS_DIR" && -f "$CORRECTNESS_DIR/comparison.json" ]] || fail "passing correctness artifact is required"
python3 - "$CORRECTNESS_DIR/comparison.json" <<'PY' || fail "correctness artifact did not pass"
import json, sys
v=json.load(open(sys.argv[1]))
assert v.get("complete") is True and v.get("accepted") is True and v.get("classification") == "PASS"
PY
head=$(git -C "$ROOT_DIR" rev-parse HEAD)
grep -qx "git_head=$head" "$CORRECTNESS_DIR/contract.txt" || fail "correctness artifact source commit does not match HEAD"
[[ -z $(git -C "$ROOT_DIR" status --short) ]] || fail "repository must be clean"

printf 'Correctness artifact: %s\n' "$CORRECTNESS_DIR"
printf 'Short contract: depths=0,2048,8192 n_gen=8 raw_reps=6 discard=1 accepted=5\n'
if [[ "$DRY_RUN" == 1 ]]; then
    echo "Dry run only; no ROCm query, lock, model load, or GPU process was started."
    exit 0
fi

if fuser /dev/kfd >/tmp/dsv4-bf16-screen-kfd.$$ 2>/dev/null; then
    cat /tmp/dsv4-bf16-screen-kfd.$$ >&2
    rm -f /tmp/dsv4-bf16-screen-kfd.$$
    fail "/dev/kfd is busy"
fi
rm -f /tmp/dsv4-bf16-screen-kfd.$$
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

screen_id="$(date -u +%Y%m%dT%H%M%S.%NZ)-bf16-hidden-short-screen-$(git -C "$ROOT_DIR" rev-parse --short=12 HEAD)-$RANDOM"
screen_dir="$OUTPUT_ROOT/$screen_id"
mkdir "$screen_dir"
printf 'screen_dir=%s\n' "$screen_dir"
printf 'correctness_artifact=%s\ngit_head=%s\noptimization_accepted=0\n' "$CORRECTNESS_DIR" "$head" > "$screen_dir/contract.txt"

run_arm() {
    local arm=$1 value=$2 label="bf16-hidden-${arm}" launch_log="$screen_dir/${arm}-launch.log" rc arm_dir
    check_gpus_idle "before $arm"
    set +e
    env \
        LLAMA_JOB_DIR="$screen_dir" \
        DSV4_TG_OUTPUT_ROOT="$screen_dir" \
        DSV4_LABEL="$label" \
        DSV4_RCCL_CANDIDATE="$label" \
        DSV4_TG_MODE=performance \
        DSV4_TG_PROFILE=none \
        DSV4_TG_DEPTH_STATE_API=context \
        DSV4_TG_DEPTHS=0,2048,8192 \
        DSV4_TG_N_GEN=8 \
        DSV4_TG_REPS=6 \
        DSV4_TG_DISCARD_FIRST=1 \
        DSV4_TG_STABILITY_LIMIT=0.03 \
        DSV4_HASH_MODE=metadata \
        DSV4_TG_SAMPLE_TIMEOUT=120 \
        DSV4_TG_SETUP_TIMEOUT=900 \
        DSV4_ALLOW_BUSY_GPUS=0 \
        DSV4_REQUIRE_ACCEPTED_STACK=1 \
        DSV4_BATCH=512 \
        DSV4_UBATCH=256 \
        DSV4_TENSOR_SPLIT=1/1/1/1 \
        DSV4_CACHE_TYPE_K=f16 \
        DSV4_CACHE_TYPE_V=f16 \
        DSV4_THREADS=12 \
        GGML_CUDA_ALLREDUCE=nccl \
        GGML_CUDA_P2P=1 \
        GGML_HIP_GRAPHS=1 \
        HSA_OVERRIDE_GFX_VERSION=10.3.0 \
        HSA_NO_SCRATCH_RECLAIM=1 \
        GGML_HIP_RDNA2_MMQ_J=16 \
        GGML_HIP_RDNA2_HC_MIXES=1 \
        GGML_HIP_RDNA2_LID_SUBWAVE=4 \
        GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE="$value" \
        "$ROOT_DIR/scripts/dsv4-rocm/run-tg.sh" 2>&1 | tee "$launch_log"
    rc=${PIPESTATUS[0]}
    set -e
    arm_dir=$(awk -F= '$1 == "run_dir" { value=$2 } END { print value }' "$launch_log")
    [[ -n "$arm_dir" && -d "$arm_dir" ]] || fail "$arm did not report a run directory"
    printf '%s\n' "$arm_dir" > "$screen_dir/${arm}-dir.txt"
    printf '%s\n' "$rc" > "$screen_dir/${arm}-exit-code.txt"
    [[ $rc -eq 0 || $rc -eq 4 ]] || fail "$arm failed with exit $rc"
    check_gpus_idle "after $arm"
}

run_arm control 0
run_arm candidate 1
control_dir=$(<"$screen_dir/control-dir.txt")
candidate_dir=$(<"$screen_dir/candidate-dir.txt")

set +e
"$ROOT_DIR/scripts/dsv4-rocm/compare-bf16-tg.py" "$control_dir" "$candidate_dir" \
    --correctness-dir "$CORRECTNESS_DIR" --json "$screen_dir/comparison.json" | tee "$screen_dir/comparison.txt"
compare_rc=${PIPESTATUS[0]}
set -e
printf 'comparison_exit_code=%s\noptimization_accepted=0\n' "$compare_rc" > "$screen_dir/final-status.txt"
printf 'artifact=%s\n' "$screen_dir"
exit "$compare_rc"