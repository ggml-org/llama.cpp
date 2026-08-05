#!/usr/bin/env bash
# Safety-guarded target-only DeepSeek-V4 raw-decode benchmark and residency audit.
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=${DSV4_MODEL:-/home/edwin/models/DeepSeek-V4-Flash-0731-GGUF/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf}
BENCH=${DSV4_BENCH:-$ROOT_DIR/build/bin/llama-bench}
OUTPUT_ROOT=${DSV4_TG_OUTPUT_ROOT:-$HOME/llama-jobs/dsv4-rocm-tg}
MODE=${DSV4_TG_MODE:-performance}
PROFILE=${DSV4_TG_PROFILE:-none}
DEPTH_STATE_API=${DSV4_TG_DEPTH_STATE_API:-context}
DEPTHS=${DSV4_TG_DEPTHS:-0,2048,3072,4096,8192,16384,32768,65536}
N_GEN=${DSV4_TG_N_GEN:-32}
RAW_REPS=${DSV4_TG_REPS:-6}
DISCARD_FIRST=${DSV4_TG_DISCARD_FIRST:-1}
STABILITY_LIMIT=${DSV4_TG_STABILITY_LIMIT:-0.03}
HASH_MODE=${DSV4_HASH_MODE:-metadata}
EXPECTED_DSV4_NODES=21
TELEMETRY_SCOPE=setup-and-discarded-first-repetition
SAMPLE_TIMEOUT_S=${DSV4_TG_SAMPLE_TIMEOUT:-300}
SETUP_TIMEOUT_S=${DSV4_TG_SETUP_TIMEOUT:-1800}
TERM_GRACE_S=${DSV4_TERM_GRACE:-2}
THREADS=${DSV4_THREADS:-12}
BATCH=${DSV4_BATCH:-512}
UBATCH=${DSV4_UBATCH:-256}
TENSOR_SPLIT=${DSV4_TENSOR_SPLIT:-1/1/1/1}
CACHE_TYPE_K=${DSV4_CACHE_TYPE_K:-f16}
CACHE_TYPE_V=${DSV4_CACHE_TYPE_V:-f16}
LOAD_MODE=${DSV4_LOAD_MODE:-mmap}
ALLOW_BUSY=${DSV4_ALLOW_BUSY_GPUS:-0}
REQUIRE_ACCEPTED_STACK=${DSV4_REQUIRE_ACCEPTED_STACK:-1}
LABEL=${DSV4_LABEL:-raw-tg}
RCCL_CANDIDATE=${DSV4_RCCL_CANDIDATE:-control-auto}
STDOUT_CAPTURE="$ROOT_DIR/scripts/dsv4-rocm/capture-tg-stdout.py"
STDOUT_MAX_NON_JSON_LINES=${DSV4_TG_STDOUT_MAX_NON_JSON_LINES:-4096}
DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage: scripts/dsv4-rocm/run-tg.sh [--dry-run]

Modes:
  DSV4_TG_MODE=performance  tg32 context-depth sweep; six raw repetitions,
                            first predeclared graph-cold sample discarded,
                            five accepted samples (default).
  DSV4_TG_MODE=residency    one target evaluation/depth with
                            GGML_SCHED_DEBUG=2 + --verbose, then parse DSV4
                            LID/TOP_K assignments and CPU/GPU graph splits.

Profiling:
  DSV4_TG_PROFILE=kernel    disk-safe rocprofv3 CSV for accepted target-only
                            generation regions only. Requires performance mode
                            and exactly one depth; setup and discarded rep 1
                            are excluded by ROCTx profiler control.

Important overrides:
  DSV4_TG_DEPTHS            default 0,2048,3072,4096,8192,16384,32768,65536
  DSV4_TG_N_GEN             fixed evaluated tokens in performance mode (32)
  DSV4_TG_REPS              raw repetitions (6; accepted = reps-discard)
  DSV4_TG_DISCARD_FIRST     predeclared graph-cold samples to discard (1)
  DSV4_TG_SAMPLE_TIMEOUT    cap for each generation sample (300 seconds)
  DSV4_TG_SETUP_TIMEOUT     cap reset for model/context/depth setup (1800 seconds)
  DSV4_TG_STABILITY_LIMIT   MAD/median acceptance threshold (0.03)
  DSV4_TG_STDOUT_MAX_NON_JSON_LINES  maximum preserved diagnostic lines (4096)
  DSV4_TG_DEPTH_STATE_API  must remain context; sequence restore failed DSV4
                           full-logit equivalence (default: context)
  DSV4_TG_OUTPUT_ROOT       default $HOME/llama-jobs/dsv4-rocm-tg
  DSV4_LABEL                safe run label
  DSV4_HASH_MODE=full       hash all GGUF shards; metadata is default
  DSV4_ROCPROF              rocprofv3 executable (fallback: /opt/rocm/bin/rocprofv3)
  DSV4_ALLOW_BUSY_GPUS=1    unsafe override; never use for controlled evidence
  DSV4_REQUIRE_ACCEPTED_STACK=0  allow non-16/1/4 MMQ/HC/LID controls

The command has no draft-model or speculative option. llama-bench supplies
exactly n_gen target evaluations with deterministic process-local std::rand
input tokens; there is no sampler or EOS early stop. Depth setup and attested
full-context restore are outside samples_ns. Performance and residency are separate runs so verbose
scheduler logging does not perturb accepted TG. The one-second ROCm telemetry
sampler runs during setup and the discarded first repetition only; accepted
performance repetitions are never sampled in-band.
USAGE
}

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown argument: $1" ;;
    esac
done

[[ "$LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "invalid DSV4_LABEL: $LABEL"
[[ "$RCCL_CANDIDATE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "invalid DSV4_RCCL_CANDIDATE: $RCCL_CANDIDATE"
[[ "$MODE" == performance || "$MODE" == residency ]] || fail "DSV4_TG_MODE must be performance or residency"
[[ "$PROFILE" == none || "$PROFILE" == kernel ]] || fail "DSV4_TG_PROFILE must be none or kernel"
[[ "$DEPTH_STATE_API" == context ]] || fail "DSV4_TG_DEPTH_STATE_API must be context; sequence restore failed the DSV4 equivalence gate"
for pair in \
    "DSV4_TG_N_GEN:$N_GEN" "DSV4_TG_REPS:$RAW_REPS" \
    "DSV4_TG_SAMPLE_TIMEOUT:$SAMPLE_TIMEOUT_S" "DSV4_TG_SETUP_TIMEOUT:$SETUP_TIMEOUT_S" \
    "DSV4_TG_STDOUT_MAX_NON_JSON_LINES:$STDOUT_MAX_NON_JSON_LINES" \
    "DSV4_THREADS:$THREADS" "DSV4_BATCH:$BATCH" "DSV4_UBATCH:$UBATCH"; do
    name=${pair%%:*}
    value=${pair#*:}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer"
done
[[ "$DISCARD_FIRST" =~ ^[0-9]+$ ]] || fail "DSV4_TG_DISCARD_FIRST must be a non-negative integer"
[[ "$TERM_GRACE_S" =~ ^[0-9]+$ ]] || fail "DSV4_TERM_GRACE must be a non-negative integer"
[[ "$ALLOW_BUSY" == 0 || "$ALLOW_BUSY" == 1 ]] || fail "DSV4_ALLOW_BUSY_GPUS must be 0 or 1"
[[ "$REQUIRE_ACCEPTED_STACK" == 0 || "$REQUIRE_ACCEPTED_STACK" == 1 ]] || fail "DSV4_REQUIRE_ACCEPTED_STACK must be 0 or 1"
[[ "$HASH_MODE" == metadata || "$HASH_MODE" == full ]] || fail "DSV4_HASH_MODE must be metadata or full"
[ "$UBATCH" -le "$BATCH" ] || fail "ubatch $UBATCH exceeds batch $BATCH"
[ "$TERM_GRACE_S" -lt "$SAMPLE_TIMEOUT_S" ] || fail "DSV4_TERM_GRACE must be below sample timeout"
[ "$TERM_GRACE_S" -lt "$SETUP_TIMEOUT_S" ] || fail "DSV4_TERM_GRACE must be below setup timeout"
python3 - "$STABILITY_LIMIT" <<'PY' || fail "DSV4_TG_STABILITY_LIMIT must be in (0,1)"
import sys
value = float(sys.argv[1])
assert 0 < value < 1
PY

IFS=',' read -r -a depth_values <<< "$DEPTHS"
[ "${#depth_values[@]}" -gt 0 ] || fail "DSV4_TG_DEPTHS is empty"
declare -A seen_depths=()
for value in "${depth_values[@]}"; do
    [[ "$value" =~ ^[0-9]+$ ]] || fail "depth must be a non-negative integer (got '$value')"
    [[ -z ${seen_depths[$value]:-} ]] || fail "duplicate depth $value"
    seen_depths[$value]=1
done

if [[ "$MODE" == residency ]]; then
    N_GEN=1
    RAW_REPS=1
    DISCARD_FIRST=0
    [[ "$PROFILE" == none ]] || fail "DSV4_TG_PROFILE requires performance mode"
else
    [ "$DISCARD_FIRST" -lt "$RAW_REPS" ] || fail "discard count must be below raw repetitions"
    [ $(( RAW_REPS - DISCARD_FIRST )) -ge 5 ] || fail "performance mode requires at least five accepted repetitions"
fi
if [[ "$PROFILE" == kernel ]]; then
    [ "${#depth_values[@]}" -eq 1 ] || fail "DSV4_TG_PROFILE=kernel requires exactly one depth"
    ROCPROF=${DSV4_ROCPROF:-$(command -v rocprofv3 || true)}
    [[ -n "$ROCPROF" ]] || ROCPROF=/opt/rocm/bin/rocprofv3
    [ -x "$ROCPROF" ] || fail "rocprofv3 not executable: $ROCPROF"
    ROCPROF=$(readlink -f "$ROCPROF")
    export LLAMA_BENCH_ROCPROF_SELECTED_REGIONS=1
    export LLAMA_BENCH_ROCPROF_SKIP_REPETITIONS="$DISCARD_FIRST"
fi

export GGML_HIP_RDNA2_MMQ_J=${GGML_HIP_RDNA2_MMQ_J:-16}
export GGML_HIP_RDNA2_HC_MIXES=${GGML_HIP_RDNA2_HC_MIXES:-1}
export GGML_HIP_RDNA2_LID_SUBWAVE=${GGML_HIP_RDNA2_LID_SUBWAVE:-4}
if [[ "$REQUIRE_ACCEPTED_STACK" == 1 ]]; then
    [[ "$GGML_HIP_RDNA2_MMQ_J" == 16 ]] || fail "accepted baseline requires GGML_HIP_RDNA2_MMQ_J=16"
    [[ "$GGML_HIP_RDNA2_HC_MIXES" == 1 ]] || fail "accepted baseline requires GGML_HIP_RDNA2_HC_MIXES=1"
    [[ "$GGML_HIP_RDNA2_LID_SUBWAVE" == 4 ]] || fail "accepted baseline requires GGML_HIP_RDNA2_LID_SUBWAVE=4"
fi
if [[ "$PROFILE" == kernel ]]; then
    [[ "$N_GEN" == 32 ]] || fail "kernel profile requires DSV4_TG_N_GEN=32"
    [[ "$RAW_REPS" =~ ^[0-9]+$ && "$RAW_REPS" -ge 6 ]] || fail "kernel profile requires DSV4_TG_REPS>=6"
    [[ "$DISCARD_FIRST" == 1 ]] || fail "kernel profile requires exactly one discarded repetition"
    [[ "$HASH_MODE" == full ]] || fail "kernel profile requires DSV4_HASH_MODE=full"
    [[ "$REQUIRE_ACCEPTED_STACK" == 1 ]] || fail "kernel profile requires DSV4_REQUIRE_ACCEPTED_STACK=1"
    [[ "$GGML_HIP_RDNA2_MMQ_J" == 16 && "$GGML_HIP_RDNA2_HC_MIXES" == 1 && "$GGML_HIP_RDNA2_LID_SUBWAVE" == 4 ]] || \
        fail "kernel profile requires exact J16/HC1/LID4 controls"
    [[ "$BATCH" == 512 && "$UBATCH" == 256 ]] || fail "kernel profile requires batch/ubatch 512/256"
    [[ "$TENSOR_SPLIT" == 1/1/1/1 ]] || fail "kernel profile requires tensor split 1/1/1/1"
    [[ "$CACHE_TYPE_K" == f16 && "$CACHE_TYPE_V" == f16 ]] || fail "kernel profile requires F16 K/V"
    [[ "$THREADS" == 12 ]] || fail "kernel profile requires 12 host threads"
    [[ "$LOAD_MODE" == mmap ]] || fail "kernel profile requires mmap load mode"
fi
export DSV4_TG_PROFILE="$PROFILE"
export DSV4_TG_STDOUT_MAX_NON_JSON_LINES="$STDOUT_MAX_NON_JSON_LINES"
export DSV4_HASH_MODE="$HASH_MODE"
export DSV4_RCCL_CANDIDATE="$RCCL_CANDIDATE"

[ -f "$MODEL" ] || fail "model not found: $MODEL"
[ -x "$BENCH" ] || fail "llama-bench not executable: $BENCH"
[ -f "$STDOUT_CAPTURE" ] || fail "stdout capture helper not found: $STDOUT_CAPTURE"
for tool in awk date flock grep mv python3 readlink rocm-smi setsid sha256sum; do
    command -v "$tool" >/dev/null || fail "$tool is required"
done
BENCH=$(readlink -f "$BENCH")
MODEL=$(readlink -f "$MODEL")
LIBRARY_PATH=${DSV4_LIBRARY_PATH:-$(dirname "$BENCH")}
export LD_LIBRARY_PATH="$LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export GGML_CUDA_ALLREDUCE=${GGML_CUDA_ALLREDUCE:-nccl}
export GGML_CUDA_P2P=${GGML_CUDA_P2P:-1}
export GGML_HIP_GRAPHS=${GGML_HIP_GRAPHS:-1}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-10.3.0}
export LLAMA_BENCH_DEPTH_STATE_API="$DEPTH_STATE_API"
if [[ "$MODE" == residency ]]; then
    export GGML_SCHED_DEBUG=2
else
    export GGML_SCHED_DEBUG=0
fi

bench_cmd=(
    "$BENCH"
    --model "$MODEL"
    --n-prompt 0
    --n-gen "$N_GEN"
    --n-depth "$DEPTHS"
    --batch-size "$BATCH"
    --ubatch-size "$UBATCH"
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --threads "$THREADS"
    --n-gpu-layers 999
    --split-mode tensor
    --tensor-split "$TENSOR_SPLIT"
    --flash-attn on
    --load-mode "$LOAD_MODE"
    --repetitions "$RAW_REPS"
    --no-warmup
    --output jsonl
    --progress
)
[[ "$MODE" == residency ]] && bench_cmd+=(--verbose)

printf 'Mode: %s; profile: %s\n' "$MODE" "$PROFILE"
printf 'Planned command:'
printf ' %q' "${bench_cmd[@]}"
printf '\n'
printf 'Environment: GGML_SCHED_DEBUG=%q LLAMA_BENCH_DEPTH_STATE_API=%q GGML_HIP_RDNA2_MMQ_J=%q GGML_HIP_RDNA2_HC_MIXES=%q GGML_HIP_RDNA2_LID_SUBWAVE=%q GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE=%q DSV4_RCCL_CANDIDATE=%q NCCL_ALGO=%q NCCL_PROTO=%q NCCL_MIN_NCHANNELS=%q NCCL_MAX_NCHANNELS=%q GGML_CUDA_DISABLE_GRAPHS=%q\n' \
    "$GGML_SCHED_DEBUG" "$LLAMA_BENCH_DEPTH_STATE_API" "$GGML_HIP_RDNA2_MMQ_J" "$GGML_HIP_RDNA2_HC_MIXES" "$GGML_HIP_RDNA2_LID_SUBWAVE" \
    "${GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE-<unset>}" "$RCCL_CANDIDATE" "${NCCL_ALGO-<unset>}" "${NCCL_PROTO-<unset>}" "${NCCL_MIN_NCHANNELS-<unset>}" "${NCCL_MAX_NCHANNELS-<unset>}" "${GGML_CUDA_DISABLE_GRAPHS-<unset>}"
printf 'Per-sample cap: %ss; per-setup cap: %ss\n' "$SAMPLE_TIMEOUT_S" "$SETUP_TIMEOUT_S"
if [[ "$PROFILE" == kernel ]]; then
    printf 'Profiler: %s; selected accepted regions only; skip repetitions: %s\n' "$ROCPROF" "$DISCARD_FIRST"
fi
if [[ "$DRY_RUN" == 1 ]]; then
    echo "Dry run only; no ROCm query, model load, or benchmark process was started."
    exit 0
fi

check_gpus_idle() {
    local phase=$1 output rc busy
    set +e
    output=$(rocm-smi --showpids 2>&1)
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        printf 'rocm-smi --showpids failed during %s (exit %s):\n%s\n' "$phase" "$rc" "$output" >&2
        [ "$ALLOW_BUSY" -eq 1 ] || fail "cannot prove GPUs are idle; refusing to continue"
        echo "WARNING: GPU discovery failed; unsafe override enabled" >&2
        return
    fi
    busy=$(printf '%s\n' "$output" | awk '$1 ~ /^[0-9]+$/ { print $0 }')
    if [[ -n "$busy" ]]; then
        printf 'ROCm reports active GPU processes during %s:\n%s\n' "$phase" "$busy" >&2
        [ "$ALLOW_BUSY" -eq 1 ] || fail "refusing to benchmark busy GPUs"
        echo "WARNING: proceeding on busy GPUs because DSV4_ALLOW_BUSY_GPUS=1" >&2
    fi
}

mkdir -p "$OUTPUT_ROOT" "$HOME/llama-jobs"
if [[ -z ${LLAMA_JOB_DIR:-} ]]; then
    exec 9>"$HOME/llama-jobs/gpu.lock"
    flock -n 9 || fail "GPU job lock is held; wait for the active job or use scripts/rdna2-job.sh"
fi
check_gpus_idle "initial safety check"

commit=nogit
if git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    commit=$(git -C "$ROOT_DIR" rev-parse --short=12 HEAD)
fi
run_id="$(date -u +%Y%m%dT%H%M%S.%NZ)-${LABEL}-${MODE}-${commit}-$RANDOM"
run_dir="$OUTPUT_ROOT/$run_id"
mkdir "$run_dir"

python3 - "$run_dir/clock-domain.txt" <<'PY'
from pathlib import Path
import sys, time
before = time.time_ns()
mono = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
after = time.time_ns()
real = (before + after) // 2
boot = Path("/proc/sys/kernel/random/boot_id")
Path(sys.argv[1]).write_text(
    f"boot_id={boot.read_text().strip() if boot.is_file() else 'unavailable'}\n"
    f"start_captured_realtime_ns={real}\n"
    f"start_captured_monotonic_ns={mono}\n"
    f"start_realtime_minus_monotonic_ns={real - mono}\n"
    f"start_calibration_span_ns={after - before}\n"
)
PY

write_repro_env_prefix() {
    local output=$1 name
    local -a optional=(NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_DEBUG NCCL_DEBUG_SUBSYS GGML_CUDA_DISABLE_GRAPHS GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE_AUDIT)
    printf 'env' > "$output"
    for name in "${optional[@]}"; do
        declare -p "$name" >/dev/null 2>&1 || printf ' -u %q' "$name" >> "$output"
    done
    printf ' DSV4_RCCL_CANDIDATE=%q' "$RCCL_CANDIDATE" >> "$output"
    for name in "${optional[@]}"; do
        declare -p "$name" >/dev/null 2>&1 && printf ' %s=%q' "$name" "${!name}" >> "$output"
    done
    printf ' LLAMA_BENCH_DEPTH_STATE_API=%q ' "$LLAMA_BENCH_DEPTH_STATE_API" >> "$output"
}

write_repro_env_prefix "$run_dir/command.sh"
if [[ "$PROFILE" == kernel ]]; then
    export LLAMA_BENCH_ROCPROF_BOUNDARIES="$run_dir/rocprof-selected-regions.tsv"
    printf 'LLAMA_BENCH_ROCPROF_SELECTED_REGIONS=1 LLAMA_BENCH_ROCPROF_SKIP_REPETITIONS=%q LLAMA_BENCH_ROCPROF_BOUNDARIES=%q ' \
        "$DISCARD_FIRST" "$LLAMA_BENCH_ROCPROF_BOUNDARIES" >> "$run_dir/command.sh"
fi
printf '%q ' "${bench_cmd[@]}" >> "$run_dir/command.sh"
printf '\n' >> "$run_dir/command.sh"
chmod +x "$run_dir/command.sh"

payload_cmd=("${bench_cmd[@]}")
if [[ "$PROFILE" == kernel ]]; then
    mkdir "$run_dir/rocprof"
    profile_args=(
        --output-directory "$run_dir/rocprof"
        --output-file dsv4-tg
        --output-format csv
        --kernel-trace
        --memory-copy-trace
        --rccl-trace
        --hip-runtime-trace
        --selected-regions
        --stats
        --summary
        --summary-per-domain
        --summary-output-file "$run_dir/rocprof-summary.txt"
    )
    payload_cmd=("$ROCPROF" "${profile_args[@]}" -- "${bench_cmd[@]}")
fi
write_repro_env_prefix "$run_dir/executed-command.sh"
if [[ "$PROFILE" == kernel ]]; then
    printf 'LLAMA_BENCH_ROCPROF_SELECTED_REGIONS=1 LLAMA_BENCH_ROCPROF_SKIP_REPETITIONS=%q LLAMA_BENCH_ROCPROF_BOUNDARIES=%q ' \
        "$DISCARD_FIRST" "$LLAMA_BENCH_ROCPROF_BOUNDARIES" >> "$run_dir/executed-command.sh"
fi
printf '%q ' "${payload_cmd[@]}" >> "$run_dir/executed-command.sh"
printf '\n' >> "$run_dir/executed-command.sh"
chmod +x "$run_dir/executed-command.sh"

{
    printf 'DSV4_TG_MODE=%q\n' "$MODE"
    printf 'DSV4_TG_PROFILE=%q\n' "$PROFILE"
    printf 'DSV4_RCCL_CANDIDATE=%q\n' "$RCCL_CANDIDATE"
    for name in NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_DEBUG NCCL_DEBUG_SUBSYS GGML_CUDA_DISABLE_GRAPHS GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE_AUDIT; do
        if declare -p "$name" >/dev/null 2>&1; then
            printf '%s_IS_SET=1\n' "$name"
            printf '%s=%q\n' "$name" "${!name}"
        else
            printf '%s_IS_SET=0\n' "$name"
        fi
    done
    [[ "$PROFILE" == kernel ]] && printf 'DSV4_ROCPROF=%q\n' "$ROCPROF"
    printf 'DSV4_TG_DEPTH_STATE_API=%q\n' "$DEPTH_STATE_API"
    printf 'LLAMA_BENCH_DEPTH_STATE_API=%q\n' "$LLAMA_BENCH_DEPTH_STATE_API"
    printf 'DSV4_MODEL=%q\n' "$MODEL"
    printf 'DSV4_BENCH=%q\n' "$BENCH"
    printf 'DSV4_TG_DEPTHS=%q\n' "$DEPTHS"
    printf 'DSV4_TG_N_GEN=%q\n' "$N_GEN"
    printf 'DSV4_TG_REPS=%q\n' "$RAW_REPS"
    printf 'DSV4_TG_DISCARD_FIRST=%q\n' "$DISCARD_FIRST"
    printf 'DSV4_TG_STABILITY_LIMIT=%q\n' "$STABILITY_LIMIT"
    printf 'DSV4_HASH_MODE=%q\n' "$HASH_MODE"
    printf 'DSV4_EXPECTED_DSV4_NODES=%q\n' "$EXPECTED_DSV4_NODES"
    printf 'DSV4_TG_TELEMETRY_SCOPE=%q\n' "$TELEMETRY_SCOPE"
    printf 'DSV4_TG_SAMPLE_TIMEOUT=%q\n' "$SAMPLE_TIMEOUT_S"
    printf 'DSV4_TG_SETUP_TIMEOUT=%q\n' "$SETUP_TIMEOUT_S"
    printf 'DSV4_TG_STDOUT_MAX_NON_JSON_LINES=%q\n' "$STDOUT_MAX_NON_JSON_LINES"
    printf 'DSV4_BATCH=%q\n' "$BATCH"
    printf 'DSV4_UBATCH=%q\n' "$UBATCH"
    printf 'DSV4_TENSOR_SPLIT=%q\n' "$TENSOR_SPLIT"
    printf 'DSV4_ALLOW_BUSY_GPUS=%q\n' "$ALLOW_BUSY"
    printf 'DSV4_REQUIRE_ACCEPTED_STACK=%q\n' "$REQUIRE_ACCEPTED_STACK"
    printf 'GGML_SCHED_DEBUG=%q\n' "$GGML_SCHED_DEBUG"
    printf 'GGML_HIP_RDNA2_MMQ_J=%q\n' "$GGML_HIP_RDNA2_MMQ_J"
    printf 'GGML_HIP_RDNA2_HC_MIXES=%q\n' "$GGML_HIP_RDNA2_HC_MIXES"
    printf 'GGML_HIP_RDNA2_LID_SUBWAVE=%q\n' "$GGML_HIP_RDNA2_LID_SUBWAVE"
    printf 'GGML_CUDA_ALLREDUCE=%q\n' "$GGML_CUDA_ALLREDUCE"
    printf 'GGML_CUDA_P2P=%q\n' "$GGML_CUDA_P2P"
    printf 'GGML_HIP_GRAPHS=%q\n' "$GGML_HIP_GRAPHS"
    printf 'HSA_NO_SCRATCH_RECLAIM=%q\n' "$HSA_NO_SCRATCH_RECLAIM"
    printf 'HSA_OVERRIDE_GFX_VERSION=%q\n' "$HSA_OVERRIDE_GFX_VERSION"
    printf 'LD_LIBRARY_PATH=%q\n' "$LD_LIBRARY_PATH"
    if [[ "$PROFILE" == kernel ]]; then
        printf 'LLAMA_BENCH_ROCPROF_SELECTED_REGIONS=%q\n' "$LLAMA_BENCH_ROCPROF_SELECTED_REGIONS"
        printf 'LLAMA_BENCH_ROCPROF_SKIP_REPETITIONS=%q\n' "$LLAMA_BENCH_ROCPROF_SKIP_REPETITIONS"
        printf 'LLAMA_BENCH_ROCPROF_BOUNDARIES=%q\n' "$LLAMA_BENCH_ROCPROF_BOUNDARIES"
    fi
} > "$run_dir/effective-settings.sh"

python3 - "$run_dir/contract.json" "$MODE" "$DEPTHS" "$N_GEN" "$RAW_REPS" "$DISCARD_FIRST" "$DEPTH_STATE_API" "${bench_cmd[@]}" <<'PY'
import json, pathlib, sys
out, mode, depths, n_gen, reps, discard, depth_state_api, *command = sys.argv[1:]
forbidden = ("--model-draft", "-md", "--spec-type", "--spec-draft", "dspark")
hits = [arg for arg in command if any(token in arg.lower() for token in forbidden)]
if hits:
    raise SystemExit(f"forbidden speculative command arguments: {hits}")
pathlib.Path(out).write_text(json.dumps({
    "target_only": True,
    "draft_model_loaded": False,
    "speculative_flags": [],
    "mode": mode,
    "depths": [int(v) for v in depths.split(",")],
    "n_gen": int(n_gen),
    "raw_repetitions": int(reps),
    "discard_first": int(discard),
    "accepted_repetitions": int(reps) - int(discard),
    "depth_state_api": depth_state_api,
    "telemetry_scope": "setup-and-discarded-first-repetition",
    "profile": __import__("os").environ.get("DSV4_TG_PROFILE", "none"),
    "profile_scope": "accepted-target-generation-selected-regions" if __import__("os").environ.get("DSV4_TG_PROFILE") == "kernel" else "none",
    "profile_skip_repetitions": int(__import__("os").environ.get("LLAMA_BENCH_ROCPROF_SKIP_REPETITIONS", "0")),
    "model_hash_mode": __import__("os").environ.get("DSV4_HASH_MODE", "metadata"),
    "require_accepted_stack": int(__import__("os").environ.get("DSV4_REQUIRE_ACCEPTED_STACK", "1")),
    "allow_busy_gpus": int(__import__("os").environ.get("DSV4_ALLOW_BUSY_GPUS", "0")),
    "stdout_capture": {
        "schema_version": 1,
        "raw_stream": "bench.stdout.log",
        "non_json_stream": "bench.stdout-nonjson.log",
        "classification": "stdout-classification.json",
        "max_non_json_lines": int(__import__("os").environ.get("DSV4_TG_STDOUT_MAX_NON_JSON_LINES", "4096")),
    },
    "communication_candidate": {
        "label": __import__("os").environ.get("DSV4_RCCL_CANDIDATE", "control-auto"),
        "backend": __import__("os").environ.get("GGML_CUDA_ALLREDUCE", ""),
        "hip_graphs": __import__("os").environ.get("GGML_HIP_GRAPHS", ""),
        "runtime_graph_disable": __import__("os").environ.get("GGML_CUDA_DISABLE_GRAPHS"),
        "algorithm": __import__("os").environ.get("NCCL_ALGO"),
        "protocol": __import__("os").environ.get("NCCL_PROTO"),
        "min_channels": __import__("os").environ.get("NCCL_MIN_NCHANNELS"),
        "max_channels": __import__("os").environ.get("NCCL_MAX_NCHANNELS"),
        "debug": __import__("os").environ.get("NCCL_DEBUG"),
        "debug_subsys": __import__("os").environ.get("NCCL_DEBUG_SUBSYS"),
        "bf16_hidden_allreduce": __import__("os").environ.get("GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE"),
        "bf16_hidden_allreduce_audit": __import__("os").environ.get("GGML_HIP_RDNA2_BF16_HIDDEN_ALLREDUCE_AUDIT"),
    },
    "accepted_stack": {
        "mmq_j": int(__import__("os").environ["GGML_HIP_RDNA2_MMQ_J"]),
        "hc_mixes": int(__import__("os").environ["GGML_HIP_RDNA2_HC_MIXES"]),
        "lid_subwave": int(__import__("os").environ["GGML_HIP_RDNA2_LID_SUBWAVE"]),
    },
    "batch": int(__import__("os").environ.get("DSV4_BATCH", "512")),
    "ubatch": int(__import__("os").environ.get("DSV4_UBATCH", "256")),
    "tensor_split": __import__("os").environ.get("DSV4_TENSOR_SPLIT", "1/1/1/1"),
    "cache_type_k": __import__("os").environ.get("DSV4_CACHE_TYPE_K", "f16"),
    "cache_type_v": __import__("os").environ.get("DSV4_CACHE_TYPE_V", "f16"),
    "threads": int(__import__("os").environ.get("DSV4_THREADS", "12")),
    "load_mode": __import__("os").environ.get("DSV4_LOAD_MODE", "mmap"),
    "expected_dsv4_nodes_per_graph": 21,
    "token_contract": "llama-bench test_gen: exactly n_gen target llama_decode calls; BOS then deterministic process-local std::rand tokens; no sampler or EOS",
    "depth_contract": "llama-bench n_depth setup occurs outside samples_ns; later repetitions restore the attested full context state; sequence-only restore is forbidden for DSV4",
    "command_argv": command,
}, indent=2) + "\n")
PY

"$ROOT_DIR/scripts/dsv4-rocm/manifest.sh" "$run_dir" "$BENCH" "$MODEL"
check_gpus_idle "pre-launch safety check"

now_ns() { date +%s%N; }
group_alive() { kill -0 -- "-$1" 2>/dev/null; }
leader_or_group_alive() { kill -0 "$1" 2>/dev/null || group_alive "$1"; }
signal_group() { kill "-$1" -- "-$2" 2>/dev/null || true; }
terminate_group_now() {
    local pgid=$1 deadline
    group_alive "$pgid" || return 0
    signal_group TERM "$pgid"
    deadline=$(( $(now_ns) + 2000000000 ))
    while group_alive "$pgid" && [ "$(now_ns)" -lt "$deadline" ]; do sleep 0.05; done
    group_alive "$pgid" && signal_group KILL "$pgid"
}

sample_smi() {
    local pgid=$1 phase_file=$2
    local event_ns event_phase event_bench event_rep event_total
    while leader_or_group_alive "$pgid"; do
        event_ns=""; event_phase=""; event_bench=""; event_rep=""; event_total=""
        if [[ -s "$phase_file" ]]; then
            IFS=$'\t' read -r event_ns event_phase event_bench event_rep event_total < "$phase_file" || true
        fi
        # rocm-smi queries are intentionally excluded from accepted TG samples.
        # Continue through initial setup and repetition 1, which is predeclared
        # graph-cold and discarded. Skip all later generation/restore windows.
        if [[ -z "$event_phase" || "$event_phase" == setup && "$event_rep" =~ ^[01]$ ]]; then
            printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
            rocm-smi --showuse --showmemuse --showpower --showclocks --showtemp --csv 2>&1 || true
        fi
        sleep 1
    done
}

stderr_consumer() {
    local stderr_fifo=$1 phase_file=$2 log_file=$3 events_file=$4 line stamp phase bench rep total phase_tmp
    phase_tmp="${phase_file}.tmp"
    while IFS= read -r line || [[ -n "$line" ]]; do
        printf '%s\n' "$line" >> "$log_file"
        phase=""; bench=""; rep=""; total=""
        if [[ "$line" =~ llama-bench:\ benchmark\ ([0-9]+)/([0-9]+):\ starting ]]; then
            phase=setup; bench=${BASH_REMATCH[1]}; rep=0; total=0
        elif [[ "$line" =~ llama-bench:\ benchmark\ ([0-9]+)/([0-9]+):\ depth\ run\ ([0-9]+)/([0-9]+) ]]; then
            phase=setup; bench=${BASH_REMATCH[1]}; rep=${BASH_REMATCH[3]}; total=${BASH_REMATCH[4]}
        elif [[ "$line" =~ llama-bench:\ benchmark\ ([0-9]+)/([0-9]+):\ generation\ run\ ([0-9]+)/([0-9]+) ]]; then
            phase=measurement; bench=${BASH_REMATCH[1]}; rep=${BASH_REMATCH[3]}; total=${BASH_REMATCH[4]}
        fi
        if [[ -n "$phase" ]]; then
            stamp=$(now_ns)
            printf '%s\t%s\t%s\t%s\t%s\n' "$stamp" "$phase" "$bench" "$rep" "$total" >> "$events_file"
            printf '%s\t%s\t%s\t%s\t%s\n' "$stamp" "$phase" "$bench" "$rep" "$total" > "$phase_tmp"
            mv -f "$phase_tmp" "$phase_file"
            [[ "$phase" == measurement ]] && printf '%s\n' "$stamp" >> "$run_dir/measurement-start.ns"
        fi
    done < "$stderr_fifo"
}

watch_group() {
    local pgid=$1 phase_file=$2 phase=setup phase_start timeout_s hard term now last_event=""
    local event_ns event_phase event_bench event_rep event_total
    phase_start=$(now_ns)
    while leader_or_group_alive "$pgid"; do
        if [[ -s "$phase_file" ]]; then
            event_ns=""; event_phase=""; event_bench=""; event_rep=""; event_total=""
            IFS=$'\t' read -r event_ns event_phase event_bench event_rep event_total < "$phase_file" || true
            if [[ "$event_ns" != "$last_event" && "$event_ns" =~ ^[0-9]+$ && ( "$event_phase" == setup || "$event_phase" == measurement ) ]]; then
                last_event=$event_ns
                phase=$event_phase
                phase_start=$event_ns
                printf 'phase=%s benchmark=%s repetition=%s started_at_ns=%s\n' "$phase" "$event_bench" "$event_rep" "$event_ns" >> "$run_dir/status.txt"
            fi
        fi
        timeout_s=$SETUP_TIMEOUT_S
        [[ "$phase" == measurement ]] && timeout_s=$SAMPLE_TIMEOUT_S
        hard=$(( phase_start + timeout_s * 1000000000 ))
        term=$(( hard - TERM_GRACE_S * 1000000000 ))
        now=$(now_ns)
        if [ "$now" -ge "$term" ]; then
            : > "$run_dir/${phase}-timeout"
            printf 'timeout_phase=%s term_at_ns=%s\n' "$phase" "$now" >> "$run_dir/status.txt"
            signal_group TERM "$pgid"
            while group_alive "$pgid"; do
                now=$(now_ns)
                if [ "$now" -ge "$hard" ]; then
                    printf 'timeout_phase=%s kill_at_ns=%s\n' "$phase" "$now" >> "$run_dir/status.txt"
                    signal_group KILL "$pgid"
                    return
                fi
                sleep 0.05
            done
            return
        fi
        sleep 0.1
    done
}

bench_pid=""; bench_pgid=""; smi_pid=""; watchdog_pid=""; stderr_pid=""; stdout_pid=""
stderr_fifo="$run_dir/bench.stderr.fifo"
stdout_fifo="$run_dir/bench.stdout.fifo"
phase_file="$run_dir/phase-latest.tsv"
cleanup_children() {
    local exit_code=$?
    trap - EXIT INT TERM HUP
    for child in "$smi_pid" "$watchdog_pid"; do
        [[ -n "$child" ]] && kill "$child" 2>/dev/null || true
    done
    [[ -n "$bench_pgid" ]] && terminate_group_now "$bench_pgid"
    [[ -n "$bench_pid" ]] && wait "$bench_pid" 2>/dev/null || true
    for child in "$stderr_pid" "$stdout_pid" "$smi_pid" "$watchdog_pid"; do
        [[ -n "$child" ]] && { kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; }
    done
    rm -f "$stderr_fifo" "$stdout_fifo"
    exit "$exit_code"
}
trap cleanup_children EXIT INT TERM HUP

mkfifo "$stderr_fifo" "$stdout_fifo"
: > "$run_dir/bench.log"
: > "$phase_file"
: > "$run_dir/result.jsonl"
: > "$run_dir/result-completed-at.ns"
: > "$run_dir/bench.stdout.log"
: > "$run_dir/bench.stdout-nonjson.log"
: > "$run_dir/measurement-start.ns"
printf 'timestamp_ns\tphase\tbenchmark\trepetition\ttotal_repetitions\n' > "$run_dir/phase-events.tsv"
stderr_consumer "$stderr_fifo" "$phase_file" "$run_dir/bench.log" "$run_dir/phase-events.tsv" 9>&- & stderr_pid=$!
python3 "$STDOUT_CAPTURE" \
    --result "$run_dir/result.jsonl" \
    --completed-at "$run_dir/result-completed-at.ns" \
    --raw "$run_dir/bench.stdout.log" \
    --non-json "$run_dir/bench.stdout-nonjson.log" \
    --classification "$run_dir/stdout-classification.json" \
    --max-non-json-lines "$STDOUT_MAX_NON_JSON_LINES" \
    < "$stdout_fifo" 9>&- & stdout_pid=$!

printf 'run_dir=%s\n' "$run_dir"
printf 'started_at_ns=%s\n' "$(now_ns)" > "$run_dir/status.txt"
set +e
setsid "${payload_cmd[@]}" > "$stdout_fifo" 2> "$stderr_fifo" &
bench_pid=$!; bench_pgid=$bench_pid
sample_smi "$bench_pgid" "$phase_file" > "$run_dir/rocm-smi.log" 9>&- & smi_pid=$!
watch_group "$bench_pgid" "$phase_file" 9>&- & watchdog_pid=$!
wait "$bench_pid"; rc=$?
if group_alive "$bench_pgid"; then terminate_group_now "$bench_pgid"; fi
wait "$stderr_pid" 2>/dev/null; stderr_consumer_rc=$?; stderr_pid=""
wait "$stdout_pid" 2>/dev/null; stdout_consumer_rc=$?; stdout_pid=""
for child in "$smi_pid" "$watchdog_pid"; do kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; done
smi_pid=""; watchdog_pid=""; bench_pid=""; bench_pgid=""
set -e
rm -f "$stderr_fifo" "$stdout_fifo"
trap - EXIT INT TERM HUP

truncated=0
timeout_phase=none
if [[ -f "$run_dir/measurement-timeout" ]]; then truncated=1; timeout_phase=measurement; fi
if [[ -f "$run_dir/setup-timeout" ]]; then truncated=1; timeout_phase=setup; fi
{
    printf 'process_exit_code=%s\n' "$rc"
    printf 'stderr_consumer_exit_code=%s\n' "$stderr_consumer_rc"
    printf 'stdout_consumer_exit_code=%s\n' "$stdout_consumer_rc"
    printf 'truncated=%s\n' "$truncated"
    printf 'timeout_phase=%s\n' "$timeout_phase"
    printf 'finished_at_ns=%s\n' "$(now_ns)"
} >> "$run_dir/status.txt"

python3 - "$run_dir/clock-domain.txt" <<'PY'
from pathlib import Path
import sys, time
before = time.time_ns(); mono = time.clock_gettime_ns(time.CLOCK_MONOTONIC); after = time.time_ns()
real = (before + after) // 2
with Path(sys.argv[1]).open("a") as out:
    out.write(f"end_captured_realtime_ns={real}\nend_captured_monotonic_ns={mono}\nend_realtime_minus_monotonic_ns={real-mono}\nend_calibration_span_ns={after-before}\n")
PY
rocm-smi --showuse --showmemuse --showpower --showclocks --showtemp --showmaxpower --showperflevel --showprofile --showoverdrive --showmemoverdrive > "$run_dir/rocm-smi-final.txt" 2>&1 || true

if [[ "$stderr_consumer_rc" -ne 0 || "$stdout_consumer_rc" -ne 0 ]]; then
    echo "Benchmark output capture failed (stderr=$stderr_consumer_rc stdout=$stdout_consumer_rc); see $run_dir" >&2
    exit 2
fi
if [[ ! -s "$run_dir/result.jsonl" ]]; then
    echo "Benchmark produced no complete result; see $run_dir/bench.log" >&2
    [[ "$timeout_phase" == setup ]] && exit 124
    [[ "$timeout_phase" == measurement ]] && exit 3
    [[ "$rc" -ne 0 ]] && exit "$rc"
    exit 1
fi
if [[ "$rc" -ne 0 && "$truncated" -ne 1 ]]; then
    echo "Benchmark failed with exit $rc; see $run_dir/bench.log" >&2
    exit "$rc"
fi

if [[ "$MODE" == performance ]]; then
    summary_args=(
        "$run_dir/result.jsonl"
        --json "$run_dir/summary.json"
        --tsv "$run_dir/summary.tsv"
        --expected-depths "$DEPTHS"
        --expected-gen "$N_GEN"
        --expected-reps "$RAW_REPS"
        --discard-first "$DISCARD_FIRST"
        --stability-limit "$STABILITY_LIMIT"
        --expected-batch "$BATCH"
        --expected-ubatch "$UBATCH"
        --expected-tensor-split "$TENSOR_SPLIT"
    )
    [[ "$truncated" -eq 1 ]] && summary_args+=(--truncated --allow-trailing-partial)
    python3 "$ROOT_DIR/scripts/dsv4-rocm/summarize-tg.py" "${summary_args[@]}"
    cat "$run_dir/summary.tsv"
    read -r complete stable < <(python3 - "$run_dir/summary.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
print(int(value["complete"]), int(value["stable"]))
PY
)
    if [[ "$complete" -ne 1 ]]; then
        echo "INCOMPLETE raw-TG sweep; no baseline/CSA decision is allowed. Artifacts: $run_dir"
        exit 3
    fi
    if [[ "$PROFILE" == kernel ]]; then
        profile_parser="$ROOT_DIR/scripts/dsv4-rocm/summarize-tg-profile.py"
        git -C "$ROOT_DIR" log -1 --format=%H -- "$profile_parser" > "$run_dir/profile-parser-commit.txt"
        {
            printf '#!/usr/bin/env bash\nset -Eeuo pipefail\n'
            printf 'cd %q\n' "$ROOT_DIR"
            printf 'test "$(git log -1 --format=%%H -- %q)" = "$(cat %q)"\n' \
                "$profile_parser" "$run_dir/profile-parser-commit.txt"
            printf 'python3 %q %q --json %q --tsv %q > %q\n' \
                "$profile_parser" "$run_dir" "$run_dir/profile-summary.json" \
                "$run_dir/profile-families.tsv" "$run_dir/profile-summary.txt"
        } > "$run_dir/profile-parser-command.sh"
        chmod +x "$run_dir/profile-parser-command.sh"
        "$run_dir/profile-parser-command.sh"
        cat "$run_dir/profile-summary.txt"
    fi
    if [[ "$stable" -ne 1 ]]; then
        echo "UNSTABLE raw-TG wall timing; selected-region attribution is preserved but cannot establish TG or decide CSA. Increase repetitions only; keep tg32 and the predeclared discard unchanged. Artifacts: $run_dir"
        exit 4
    fi
else
    python3 "$ROOT_DIR/scripts/dsv4-rocm/parse-sched-debug.py" "$run_dir/bench.log" \
        --depths "$DEPTHS" --expected-nodes "$EXPECTED_DSV4_NODES" \
        --json "$run_dir/scheduler-summary.json" --tsv "$run_dir/scheduler-summary.tsv"
    cat "$run_dir/scheduler-summary.tsv"
    read -r complete resident < <(python3 - "$run_dir/scheduler-summary.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
print(int(value["complete"]), int(value["rocm_residency_ok"]))
PY
)
    if [[ "$complete" -ne 1 ]]; then
        echo "INCOMPLETE scheduler audit; inspect logs. Artifacts: $run_dir"
        exit 3
    fi
    if [[ "$resident" -ne 1 ]]; then
        echo "OBSERVED residency defect: this is valid pre-fix evidence but not a deployment baseline. Artifacts: $run_dir"
        exit 4
    fi
fi

echo "Artifacts: $run_dir"