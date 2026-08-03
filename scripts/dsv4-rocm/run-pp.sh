#!/usr/bin/env bash
# Reproducible, safety-guarded DeepSeek-V4 prompt-processing benchmark for ROCm.
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=${DSV4_MODEL:-/home/edwin/models/DeepSeek-V4-Flash-0731-GGUF/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf}
BENCH=${DSV4_BENCH:-$ROOT_DIR/build/bin/llama-bench}
OUTPUT_ROOT=${DSV4_OUTPUT_ROOT:-$HOME/llama-jobs/dsv4-rocm-pp}
PROMPTS=${DSV4_PROMPTS:-512,2048}
UBATCHES=${DSV4_UBATCHES:-256}
BATCH=${DSV4_BATCH:-512}
REPS=${DSV4_REPS:-3}
TIMEOUT_S=${DSV4_TIMEOUT:-300}
STARTUP_TIMEOUT_S=${DSV4_STARTUP_TIMEOUT:-1200}
TERM_GRACE_S=${DSV4_TERM_GRACE:-2}
THREADS=${DSV4_THREADS:-12}
TENSOR_SPLIT=${DSV4_TENSOR_SPLIT:-1/1/1/1}
CACHE_TYPE_K=${DSV4_CACHE_TYPE_K:-f16}
CACHE_TYPE_V=${DSV4_CACHE_TYPE_V:-f16}
LOAD_MODE=${DSV4_LOAD_MODE:-mmap}
PROFILE=${DSV4_PROFILE:-none}
NO_WARMUP=${DSV4_NO_WARMUP:-0}
ALLOW_BUSY=${DSV4_ALLOW_BUSY_GPUS:-0}
DRY_RUN=0
LABEL=${DSV4_LABEL:-baseline}

usage() {
    cat <<'USAGE'
Usage: scripts/dsv4-rocm/run-pp.sh [--dry-run]

The script uses llama-bench for PP-only measurements, records raw samples and a
machine/source/model manifest, and fails closed if ROCm process discovery is
unavailable or any GPU process is active. It never stops a pre-existing process.

Important environment overrides:
  DSV4_MODEL              first GGUF shard
  DSV4_BENCH              llama-bench binary
  DSV4_PROMPTS            comma-separated prompt token counts (default 512,2048)
  DSV4_UBATCHES           comma-separated ubatches (default 256)
  DSV4_BATCH              logical batch (default 512)
  DSV4_REPS               measured repetitions per shape (default 3)
  DSV4_TIMEOUT            absolute measured-loop cap; excludes initial load/warmup (default 300 seconds)
  DSV4_STARTUP_TIMEOUT    separate load/first-warmup safety cap (default 1200 seconds)
  DSV4_TERM_GRACE         TERM lead time before each absolute cap (default 2 seconds)
  DSV4_TENSOR_SPLIT       slash-separated split for llama-bench (default 1/1/1/1)
  DSV4_OUTPUT_ROOT        artifact root (default $HOME/llama-jobs/dsv4-rocm-pp)
  DSV4_LABEL              safe run label
  DSV4_PROFILE=trace      full rocprofv3 runtime trace (CSV+JSON)
  DSV4_PROFILE=kernel     compact kernel/memory-copy/RCCL trace (CSV only)
  DSV4_NO_WARMUP=1        disable llama-bench warmup (profile wrapper default only)
  DSV4_ALLOW_BUSY_GPUS=1  override safety refusal (never use for controlled A/B)
  DSV4_HASH_MODE=full     hash every GGUF shard; default records path/size/mtime only
  DSV4_LIBRARY_PATH       selected binary's DSO directory (default: llama-bench directory)

A measurement-truncated run preserves valid complete JSONL records but exits 3.
Production baselines should never set DSV4_ALLOW_BUSY_GPUS=1.
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
for pair in "DSV4_REPS:$REPS" "DSV4_BATCH:$BATCH" "DSV4_TIMEOUT:$TIMEOUT_S" \
            "DSV4_STARTUP_TIMEOUT:$STARTUP_TIMEOUT_S" "DSV4_THREADS:$THREADS"; do
    name=${pair%%:*}
    value=${pair#*:}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer"
done
[[ "$TERM_GRACE_S" =~ ^[0-9]+$ ]] || fail "DSV4_TERM_GRACE must be a non-negative integer"
[ "$TERM_GRACE_S" -lt "$TIMEOUT_S" ] || fail "DSV4_TERM_GRACE must be less than DSV4_TIMEOUT"
[ "$TERM_GRACE_S" -lt "$STARTUP_TIMEOUT_S" ] || fail "DSV4_TERM_GRACE must be less than DSV4_STARTUP_TIMEOUT"
[[ "$PROFILE" == none || "$PROFILE" == trace || "$PROFILE" == kernel ]] || fail "DSV4_PROFILE must be none, trace, or kernel"
[[ "$NO_WARMUP" == 0 || "$NO_WARMUP" == 1 ]] || fail "DSV4_NO_WARMUP must be 0 or 1"
[[ "$ALLOW_BUSY" == 0 || "$ALLOW_BUSY" == 1 ]] || fail "DSV4_ALLOW_BUSY_GPUS must be 0 or 1"
[ -f "$MODEL" ] || fail "model not found: $MODEL"
[ -x "$BENCH" ] || fail "llama-bench not executable: $BENCH"
for tool in awk date flock grep python3 readlink rocm-smi setsid sha256sum; do
    command -v "$tool" >/dev/null || fail "$tool is required"
done
if [[ "$PROFILE" != none ]]; then
    command -v rocprofv3 >/dev/null || fail "rocprofv3 is required for DSV4_PROFILE=$PROFILE"
fi

BENCH=$(readlink -f "$BENCH")
MODEL=$(readlink -f "$MODEL")
LIBRARY_PATH=${DSV4_LIBRARY_PATH:-$(dirname "$BENCH")}
export LD_LIBRARY_PATH="$LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

IFS=',' read -r -a prompt_values <<< "$PROMPTS"
IFS=',' read -r -a ubatch_values <<< "$UBATCHES"
for value in "${prompt_values[@]}" "${ubatch_values[@]}"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "prompt and ubatch values must be positive integers (got '$value')"
done
for value in "${ubatch_values[@]}"; do
    [ "$value" -le "$BATCH" ] || fail "ubatch $value exceeds batch $BATCH"
done

bench_cmd=(
    "$BENCH"
    --model "$MODEL"
    --n-prompt "$PROMPTS"
    --n-gen 0
    --batch-size "$BATCH"
    --ubatch-size "$UBATCHES"
    --cache-type-k "$CACHE_TYPE_K"
    --cache-type-v "$CACHE_TYPE_V"
    --threads "$THREADS"
    --n-gpu-layers 999
    --split-mode tensor
    --tensor-split "$TENSOR_SPLIT"
    --flash-attn on
    --load-mode "$LOAD_MODE"
    --repetitions "$REPS"
    --output jsonl
    --progress
)
if [ "$NO_WARMUP" -eq 1 ]; then
    bench_cmd+=(--no-warmup)
fi

printf 'Planned command:'
printf ' %q' "${bench_cmd[@]}"
printf '\n'
printf 'Library path: %s\n' "$LD_LIBRARY_PATH"
printf 'Measurement cap: %ss absolute, initial model load/warmup excluded\n' "$TIMEOUT_S"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run only; no ROCm query or GPU process was started."
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
        echo "WARNING: proceeding after failed GPU discovery because DSV4_ALLOW_BUSY_GPUS=1" >&2
        return 0
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
    flock -n 9 || fail "GPU job lock is held; wait for the current job or use scripts/rdna2-job.sh"
fi
check_gpus_idle "initial safety check"

commit=nogit
if git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    commit=$(git -C "$ROOT_DIR" rev-parse --short=12 HEAD)
fi
run_id="$(date -u +%Y%m%dT%H%M%S.%NZ)-${LABEL}-${commit}-$RANDOM"
run_dir="$OUTPUT_ROOT/$run_id"
mkdir "$run_dir"

python3 - "$run_dir/clock-domain.txt" <<'PY'
from pathlib import Path
import time
import sys

realtime_before = time.time_ns()
monotonic = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
realtime_after = time.time_ns()
realtime = (realtime_before + realtime_after) // 2
boot_id_path = Path("/proc/sys/kernel/random/boot_id")
boot_id = boot_id_path.read_text().strip() if boot_id_path.is_file() else "unavailable"
Path(sys.argv[1]).write_text(
    f"boot_id={boot_id}\n"
    f"start_captured_realtime_ns={realtime}\n"
    f"start_captured_monotonic_ns={monotonic}\n"
    f"start_realtime_minus_monotonic_ns={realtime - monotonic}\n"
    f"start_calibration_span_ns={realtime_after - realtime_before}\n"
)
PY

printf '%q ' "${bench_cmd[@]}" > "$run_dir/command.sh"
printf '\n' >> "$run_dir/command.sh"
chmod +x "$run_dir/command.sh"

export GGML_CUDA_ALLREDUCE=${GGML_CUDA_ALLREDUCE:-nccl}
export GGML_CUDA_P2P=${GGML_CUDA_P2P:-1}
export GGML_HIP_GRAPHS=${GGML_HIP_GRAPHS:-1}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-10.3.0}

{
    printf 'DSV4_MODEL=%q\n' "$MODEL"
    printf 'DSV4_BENCH=%q\n' "$BENCH"
    printf 'DSV4_PROMPTS=%q\n' "$PROMPTS"
    printf 'DSV4_UBATCHES=%q\n' "$UBATCHES"
    printf 'DSV4_BATCH=%q\n' "$BATCH"
    printf 'DSV4_REPS=%q\n' "$REPS"
    printf 'DSV4_TIMEOUT=%q\n' "$TIMEOUT_S"
    printf 'DSV4_STARTUP_TIMEOUT=%q\n' "$STARTUP_TIMEOUT_S"
    printf 'DSV4_TERM_GRACE=%q\n' "$TERM_GRACE_S"
    printf 'DSV4_PROFILE=%q\n' "$PROFILE"
    printf 'DSV4_NO_WARMUP=%q\n' "$NO_WARMUP"
    printf 'DSV4_ALLOW_BUSY_GPUS=%q\n' "$ALLOW_BUSY"
    printf 'DSV4_LIBRARY_PATH=%q\n' "$LIBRARY_PATH"
    printf 'LD_LIBRARY_PATH=%q\n' "$LD_LIBRARY_PATH"
    for name in GGML_HIP_RDNA2_MMQ_J GGML_HIP_RDNA2_HC_MIXES GGML_HIP_RDNA2_LID_SUBWAVE; do
        if declare -p "$name" >/dev/null 2>&1; then
            printf '%s=%q\n' "$name" "${!name}"
        else
            printf 'unset %s\n' "$name"
        fi
    done
} > "$run_dir/effective-settings.sh"

"$ROOT_DIR/scripts/dsv4-rocm/manifest.sh" "$run_dir" "$BENCH" "$MODEL"

payload_cmd=("${bench_cmd[@]}")
if [[ "$PROFILE" != none ]]; then
    mkdir "$run_dir/rocprof"
    profile_args=(
        --output-directory "$run_dir/rocprof"
        --output-file dsv4-pp
        --stats
        --summary
        --summary-per-domain
        --summary-output-file "$run_dir/rocprof-summary.txt"
    )
    if [[ "$PROFILE" == trace ]]; then
        profile_args+=(--output-format csv json --runtime-trace)
    else
        # Kernel dispatches dominate full-trace size. This compact mode keeps the
        # measured-region attribution inputs without HIP API events or multi-GB JSON.
        profile_args+=(--output-format csv --kernel-trace --memory-copy-trace --rccl-trace)
    fi
    payload_cmd=(rocprofv3 "${profile_args[@]}" -- "${bench_cmd[@]}")
fi
printf '%q ' "${payload_cmd[@]}" > "$run_dir/executed-command.sh"
printf '\n' >> "$run_dir/executed-command.sh"
chmod +x "$run_dir/executed-command.sh"

# Recheck immediately before launch; manifest hashing or source capture can take time.
check_gpus_idle "pre-launch safety check"

now_ns() {
    date +%s%N
}

group_alive() {
    kill -0 -- "-$1" 2>/dev/null
}

signal_group() {
    local signal=$1 pgid=$2
    kill "-$signal" -- "-$pgid" 2>/dev/null || true
}

terminate_group_now() {
    local pgid=$1
    group_alive "$pgid" || return 0
    signal_group TERM "$pgid"
    local deadline=$(( $(now_ns) + 2000000000 ))
    while group_alive "$pgid" && [ "$(now_ns)" -lt "$deadline" ]; do
        sleep 0.05
    done
    group_alive "$pgid" && signal_group KILL "$pgid"
}

sample_smi() {
    local pgid=$1
    while group_alive "$pgid"; do
        printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
        rocm-smi --showuse --showmemuse --showpower --showclocks --csv 2>&1 || true
        sleep 1
    done
}

stderr_consumer() {
    local fifo=$1 log_file=$2 marker_file=$3
    while IFS= read -r line || [[ -n "$line" ]]; do
        printf '%s\n' "$line" >> "$log_file"
        if [[ ! -e "$marker_file" && "$line" =~ llama-bench:\ benchmark\ [0-9]+/[0-9]+:\ prompt\ run\ 1/[0-9]+ ]]; then
            local stamp
            stamp=$(now_ns)
            (set -o noclobber; printf '%s\n' "$stamp" > "$marker_file") 2>/dev/null || true
            printf 'measurement_started_at_ns=%s\n' "$stamp" >> "$run_dir/status.txt"
        fi
    done < "$fifo"
}

stdout_consumer() {
    local fifo=$1 result_file=$2 completion_file=$3
    while IFS= read -r line || [[ -n "$line" ]]; do
        printf '%s\n' "$line" >> "$result_file"
        printf '%s\n' "$(now_ns)" >> "$completion_file"
    done < "$fifo"
}

watch_group() {
    local pgid=$1 marker_file=$2
    local startup_start hard_deadline term_deadline now
    startup_start=$(now_ns)
    while group_alive "$pgid"; do
        if [ -s "$marker_file" ]; then
            local measurement_start
            measurement_start=$(<"$marker_file")
            hard_deadline=$(( measurement_start + TIMEOUT_S * 1000000000 ))
            term_deadline=$(( hard_deadline - TERM_GRACE_S * 1000000000 ))
            while group_alive "$pgid"; do
                now=$(now_ns)
                if [ "$now" -ge "$term_deadline" ]; then
                    : > "$run_dir/measurement-timeout"
                    printf 'measurement_term_at_ns=%s\n' "$now" >> "$run_dir/status.txt"
                    signal_group TERM "$pgid"
                    while group_alive "$pgid"; do
                        now=$(now_ns)
                        if [ "$now" -ge "$hard_deadline" ]; then
                            printf 'measurement_kill_at_ns=%s\n' "$now" >> "$run_dir/status.txt"
                            signal_group KILL "$pgid"
                            return 0
                        fi
                        sleep 0.05
                    done
                    return 0
                fi
                sleep 0.05
            done
            return 0
        fi
        now=$(now_ns)
        hard_deadline=$(( startup_start + STARTUP_TIMEOUT_S * 1000000000 ))
        term_deadline=$(( hard_deadline - TERM_GRACE_S * 1000000000 ))
        if [ "$now" -ge "$term_deadline" ]; then
            : > "$run_dir/startup-timeout"
            printf 'startup_term_at_ns=%s\n' "$now" >> "$run_dir/status.txt"
            signal_group TERM "$pgid"
            while group_alive "$pgid"; do
                now=$(now_ns)
                if [ "$now" -ge "$hard_deadline" ]; then
                    printf 'startup_kill_at_ns=%s\n' "$now" >> "$run_dir/status.txt"
                    signal_group KILL "$pgid"
                    return 0
                fi
                sleep 0.05
            done
            return 0
        fi
        sleep 0.05
    done
}

bench_pid=""
bench_pgid=""
smi_pid=""
watchdog_pid=""
stderr_pid=""
stdout_pid=""
stderr_fifo="$run_dir/bench.stderr.fifo"
stdout_fifo="$run_dir/bench.stdout.fifo"
cleanup_children() {
    local exit_code=$?
    trap - EXIT INT TERM HUP
    for child_pid in "$smi_pid" "$watchdog_pid"; do
        if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
            kill "$child_pid" 2>/dev/null || true
        fi
    done
    if [[ -n "$bench_pgid" ]]; then
        terminate_group_now "$bench_pgid"
    fi
    if [[ -n "$bench_pid" ]]; then
        wait "$bench_pid" 2>/dev/null || true
    fi
    for child_pid in "$stderr_pid" "$stdout_pid" "$smi_pid" "$watchdog_pid"; do
        if [[ -n "$child_pid" ]]; then
            kill "$child_pid" 2>/dev/null || true
            wait "$child_pid" 2>/dev/null || true
        fi
    done
    rm -f "$stderr_fifo" "$stdout_fifo"
    exit "$exit_code"
}
trap cleanup_children EXIT INT TERM HUP

mkfifo "$stderr_fifo" "$stdout_fifo"
: > "$run_dir/bench.log"
: > "$run_dir/result.jsonl"
: > "$run_dir/result-completed-at.ns"
stderr_consumer "$stderr_fifo" "$run_dir/bench.log" "$run_dir/measurement-start.ns" &
stderr_pid=$!
stdout_consumer "$stdout_fifo" "$run_dir/result.jsonl" "$run_dir/result-completed-at.ns" &
stdout_pid=$!

printf 'run_dir=%s\n' "$run_dir"
printf 'started_at_ns=%s\n' "$(now_ns)" > "$run_dir/status.txt"
set +e
setsid "${payload_cmd[@]}" > "$stdout_fifo" 2> "$stderr_fifo" &
bench_pid=$!
bench_pgid=$bench_pid
sample_smi "$bench_pgid" > "$run_dir/rocm-smi.log" &
smi_pid=$!
watch_group "$bench_pgid" "$run_dir/measurement-start.ns" &
watchdog_pid=$!
wait "$bench_pid"
rc=$?
# The leader can exit before a trace-wrapper child. Never clear PGID ownership
# until the entire group is gone or has been forcibly terminated.
if group_alive "$bench_pgid"; then
    terminate_group_now "$bench_pgid"
fi
wait "$stderr_pid" 2>/dev/null || true
stderr_pid=""
wait "$stdout_pid" 2>/dev/null || true
stdout_pid=""
for child_pid in "$smi_pid" "$watchdog_pid"; do
    kill "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
done
smi_pid=""
watchdog_pid=""
bench_pid=""
bench_pgid=""
set -e
rm -f "$stderr_fifo" "$stdout_fifo"
trap - EXIT INT TERM HUP

truncated=0
[ -f "$run_dir/measurement-timeout" ] && truncated=1
{
    printf 'process_exit_code=%s\n' "$rc"
    printf 'truncated=%s\n' "$truncated"
    printf 'finished_at_ns=%s\n' "$(now_ns)"
} >> "$run_dir/status.txt"

python3 - "$run_dir/clock-domain.txt" <<'PY'
from pathlib import Path
import time
import sys

realtime_before = time.time_ns()
monotonic = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
realtime_after = time.time_ns()
realtime = (realtime_before + realtime_after) // 2
with Path(sys.argv[1]).open("a") as handle:
    handle.write(
        f"end_captured_realtime_ns={realtime}\n"
        f"end_captured_monotonic_ns={monotonic}\n"
        f"end_realtime_minus_monotonic_ns={realtime - monotonic}\n"
        f"end_calibration_span_ns={realtime_after - realtime_before}\n"
    )
PY

if [ -f "$run_dir/startup-timeout" ]; then
    echo "Benchmark did not reach its first measured prompt run within ${STARTUP_TIMEOUT_S}s; see $run_dir/bench.log" >&2
    exit 124
fi
if [ ! -s "$run_dir/result.jsonl" ]; then
    echo "Benchmark produced no complete result before exit $rc; see $run_dir/bench.log" >&2
    [ "$rc" -ne 0 ] && exit "$rc"
    exit 1
fi
if [ "$rc" -ne 0 ] && [ "$truncated" -ne 1 ]; then
    echo "Benchmark failed with exit code $rc; see $run_dir/bench.log" >&2
    exit "$rc"
fi

summary_args=(
    "$run_dir/result.jsonl"
    --json "$run_dir/summary.json"
    --tsv "$run_dir/summary.tsv"
    --expected-prompts "$PROMPTS"
    --expected-ubatches "$UBATCHES"
    --expected-reps "$REPS"
)
if [ "$truncated" -eq 1 ]; then
    summary_args+=(--truncated --allow-trailing-partial)
fi
python3 "$ROOT_DIR/scripts/dsv4-rocm/summarize.py" "${summary_args[@]}"
cat "$run_dir/summary.tsv"

complete=$(python3 - "$run_dir/summary.json" <<'PY'
import json, sys
with open(sys.argv[1]) as handle:
    print("1" if json.load(handle)["complete"] else "0")
PY
)
if [ "$truncated" -eq 1 ] || [ "$complete" -ne 1 ]; then
    echo "INCOMPLETE: measured-loop cap reached or expected shapes/repetitions are missing; compare only matching complete rows."
    echo "Artifacts: $run_dir"
    exit 3
fi

echo "Artifacts: $run_dir"