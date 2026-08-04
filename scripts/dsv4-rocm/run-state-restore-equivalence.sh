#!/usr/bin/env bash
# Safety-guarded fresh-prefill versus restored sequence-state equivalence gate.
set -Eeuo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL=${DSV4_MODEL:-/home/edwin/models/DeepSeek-V4-Flash-0731-GGUF/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf}
BINARY=${DSV4_STATE_BINARY:-$ROOT_DIR/build/bin/test-state-restore-equivalence}
OUTPUT_ROOT=${DSV4_STATE_OUTPUT_ROOT:-$HOME/llama-jobs/dsv4-rocm-state-equivalence}
DEPTHS=${DSV4_STATE_DEPTHS:-2048,3072,16384}
N_GEN=${DSV4_STATE_N_GEN:-4}
SEED=${DSV4_STATE_SEED:-12345}
ABS_TOL=${DSV4_STATE_ABS_TOL:-1e-5}
REL_TOL=${DSV4_STATE_REL_TOL:-1e-5}
STATE_API=${DSV4_STATE_API:-sequence}
TIMEOUT_S=${DSV4_STATE_TIMEOUT:-1800}
THREADS=${DSV4_THREADS:-12}
BATCH=${DSV4_BATCH:-512}
UBATCH=${DSV4_UBATCH:-256}
TENSOR_SPLIT=${DSV4_TENSOR_SPLIT:-1,1,1,1}
ALLOW_BUSY=${DSV4_ALLOW_BUSY_GPUS:-0}
LABEL=${DSV4_LABEL:-state-restore-equivalence}
DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage: scripts/dsv4-rocm/run-state-restore-equivalence.sh [--dry-run]

The model is loaded once. At each requested depth, the tool:
  1. builds a deterministic target-only prefix from scratch;
  2. saves sequence 0 with llama_state_seq_get_data;
  3. runs a short greedy continuation and captures every full-vocabulary logit;
  4. clears memory, recomputes the exact prefix, and replays those inputs as a
     fresh-repeat control for ordinary backend repeatability;
  5. clears memory exactly as llama-bench does, restores with
     llama_state_seq_set_data, and replays the same continuation inputs;
  6. requires all three argmax paths to match and both full-logit comparisons
     to satisfy abs_tol + rel_tol*max(abs(a),abs(b)).

Defaults:
  DSV4_STATE_DEPTHS=2048,3072,16384
  DSV4_STATE_N_GEN=4
  DSV4_STATE_SEED=12345
  DSV4_STATE_ABS_TOL=1e-5
  DSV4_STATE_REL_TOL=1e-5
  DSV4_STATE_API=sequence      sequence or full context state
  DSV4_STATE_TIMEOUT=1800

No draft model, sampler, DSpark, MTP, or speculative path is used. Raw JSON
preserves exact prefix/input/argmax token IDs and full-logit comparison counts.
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

[[ "$LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || fail "invalid DSV4_LABEL: $LABEL"
for pair in "DSV4_STATE_N_GEN:$N_GEN" "DSV4_STATE_TIMEOUT:$TIMEOUT_S" "DSV4_THREADS:$THREADS" "DSV4_BATCH:$BATCH" "DSV4_UBATCH:$UBATCH"; do
    name=${pair%%:*}; value=${pair#*:}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer"
done
[[ "$SEED" =~ ^[0-9]+$ ]] || fail "DSV4_STATE_SEED must be a non-negative integer"
[[ "$ALLOW_BUSY" == 0 || "$ALLOW_BUSY" == 1 ]] || fail "DSV4_ALLOW_BUSY_GPUS must be 0 or 1"
[[ "$STATE_API" == sequence || "$STATE_API" == context ]] || fail "DSV4_STATE_API must be sequence or context"
(( UBATCH <= BATCH )) || fail "ubatch $UBATCH exceeds batch $BATCH"
if ! python3 - "$DEPTHS" "$ABS_TOL" "$REL_TOL" <<'PY'
import sys
values = [int(v) for v in sys.argv[1].split(',')]
assert values and len(values) == len(set(values)) and all(v > 0 for v in values)
assert float(sys.argv[2]) >= 0 and float(sys.argv[3]) >= 0
PY
then
    fail "invalid depths or tolerances"
fi

export GGML_HIP_RDNA2_MMQ_J=${GGML_HIP_RDNA2_MMQ_J:-16}
export GGML_HIP_RDNA2_HC_MIXES=${GGML_HIP_RDNA2_HC_MIXES:-1}
export GGML_HIP_RDNA2_LID_SUBWAVE=${GGML_HIP_RDNA2_LID_SUBWAVE:-4}
[[ "$GGML_HIP_RDNA2_MMQ_J" == 16 ]] || fail "gate requires GGML_HIP_RDNA2_MMQ_J=16"
[[ "$GGML_HIP_RDNA2_HC_MIXES" == 1 ]] || fail "gate requires GGML_HIP_RDNA2_HC_MIXES=1"
[[ "$GGML_HIP_RDNA2_LID_SUBWAVE" == 4 ]] || fail "gate requires GGML_HIP_RDNA2_LID_SUBWAVE=4"

[[ -f "$MODEL" ]] || fail "model not found: $MODEL"
[[ -x "$BINARY" ]] || fail "test binary not executable: $BINARY"
for tool in awk date flock python3 readlink rocm-smi setsid sha256sum timeout; do
    command -v "$tool" >/dev/null || fail "$tool is required"
done
MODEL=$(readlink -f "$MODEL")
BINARY=$(readlink -f "$BINARY")
LIBRARY_PATH=${DSV4_LIBRARY_PATH:-$(dirname "$BINARY")}
export LD_LIBRARY_PATH="$LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_CUDA_ALLREDUCE=${GGML_CUDA_ALLREDUCE:-nccl}
export GGML_CUDA_P2P=${GGML_CUDA_P2P:-1}
export GGML_HIP_GRAPHS=${GGML_HIP_GRAPHS:-1}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-10.3.0}
export DSV4_STATE_DEPTHS="$DEPTHS"
export DSV4_STATE_N_GEN="$N_GEN"
export DSV4_STATE_SEED="$SEED"
export DSV4_STATE_ABS_TOL="$ABS_TOL"
export DSV4_STATE_REL_TOL="$REL_TOL"
export DSV4_STATE_API="$STATE_API"

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
printf 'Environment: DSV4_STATE_DEPTHS=%q DSV4_STATE_N_GEN=%q DSV4_STATE_SEED=%q DSV4_STATE_ABS_TOL=%q DSV4_STATE_REL_TOL=%q DSV4_STATE_API=%q\n' \
    "$DEPTHS" "$N_GEN" "$SEED" "$ABS_TOL" "$REL_TOL" "$STATE_API"
if [[ "$DRY_RUN" == 1 ]]; then
    echo "Dry run only; no ROCm query, model load, or test process was started."
    exit 0
fi

check_gpus_idle() {
    local phase=$1 output rc busy
    set +e; output=$(rocm-smi --showpids 2>&1); rc=$?; set -e
    if [[ $rc -ne 0 ]]; then
        printf 'rocm-smi --showpids failed during %s (exit %s):\n%s\n' "$phase" "$rc" "$output" >&2
        [[ "$ALLOW_BUSY" == 1 ]] || fail "cannot prove GPUs are idle"
        return
    fi
    busy=$(printf '%s\n' "$output" | awk '$1 ~ /^[0-9]+$/ { print }')
    if [[ -n "$busy" ]]; then
        printf 'ROCm reports active GPU processes during %s:\n%s\n' "$phase" "$busy" >&2
        [[ "$ALLOW_BUSY" == 1 ]] || fail "refusing to use busy GPUs"
    fi
}

mkdir -p "$OUTPUT_ROOT" "$HOME/llama-jobs"
if [[ -z ${LLAMA_JOB_DIR:-} ]]; then
    exec 9>"$HOME/llama-jobs/gpu.lock"
    flock -n 9 || fail "GPU job lock is held"
fi
check_gpus_idle "initial safety check"

commit=nogit
if git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    commit=$(git -C "$ROOT_DIR" rev-parse --short=12 HEAD)
fi
run_id="$(date -u +%Y%m%dT%H%M%S.%NZ)-${LABEL}-${commit}-$RANDOM"
run_dir="$OUTPUT_ROOT/$run_id"
mkdir "$run_dir"

printf 'env DSV4_STATE_DEPTHS=%q DSV4_STATE_N_GEN=%q DSV4_STATE_SEED=%q DSV4_STATE_ABS_TOL=%q DSV4_STATE_REL_TOL=%q DSV4_STATE_API=%q ' \
    "$DEPTHS" "$N_GEN" "$SEED" "$ABS_TOL" "$REL_TOL" "$STATE_API" > "$run_dir/command.sh"
printf '%q ' "${command[@]}" >> "$run_dir/command.sh"
printf '\n' >> "$run_dir/command.sh"
chmod +x "$run_dir/command.sh"
{
    printf 'DSV4_STATE_DEPTHS=%q\n' "$DEPTHS"
    printf 'DSV4_STATE_N_GEN=%q\n' "$N_GEN"
    printf 'DSV4_STATE_SEED=%q\n' "$SEED"
    printf 'DSV4_STATE_ABS_TOL=%q\n' "$ABS_TOL"
    printf 'DSV4_STATE_REL_TOL=%q\n' "$REL_TOL"
    printf 'DSV4_STATE_API=%q\n' "$STATE_API"
    printf 'DSV4_STATE_TIMEOUT=%q\n' "$TIMEOUT_S"
    printf 'DSV4_BATCH=%q\n' "$BATCH"
    printf 'DSV4_UBATCH=%q\n' "$UBATCH"
    printf 'DSV4_TENSOR_SPLIT=%q\n' "$TENSOR_SPLIT"
    printf 'GGML_HIP_RDNA2_MMQ_J=%q\n' "$GGML_HIP_RDNA2_MMQ_J"
    printf 'GGML_HIP_RDNA2_HC_MIXES=%q\n' "$GGML_HIP_RDNA2_HC_MIXES"
    printf 'GGML_HIP_RDNA2_LID_SUBWAVE=%q\n' "$GGML_HIP_RDNA2_LID_SUBWAVE"
    printf 'GGML_CUDA_ALLREDUCE=%q\n' "$GGML_CUDA_ALLREDUCE"
    printf 'GGML_CUDA_P2P=%q\n' "$GGML_CUDA_P2P"
    printf 'GGML_HIP_GRAPHS=%q\n' "$GGML_HIP_GRAPHS"
    printf 'HSA_NO_SCRATCH_RECLAIM=%q\n' "$HSA_NO_SCRATCH_RECLAIM"
    printf 'HSA_OVERRIDE_GFX_VERSION=%q\n' "$HSA_OVERRIDE_GFX_VERSION"
    printf 'LD_LIBRARY_PATH=%q\n' "$LD_LIBRARY_PATH"
} > "$run_dir/effective-settings.sh"
"$ROOT_DIR/scripts/dsv4-rocm/manifest.sh" "$run_dir" "$BINARY" "$MODEL"
check_gpus_idle "pre-launch safety check"

now_ns() { date +%s%N; }
group_alive() { kill -0 -- "-$1" 2>/dev/null; }
terminate_group() {
    local pgid=$1 deadline
    group_alive "$pgid" || return
    kill -TERM -- "-$pgid" 2>/dev/null || true
    deadline=$(( $(now_ns) + 5000000000 ))
    while group_alive "$pgid" && [[ $(now_ns) -lt $deadline ]]; do sleep 0.05; done
    group_alive "$pgid" && kill -KILL -- "-$pgid" 2>/dev/null || true
}
sample_smi() {
    local pgid=$1
    while group_alive "$pgid"; do
        printf 'timestamp=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
        rocm-smi --showuse --showmemuse --showpower --showclocks --showtemp --csv 2>&1 || true
        sleep 1
    done
}

pid=""; pgid=""; smi_pid=""
cleanup() {
    local rc=$?
    trap - EXIT INT TERM HUP
    [[ -n "$smi_pid" ]] && kill "$smi_pid" 2>/dev/null || true
    [[ -n "$pgid" ]] && terminate_group "$pgid"
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
    [[ -n "$smi_pid" ]] && wait "$smi_pid" 2>/dev/null || true
    exit "$rc"
}
trap cleanup EXIT INT TERM HUP

printf 'run_dir=%s\n' "$run_dir"
printf 'started_at_ns=%s\n' "$(now_ns)" > "$run_dir/status.txt"
set +e
setsid timeout --signal=TERM --kill-after=5s "${TIMEOUT_S}s" "${command[@]}" \
    > "$run_dir/result.json" 2> "$run_dir/bench.log" &
pid=$!; pgid=$pid
sample_smi "$pgid" > "$run_dir/rocm-smi.log" & smi_pid=$!
wait "$pid"; rc=$?
if group_alive "$pgid"; then terminate_group "$pgid"; fi
kill "$smi_pid" 2>/dev/null || true; wait "$smi_pid" 2>/dev/null || true
pid=""; pgid=""; smi_pid=""
set -e
trap - EXIT INT TERM HUP
printf 'process_exit_code=%s\nfinished_at_ns=%s\n' "$rc" "$(now_ns)" >> "$run_dir/status.txt"
rocm-smi --showuse --showmemuse --showpower --showclocks --showtemp > "$run_dir/rocm-smi-final.txt" 2>&1 || true

if [[ ! -s "$run_dir/result.json" ]]; then
    echo "No result JSON; see $run_dir/bench.log" >&2
    [[ $rc -ne 0 ]] && exit "$rc"
    exit 1
fi
set +e
python3 - "$run_dir/result.json" "$run_dir/summary.json" "$run_dir/summary.tsv" \
    "$DEPTHS" "$N_GEN" "$SEED" "$ABS_TOL" "$REL_TOL" "$BATCH" "$UBATCH" "$STATE_API" <<'PY'
import json, math, pathlib, re, sys
raw, summary_path, tsv_path, expected, n_gen_text, seed_text, abs_text, rel_text, batch_text, ubatch_text, state_api = sys.argv[1:]
def reject_constant(text):
    raise ValueError(f'non-standard JSON constant: {text}')
with open(raw) as handle:
    value = json.load(handle, parse_constant=reject_constant)
expected_depths = [int(v) for v in expected.split(',')]
n_gen, seed = int(n_gen_text), int(seed_text)
abs_tol, rel_tol = float(abs_text), float(rel_text)
batch, ubatch = int(batch_text), int(ubatch_text)
errors = []
def exact_bool(obj, key, expected_value=True, prefix='root'):
    if type(obj.get(key)) is not bool or obj[key] is not expected_value:
        errors.append(f'{prefix}.{key} must be boolean {expected_value}')
exact_bool(value, 'complete')
exact_bool(value, 'accepted')
exact_bool(value, 'target_only')
for key, expected_value in (
    ('state_restore_scope', 'same_context_same_benchmark_instance'),
    ('state_api', state_api),
    ('n_gen', n_gen), ('seed', seed), ('n_batch', batch), ('n_ubatch', ubatch),
    ('cache_type_k', 'f16'), ('cache_type_v', 'f16'), ('flash_attn', 'enabled'),
):
    if value.get(key) != expected_value or (isinstance(expected_value, int) and type(value.get(key)) is not int):
        errors.append(f'root.{key}={value.get(key)!r}, expected {expected_value!r}')
for key, expected_value in (('abs_tolerance', abs_tol), ('rel_tolerance', rel_tol)):
    actual = value.get(key)
    if type(actual) not in (int, float) or not math.isfinite(actual) or actual != expected_value:
        errors.append(f'root.{key}={actual!r}, expected {expected_value!r}')
records = value.get('records')
if not isinstance(records, list):
    errors.append('root.records must be a list')
    records = []
seen = [r.get('depth') if isinstance(r, dict) else None for r in records]
if seen != expected_depths:
    errors.append(f'depths {seen!r}, expected {expected_depths!r}')
hex64 = re.compile(r'^[0-9a-f]{16}$')
for index, (depth, record) in enumerate(zip(expected_depths, records)):
    prefix = f'records[{index}]'
    if not isinstance(record, dict):
        errors.append(f'{prefix} must be an object')
        continue
    exact_bool(record, 'complete', prefix=prefix)
    exact_bool(record, 'accepted', prefix=prefix)
    if type(record.get('state_bytes')) is not int or record['state_bytes'] <= 0:
        errors.append(f'{prefix}.state_bytes must be a positive integer')
    for key in (
        'state_fnv1a64','reprefill_state_fnv1a64','prefix_fnv1a64',
        'original_logits_fnv1a64','fresh_logits_fnv1a64','restored_logits_fnv1a64',
    ):
        if not isinstance(record.get(key), str) or not hex64.fullmatch(record[key]):
            errors.append(f'{prefix}.{key} is not a 16-digit lowercase hex hash')
    for key in (
        'state_bitwise_mismatches', 'repeat_bitwise_mismatches',
        'repeat_tolerance_violations', 'repeat_nonfinite_mismatches',
        'bitwise_mismatches', 'tolerance_violations', 'nonfinite_mismatches',
    ):
        if type(record.get(key)) is not int or record[key] < 0:
            errors.append(f'{prefix}.{key} must be a non-negative integer')
    if record.get('repeat_tolerance_violations') != 0 or record.get('repeat_nonfinite_mismatches') != 0:
        errors.append(f'{prefix} fresh-repeat control has numeric comparison violations')
    if record.get('tolerance_violations') != 0 or record.get('nonfinite_mismatches') != 0:
        errors.append(f'{prefix} restored-state comparison has numeric comparison violations')
    for key in ('repeat_max_abs_diff','repeat_max_rel_diff','max_abs_diff','max_rel_diff'):
        number = record.get(key)
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            errors.append(f'{prefix}.{key} must be finite and non-negative')
    token_arrays = {}
    for key, length in (
        ('prefix_tokens', depth), ('generation_input_tokens', n_gen),
        ('original_argmax_tokens', n_gen), ('fresh_argmax_tokens', n_gen),
        ('restored_argmax_tokens', n_gen),
    ):
        array = record.get(key)
        if not isinstance(array, list) or len(array) != length or any(type(token) is not int for token in array):
            errors.append(f'{prefix}.{key} must contain exactly {length} integer tokens')
        else:
            token_arrays[key] = array
    if token_arrays.get('original_argmax_tokens') != token_arrays.get('fresh_argmax_tokens'):
        errors.append(f'{prefix} original/fresh-repeat argmax token arrays differ')
    if token_arrays.get('fresh_argmax_tokens') != token_arrays.get('restored_argmax_tokens'):
        errors.append(f'{prefix} fresh-repeat/restored argmax token arrays differ')
complete = not errors
accepted = complete
summary = {
    'complete': complete,
    'accepted': accepted,
    'validation_errors': errors,
    'expected_depths': expected_depths,
    'seen_depths': seen,
    'state_restore_scope': value.get('state_restore_scope'),
    'state_api': value.get('state_api'),
    'continuation_contract': value.get('continuation_contract'),
    'n_gen': value.get('n_gen'),
    'seed': value.get('seed'),
    'abs_tolerance': value.get('abs_tolerance'),
    'rel_tolerance': value.get('rel_tolerance'),
    'records': [{k: r.get(k) for k in (
        'depth', 'complete', 'accepted', 'state_bytes', 'state_fnv1a64',
        'reprefill_state_fnv1a64', 'state_bitwise_mismatches', 'prefix_fnv1a64',
        'original_logits_fnv1a64', 'fresh_logits_fnv1a64', 'restored_logits_fnv1a64',
        'repeat_bitwise_mismatches', 'repeat_tolerance_violations',
        'repeat_nonfinite_mismatches', 'repeat_max_abs_diff', 'repeat_max_rel_diff',
        'bitwise_mismatches', 'tolerance_violations', 'nonfinite_mismatches',
        'max_abs_diff', 'max_rel_diff', 'generation_input_tokens',
        'original_argmax_tokens', 'fresh_argmax_tokens', 'restored_argmax_tokens')} for r in records if isinstance(r, dict)],
}
pathlib.Path(summary_path).write_text(json.dumps(summary, indent=2) + '\n')
columns = [
    'depth','accepted','state_bytes','state_bitwise_mismatches',
    'repeat_tolerance_violations','repeat_max_abs_diff',
    'tolerance_violations','max_abs_diff',
]
lines = ['\t'.join(columns)]
for r in summary['records']:
    lines.append('\t'.join(str(int(r[c])) if isinstance(r[c], bool) else str(r[c]) for c in columns))
pathlib.Path(tsv_path).write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
if errors:
    print('\n'.join(f'validation error: {error}' for error in errors), file=sys.stderr)
    raise SystemExit(4)
PY
summary_rc=$?
set -e
if [[ $rc -ne 0 ]]; then
    echo "Equivalence binary exited $rc; artifacts: $run_dir" >&2
    exit "$rc"
fi
if [[ $summary_rc -ne 0 ]]; then
    echo "Equivalence gate rejected; artifacts: $run_dir" >&2
    exit "$summary_rc"
fi
echo "Fresh/restored state equivalence accepted."
echo "Artifacts: $run_dir"