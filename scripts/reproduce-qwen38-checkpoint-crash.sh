#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SERVER="$ROOT/build/bin/llama-server"
CLIENT="$ROOT/scripts/reproduce-qwen38-long-checkpoint-rewind.py"
MODEL=/home/edwin/models/qwen38-27b-q4s8/unsloth-q8/Qwen3.8-27B-Q8_0.gguf
DRAFT=/home/edwin/models/qwen38-27b-q4s8/draft-q4/Qwen3.8-27B-MTP-Draft-Q4_0.gguf
OUT=
PORT=18170
CYCLES=10
N_PREDICT=8
N_PROBS=0
RETURN_TOKENS=0
SAFE_STATE_IO=0
SPID=

usage() {
    cat <<EOF
Usage: $0 --out DIR [--server PATH] [--model PATH] [--draft PATH]
          [--port N] [--cycles N] [--n-predict N] [--n-probs N]
          [--return-tokens] [--safe-state-io]

Runs the exact TP4 Qwen3.8 Q8_0 + external-MTP checkpoint stress case.
Success means every request completed, every expected restore occurred without
post-startup scheduler re-reservation, and the server remained healthy. A stock
reproduction is therefore expected to exit nonzero and preserve the crash log.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT=$2; shift 2 ;;
        --server) SERVER=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        --draft) DRAFT=$2; shift 2 ;;
        --port) PORT=$2; shift 2 ;;
        --cycles) CYCLES=$2; shift 2 ;;
        --n-predict) N_PREDICT=$2; shift 2 ;;
        --n-probs) N_PROBS=$2; shift 2 ;;
        --return-tokens) RETURN_TOKENS=1; shift ;;
        --safe-state-io) SAFE_STATE_IO=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$OUT" ]] || { usage >&2; exit 2; }
OUT=$(realpath -m -- "$OUT") || { echo "cannot resolve output directory: $OUT" >&2; exit 2; }
case "$OUT" in
    /|"$HOME"|"$ROOT")
        echo "refusing unsafe output directory: $OUT" >&2
        exit 2
        ;;
esac
[[ "$CYCLES" =~ ^[1-9][0-9]*$ ]] || { echo "cycles must be a positive integer" >&2; exit 2; }
[[ "$N_PREDICT" =~ ^[1-9][0-9]*$ ]] || { echo "n-predict must be a positive integer" >&2; exit 2; }
[[ "$N_PROBS" =~ ^[0-9]+$ ]] || { echo "n-probs must be a non-negative integer" >&2; exit 2; }
[[ -x "$SERVER" ]] || { echo "server is not executable: $SERVER" >&2; exit 2; }
[[ -f "$CLIENT" && -f "$MODEL" && -f "$DRAFT" ]] || { echo "missing client or model fixture" >&2; exit 2; }
SERVER=$(realpath -e -- "$SERVER")
CLIENT=$(realpath -e -- "$CLIENT")
MODEL=$(realpath -e -- "$MODEL")
DRAFT=$(realpath -e -- "$DRAFT")
if ps -eo comm= | grep -Eq '^(llama-server|llama-bench|llama-cli|rocprofv3)$'; then
    echo "refusing to run while a GPU workload is active" >&2
    exit 2
fi

cleanup() {
    if [[ -n "${SPID:-}" ]] && kill -0 "$SPID" 2>/dev/null; then
        kill -TERM -- -"$SPID" 2>/dev/null || kill -TERM "$SPID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

rm -rf "$OUT"
mkdir -p "$OUT"
START_TS=$(date --iso-8601=seconds)
{
    printf 'WORKTREE=%s\n' "$ROOT"
    printf 'SERVER=%s\nMODEL=%s\nDRAFT=%s\nCLIENT=%s\n' "$SERVER" "$MODEL" "$DRAFT" "$CLIENT"
    printf 'START_TS=%s\n' "$START_TS"
    git -C "$ROOT" rev-parse HEAD
    git -C "$ROOT" status --short
    "$SERVER" --version
} > "$OUT/provenance.txt" 2>&1 || { echo "failed to capture provenance" >&2; exit 2; }
sha256sum "$SERVER" "$MODEL" "$DRAFT" "$CLIENT" > "$OUT/input-sha256.txt" || {
    echo "failed to hash inputs" >&2
    exit 2
}
rocm-smi --showuse --showmemuse --showtemp > "$OUT/gpu-before.txt" 2>&1 || true
journalctl -k --since "$START_TS" --no-pager > "$OUT/kernel-before.txt" 2>&1 || true

ENV_ARGS=(
    -u GGML_CUDA_DISABLE_GRAPHS
    HSA_OVERRIDE_GFX_VERSION=10.3.0
    HSA_NO_SCRATCH_RECLAIM=1
    GGML_TP_SHARDED_OUTPUT=1
    GGML_CUDA_ALLREDUCE=nccl
)
if [[ $SAFE_STATE_IO -eq 1 ]]; then
    ENV_ARGS+=(GGML_HIP_SAFE_STATE_IO=1)
else
    ENV_ARGS=(-u GGML_HIP_SAFE_STATE_IO "${ENV_ARGS[@]}")
fi
SERVER_ARGS=(
    -m "$MODEL" -ngl all --split-mode tensor --fit off
    -dev ROCm0,ROCm1,ROCm2,ROCm3 -ts 1,1,1,1 -fa on
    -c 65536 -b 2048 -ub 256 -np 1 -ctk q8_0 -ctv q8_0
    --spec-draft-model "$DRAFT" --spec-type draft-mtp
    --spec-draft-n-max 3 --spec-draft-p-min 0 --spec-draft-ngl 999
    -devd ROCm0,ROCm1,ROCm2,ROCm3
    --ctx-checkpoints 128 --checkpoint-min-step 256 --cache-ram 24576
    --host 127.0.0.1 --port "$PORT" -lv 4
)
{
    printf 'env'; printf ' %q' "${ENV_ARGS[@]}"; printf ' %q' "$SERVER"; printf ' %q' "${SERVER_ARGS[@]}"; printf '\n'
} > "$OUT/server-command.txt"

setsid env "${ENV_ARGS[@]}" "$SERVER" "${SERVER_ARGS[@]}" > "$OUT/server.log" 2>&1 &
SPID=$!
echo "$SPID" > "$OUT/server.pid"

READY=0
for _ in $(seq 1 400); do
    if curl -fsS "http://127.0.0.1:$PORT/health" > "$OUT/health-ready.json" 2>/dev/null; then
        READY=1
        break
    fi
    kill -0 "$SPID" 2>/dev/null || break
    sleep 1
done
printf 'READY=%s\n' "$READY" | tee "$OUT/status.txt"

CLIENT_RC=99
COMPLETED=0
if [[ $READY -eq 1 ]]; then
    CLIENT_RC=0
    for cycle in $(seq -w 1 "$CYCLES"); do
        CLIENT_ARGS=(
            --url "http://127.0.0.1:$PORT/completion"
            --out "$OUT/client-$cycle"
            --base-repeats 1500 --tail-repeats 300
            --n-predict "$N_PREDICT" --n-probs "$N_PROBS" --timeout 1200
        )
        if [[ $RETURN_TOKENS -eq 1 ]]; then
            CLIENT_ARGS+=(--return-tokens)
        fi
        timeout --signal=TERM --kill-after=10s 1800s \
            python3 "$CLIENT" "${CLIENT_ARGS[@]}" \
                > "$OUT/client-$cycle.stdout" 2> "$OUT/client-$cycle.stderr"
        rc=$?
        if [[ $rc -ne 0 ]]; then
            CLIENT_RC=$rc
            break
        fi
        COMPLETED=$((COMPLETED + 1))
    done
fi
printf 'CLIENT_RC=%s\nCOMPLETED_CYCLES=%s\n' "$CLIENT_RC" "$COMPLETED" | tee -a "$OUT/status.txt"

ALIVE=0
if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" > "$OUT/health-after.json" 2> "$OUT/health-after.err"; then
    ALIVE=1
fi
printf 'SERVER_ALIVE_AFTER=%s\n' "$ALIVE" | tee -a "$OUT/status.txt"

if kill -0 "$SPID" 2>/dev/null; then
    kill -TERM -- -"$SPID" 2>/dev/null || kill -TERM "$SPID" 2>/dev/null || true
    for _ in $(seq 1 30); do
        kill -0 "$SPID" 2>/dev/null || break
        sleep 1
    done
fi
if kill -0 "$SPID" 2>/dev/null; then
    kill -KILL -- -"$SPID" 2>/dev/null || kill -KILL "$SPID" 2>/dev/null || true
fi
wait "$SPID" 2>/dev/null
SERVER_RC=$?
printf 'SERVER_RC=%s\n' "$SERVER_RC" | tee -a "$OUT/status.txt"

sleep 3
ps -eo pid=,comm=,args= | awk '$2 ~ /^(llama-server|llama-bench|llama-cli|rocprofv3)$/' > "$OUT/processes-after.txt"
rocm-smi --showuse --showmemuse --showtemp > "$OUT/gpu-after.txt" 2>&1 || true
journalctl -k --since "$START_TS" --no-pager > "$OUT/kernel-after.txt" 2>&1 || true
grep -nE '(new prompt|found better prompt|restored context checkpoint|checkpoint|seq_rm|ROCm error|HIP error|illegal|failed to remove|GGML_ABORT|fatal|shrunk recurrent|expanded recurrent|failed to restore|forcing full prompt|prompt eval time|stop processing|draft acceptance|sched_reserve: reserving)' \
    "$OUT/server.log" > "$OUT/relevant.log" || true
RESTORES=$(grep -c 'restored context checkpoint' "$OUT/server.log" || true)
SCHED_RESERVES=$(grep -c 'sched_reserve: reserving' "$OUT/server.log" || true)
ERROR_LINES=$(grep -Ec '(ROCm error|HIP error|illegal|failed to remove|GGML_ABORT|fatal error|assert|Aborted|Segmentation|failed to restore)' "$OUT/server.log" || true)
EXPECTED_RESTORES=$((5 * CYCLES - 1))
printf 'RESTORES=%s\nEXPECTED_RESTORES=%s\nSCHED_RESERVES=%s\nERROR_LINES=%s\n' \
    "$RESTORES" "$EXPECTED_RESTORES" "$SCHED_RESERVES" "$ERROR_LINES" | tee -a "$OUT/status.txt"

[[ $READY -eq 1 && $CLIENT_RC -eq 0 && $COMPLETED -eq $CYCLES && $ALIVE -eq 1 && $SERVER_RC -eq 0 &&
   $RESTORES -eq $EXPECTED_RESTORES && $SCHED_RESERVES -eq 3 && $ERROR_LINES -eq 0 ]]
