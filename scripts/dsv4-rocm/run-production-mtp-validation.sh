#!/usr/bin/env bash
# Matched production-config DSV4 main-only versus MTP validation.
set -Eeuo pipefail

ROOT_DIR=${DSV4_ROOT_DIR:-/home/edwin/llama.cpp-rdna2}
MODEL=${DSV4_MODEL:-/home/edwin/models/DeepSeek-V4-Flash-0731-GGUF/UD-IQ2_M/DeepSeek-V4-Flash-0731-UD-IQ2_M-00001-of-00003.gguf}
DRAFT_MODEL=${DSV4_DRAFT_MODEL:-/home/edwin/models/DeepSeek-V4-Flash-MTP-GGUF/DeepSeek-V4-Flash-MTP-Q4_0.gguf}
SERVER=${DSV4_SERVER:-$ROOT_DIR/build/bin/llama-server}
CORPUS=${DSV4_PROMPT_FILE:-$ROOT_DIR/scripts/dsv4-rocm/corpus/technical-proxy.txt}
OUTPUT_DIR=${LLAMA_JOB_DIR:-${DSV4_OUTPUT_DIR:-$HOME/llama-jobs/dsv4-production-mtp-$(date -u +%Y%m%dT%H%M%S.%NZ)}}
BASE_PORT=${DSV4_PORT:-18240}
OUTER_TIMEOUT=${DSV4_OUTER_TIMEOUT:-unknown}
N_PREDICT=${DSV4_N_PREDICT:-128}
STARTUP_RETRIES=${DSV4_STARTUP_RETRIES:-900}

fail() {
    echo "ERROR: $*" >&2
    exit 2
}

for pair in "DSV4_PORT:$BASE_PORT" "DSV4_N_PREDICT:$N_PREDICT" "DSV4_STARTUP_RETRIES:$STARTUP_RETRIES"; do
    name=${pair%%:*}
    value=${pair#*:}
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer"
done
git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 || fail "checkout not found: $ROOT_DIR"
[[ -z $(git -C "$ROOT_DIR" status --porcelain=v1) ]] || fail "source tree must be clean"
[[ -x "$SERVER" ]] || fail "server is not executable: $SERVER"
[[ -f "$MODEL" ]] || fail "main model not found: $MODEL"
[[ -f "$DRAFT_MODEL" ]] || fail "draft model not found: $DRAFT_MODEL"
[[ -f "$CORPUS" ]] || fail "prompt corpus not found: $CORPUS"
for tool in curl flock git python3 rocm-smi sha256sum; do
    command -v "$tool" >/dev/null || fail "$tool is required"
done

if [[ -n ${LLAMA_JOB_DIR:-} ]]; then
    mkdir -p "$OUTPUT_DIR"
else
    mkdir -- "$OUTPUT_DIR" || fail "output directory must not already exist: $OUTPUT_DIR"
fi
cp "${BASH_SOURCE[0]}" "$OUTPUT_DIR/production-runner.sh"
chmod +x "$OUTPUT_DIR/production-runner.sh"
sha256sum "$OUTPUT_DIR/production-runner.sh" > "$OUTPUT_DIR/production-runner.sha256"
printf 'outer_timeout_seconds=%s\n' "$OUTER_TIMEOUT" > "$OUTPUT_DIR/runner-settings.txt"
if [[ -z ${LLAMA_JOB_DIR:-} ]]; then
    exec 9>"$HOME/llama-jobs/gpu.lock"
    flock -n 9 || fail "GPU job lock is held"
fi

check_gpus_idle() {
    local phase=$1 output busy
    output=$(rocm-smi --showpids 2>&1) || fail "rocm-smi failed during $phase: $output"
    busy=$(printf '%s\n' "$output" | awk '$1 ~ /^[0-9]+$/ { print $0 }')
    [[ -z "$busy" ]] || fail "ROCm reports active processes during $phase: $busy"
}
check_gpus_idle "initial check"

export DSV4_HASH_MODE=full
export LD_LIBRARY_PATH="$(dirname "$SERVER")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_CUDA_ALLREDUCE=${GGML_CUDA_ALLREDUCE:-nccl}
export GGML_CUDA_P2P=${GGML_CUDA_P2P:-1}
export GGML_HIP_GRAPHS=${GGML_HIP_GRAPHS:-1}
export GGML_HIP_RDNA2_MMQ_J=${GGML_HIP_RDNA2_MMQ_J:-16}
export GGML_HIP_RDNA2_HC_MIXES=${GGML_HIP_RDNA2_HC_MIXES:-1}
export HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1}
export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-10.3.0}

"$ROOT_DIR/scripts/dsv4-rocm/manifest.sh" "$OUTPUT_DIR" "$SERVER" "$MODEL"
sha256sum "$DRAFT_MODEL" > "$OUTPUT_DIR/draft-model.sha256"
sha256sum "$CORPUS" > "$OUTPUT_DIR/corpus.sha256"
cp "$CORPUS" "$OUTPUT_DIR/corpus.txt"
cat > "$OUTPUT_DIR/production-client.py" <<'PYCLIENT'
#!/usr/bin/env python3
"""Issue one fixed PP-then-TG completion request and preserve server metrics."""
from __future__ import annotations
import json
import math
import pathlib
import sys
import urllib.request

if len(sys.argv) != 4:
    raise SystemExit(f"usage: {sys.argv[0]} BASE_URL PROMPT_FILE N_PREDICT")
base_url, prompt_path, n_predict_text = sys.argv[1:]
n_predict = int(n_predict_text)
if n_predict <= 0:
    raise SystemExit("N_PREDICT must be positive")
prompt = pathlib.Path(prompt_path).read_text(encoding="utf-8")
payload = {
    "model": "dsv4-production-validation",
    "prompt": prompt,
    "n_predict": n_predict,
    "seed": 123,
    "temperature": 0,
    "ignore_eos": True,
    "cache_prompt": False,
    "return_tokens": True,
}
request = urllib.request.Request(
    f"{base_url}/completion",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    result = json.load(response)
if not isinstance(result, dict):
    raise SystemExit("completion response is not an object")
timings = result.get("timings")
if not isinstance(timings, dict):
    raise SystemExit("completion response has no timings")
for name in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second"):
    value = timings.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise SystemExit(f"invalid timing {name}={value!r}")
for name in ("prompt_n", "predicted_n"):
    value = timings[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise SystemExit(f"timing count {name} is not an integer: {value!r}")
if timings["predicted_n"] != n_predict:
    raise SystemExit(f"expected {n_predict} generated tokens, got {timings['predicted_n']!r}")
if not isinstance(result.get("content"), str) or not result["content"]:
    raise SystemExit("completion returned no content")
tokens = result.get("tokens")
if not isinstance(tokens, list) or len(tokens) != timings["predicted_n"]:
    raise SystemExit("completion token list length does not match predicted_n")
if any(not isinstance(token, int) or isinstance(token, bool) for token in tokens):
    raise SystemExit("completion token list contains a non-integer token ID")
with urllib.request.urlopen(f"{base_url}/metrics", timeout=10) as response:
    metrics = response.read().decode("utf-8", errors="replace")
print(json.dumps({"request": payload, "response": result, "metrics": metrics}, separators=(",", ":")))
PYCLIENT
chmod +x "$OUTPUT_DIR/production-client.py"
sha256sum "$OUTPUT_DIR/production-client.py" > "$OUTPUT_DIR/production-client.sha256"

server_pid=""
cleanup_server() {
    if [[ -n "$server_pid" ]]; then
        kill -TERM "$server_pid" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=""
    fi
}
trap cleanup_server EXIT INT TERM HUP

common_args=(
    "$SERVER"
    --model "$MODEL"
    --alias dsv4-production-validation
    --n-gpu-layers all
    --split-mode tensor
    --tensor-split 1,1,1,1
    --flash-attn on
    --ctx-size 262144
    --cache-type-k f16
    --cache-type-v f16
    --kv-unified
    --batch-size 512
    --ubatch-size 256
    --parallel 1
    --host 127.0.0.1
    --temp 0
    --ctx-checkpoints 0
    --cache-ram 32768
    --metrics
)
draft_args=(
    --spec-draft-model "$DRAFT_MODEL"
    --spec-type draft-mtp
    --spec-draft-n-max 3
    --spec-draft-p-min 0
    --spec-draft-p-split 0.10
    --spec-draft-ngl all
    --spec-draft-device ROCm0,ROCm1
    --spec-draft-type-k f16
    --spec-draft-type-v f16
    --spec-draft-backend-sampling
)

run_arm() {
    local arm=$1 port=$2
    shift 2
    local arm_dir="$OUTPUT_DIR/$arm" ready=0
    local -a command=("${common_args[@]}" --port "$port" "$@")
    mkdir "$arm_dir"
    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\n'
        printf 'export LD_LIBRARY_PATH=%q\n' "$LD_LIBRARY_PATH"
        for name in GGML_CUDA_ALLREDUCE GGML_CUDA_P2P GGML_HIP_GRAPHS GGML_HIP_RDNA2_MMQ_J GGML_HIP_RDNA2_HC_MIXES HSA_NO_SCRATCH_RECLAIM HSA_OVERRIDE_GFX_VERSION; do
            printf 'export %s=%q\n' "$name" "${!name}"
        done
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "$arm_dir/server-command.sh"
    chmod +x "$arm_dir/server-command.sh"
    printf 'timeout --signal=TERM --kill-after=10s 300s python3 %q %q %q %q\n' "$OUTPUT_DIR/production-client.py" "http://127.0.0.1:$port" "$OUTPUT_DIR/corpus.txt" "$N_PREDICT" > "$arm_dir/client-command.sh"
    chmod +x "$arm_dir/client-command.sh"

    if curl --silent --fail --max-time 2 "http://127.0.0.1:$port/props" >/dev/null 2>&1; then
        fail "port $port is already serving a model"
    fi
    check_gpus_idle "before $arm launch"
    "${command[@]}" > "$arm_dir/server.log" 2>&1 &
    server_pid=$!
    for _ in $(seq 1 "$STARTUP_RETRIES"); do
        if curl --silent --fail --max-time 2 "http://127.0.0.1:$port/props" > "$arm_dir/props.json" 2>/dev/null &&
           kill -0 "$server_pid" 2>/dev/null &&
           python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["model_alias"] == "dsv4-production-validation"; assert p["model_path"] == sys.argv[2]; assert p["total_slots"] == 1; assert p["endpoint_metrics"] is True' "$arm_dir/props.json" "$MODEL" 2>/dev/null; then
            ready=1
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -100 "$arm_dir/server.log" >&2 || true
            fail "$arm server exited before readiness"
        fi
        sleep 1
    done
    [[ "$ready" == 1 ]] || fail "$arm server did not become ready"
    timeout --signal=TERM --kill-after=10s 300s \
        python3 "$OUTPUT_DIR/production-client.py" "http://127.0.0.1:$port" "$OUTPUT_DIR/corpus.txt" "$N_PREDICT" \
        > "$arm_dir/response.json"
    cleanup_server
    check_gpus_idle "after $arm"
}

run_arm main-only "$BASE_PORT"
run_arm mtp "$((BASE_PORT + 1))" "${draft_args[@]}"

python3 - "$OUTPUT_DIR" <<'PY'
import json
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
def load(arm):
    value = json.loads((root / arm / "response.json").read_text())
    if not isinstance(value, dict) or not isinstance(value.get("response"), dict):
        raise SystemExit(f"{arm}: malformed preserved response")
    return value

base = load("main-only")
mtp = load("mtp")
base_response = base["response"]
mtp_response = mtp["response"]
for field in ("content", "tokens"):
    if base_response.get(field) != mtp_response.get(field):
        raise SystemExit(f"main-only/MTP {field} differ")
if base.get("request") != mtp.get("request"):
    raise SystemExit("main-only/MTP requests differ")
if base_response["timings"]["prompt_n"] != mtp_response["timings"]["prompt_n"]:
    raise SystemExit("main-only/MTP prompt token counts differ")
if base_response["timings"]["predicted_n"] != mtp_response["timings"]["predicted_n"]:
    raise SystemExit("main-only/MTP generated token counts differ")

spec_fields = ("draft_attempts", "draft_empty", "draft_n", "draft_n_accepted", "draft_verification_steps")
base_timings = base_response["timings"]
mtp_timings = mtp_response["timings"]
if any(field in base_timings for field in spec_fields):
    raise SystemExit("main-only arm unexpectedly reports speculative counters")
if any(field not in mtp_timings for field in spec_fields):
    raise SystemExit("MTP arm is missing speculative counters")
for field in spec_fields:
    value = mtp_timings[field]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SystemExit(f"invalid MTP speculative counter {field}={value!r}")
if mtp_timings["draft_attempts"] <= 0 or mtp_timings["draft_n"] <= 0:
    raise SystemExit("MTP arm did not draft any tokens")
if mtp_timings["draft_empty"] > mtp_timings["draft_attempts"]:
    raise SystemExit("MTP draft_empty exceeds draft_attempts")
if mtp_timings["draft_n_accepted"] > mtp_timings["draft_n"]:
    raise SystemExit("MTP accepted count exceeds drafted tokens")
if mtp_timings["draft_verification_steps"] <= 0:
    raise SystemExit("MTP arm reports no verification steps")

def timing(response, key):
    value = response["timings"][key]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise SystemExit(f"invalid {key}: {value!r}")
    return value

summary = {
    "measurement_note": "single main-only-then-MTP observation; correctness and production-routing evidence, not a stable speed claim",
    "responses_match": True,
    "prompt_tokens": int(timing(base_response, "prompt_n")),
    "generated_tokens": int(timing(base_response, "predicted_n")),
    "main_only": {
        "pp_tps": timing(base_response, "prompt_per_second"),
        "pp_ms": timing(base_response, "prompt_ms"),
        "tg_tps": timing(base_response, "predicted_per_second"),
        "tg_ms": timing(base_response, "predicted_ms"),
    },
    "mtp": {
        "pp_tps": timing(mtp_response, "prompt_per_second"),
        "pp_ms": timing(mtp_response, "prompt_ms"),
        "tg_tps": timing(mtp_response, "predicted_per_second"),
        "tg_ms": timing(mtp_response, "predicted_ms"),
        "draft_attempts": mtp_timings["draft_attempts"],
        "draft_empty": mtp_timings["draft_empty"],
        "draft_n": mtp_timings["draft_n"],
        "draft_n_accepted": mtp_timings["draft_n_accepted"],
        "draft_verification_steps": mtp_timings["draft_verification_steps"],
        "draft_acceptance_pct": 100 * mtp_timings["draft_n_accepted"] / mtp_timings["draft_n"],
    },
}
for stage, metric in (("pp", "pp_tps"), ("tg", "tg_tps")):
    before = summary["main_only"][metric]
    after = summary["mtp"][metric]
    summary[f"{stage}_delta_pct"] = 100 * (after / before - 1)
(root / "comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

printf 'complete=1\nfinished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" > "$OUTPUT_DIR/production-status.txt"
echo "Artifacts: $OUTPUT_DIR"