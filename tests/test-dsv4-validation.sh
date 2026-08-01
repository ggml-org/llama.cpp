#!/usr/bin/env bash
# Reproducible, model-dependent DSv4 validation.  No model is downloaded by this script.
set -Eeuo pipefail

usage() {
    cat <<'USAGE'
Usage: DSV4_MODEL=/path/to/model.gguf tests/test-dsv4-validation.sh

The harness starts llama-server twice, once in layer (reference) mode and once
in tensor-split mode, then checks a prompt, a continuation, and prompt/KV
reuse.  The server binary defaults to build/bin/llama-server.

Environment overrides:
  DSV4_SERVER             llama-server path
  DSV4_TENSOR_SPLIT       tensor proportions (default: 1,1,1,1)
  DSV4_REFERENCE_SPLIT    reference split mode (default: layer)
  DSV4_FLASH_ATTN         on, off, or auto (default: auto)
  DSV4_PARALLEL            server parallel slots (default: 1)
  DSV4_DRAFT_MODEL         optional speculative draft GGUF
  DSV4_SPEC_TYPE           speculative type (default: draft-mtp with draft model)
  DSV4_DRAFT_N_MAX          optional speculative draft token count
  DSV4_PORT               first localhost port (default: 18080)
  DSV4_CTX_SIZE           context size (default: 4096)
  DSV4_N_PREDICT          generated tokens per request (default: 8)
  DSV4_CACHE_REUSE        minimum cache-reuse chunk (default: 16)
USAGE
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

: "${DSV4_MODEL:?DSV4_MODEL must point to a real DSv4 GGUF}"
[[ -f "$DSV4_MODEL" ]] || { echo "error: DSV4_MODEL is not a file: $DSV4_MODEL" >&2; exit 2; }

TENSOR_SPLIT=${DSV4_TENSOR_SPLIT:-1,1,1,1}
REFERENCE_SPLIT=${DSV4_REFERENCE_SPLIT:-layer}
FLASH_ATTN=${DSV4_FLASH_ATTN:-auto}
PARALLEL=${DSV4_PARALLEL:-1}
BASE_PORT=${DSV4_PORT:-18080}
CTX_SIZE=${DSV4_CTX_SIZE:-4096}
N_PREDICT=${DSV4_N_PREDICT:-8}
CACHE_REUSE=${DSV4_CACHE_REUSE:-16}
DRAFT_MODEL=${DSV4_DRAFT_MODEL:-}
SPEC_TYPE=${DSV4_SPEC_TYPE:-}
DRAFT_N_MAX=${DSV4_DRAFT_N_MAX:-}
case "$FLASH_ATTN" in
    on|off|auto) ;;
    *) echo "error: DSV4_FLASH_ATTN must be on, off, or auto (got '$FLASH_ATTN')" >&2; exit 2 ;;
esac
if [[ -n "$DRAFT_MODEL" ]]; then
    [[ -f "$DRAFT_MODEL" ]] || { echo "error: DSV4 draft model is not a file: $DRAFT_MODEL" >&2; exit 2; }
    SPEC_TYPE=${SPEC_TYPE:-draft-mtp}
elif [[ -n "$SPEC_TYPE" || -n "$DRAFT_N_MAX" ]]; then
    echo "error: DSV4_DRAFT_MODEL is required when DSV4_SPEC_TYPE or DSV4_DRAFT_N_MAX is set" >&2
    exit 2
fi

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SERVER=${DSV4_SERVER:-$ROOT_DIR/build/bin/llama-server}
[[ -x "$SERVER" ]] || { echo "error: llama-server is not executable: $SERVER" >&2; exit 2; }
command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 2; }
command -v python3 >/dev/null || { echo "error: python3 is required" >&2; exit 2; }

TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/dsv4-validation.XXXXXX")
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$TMP_ROOT"
}
trap cleanup EXIT INT TERM

PROMPT=${DSV4_PROMPT:-'A deterministic DSv4 validation prompt: explain why caching a shared prefix helps inference.'}
CONTINUATION=${DSV4_CONTINUATION:-' Continue with one short sentence about the same topic.'}

request() {
    local port=$1 body=$2 output=$3 status
    status=$(curl --silent --show-error --output "$output" --write-out '%{http_code}' \
        --max-time "${DSV4_REQUEST_TIMEOUT:-300}" \
        -H 'Content-Type: application/json' \
        --data "$body" "http://127.0.0.1:${port}/completion") || {
        echo "error: request failed (server log: $TMP_ROOT/server.log)" >&2
        return 1
    }
    if [[ "$status" != 200 ]]; then
        echo "error: /completion returned HTTP $status:" >&2
        cat "$output" >&2
        return 1
    fi
}

run_mode() {
    local mode=$1 label=$2 port=$3 out_dir="$TMP_ROOT/$2"
    mkdir -p "$out_dir"
    echo "[$label] split-mode=$mode tensor-split=$TENSOR_SPLIT flash-attn=$FLASH_ATTN parallel=$PARALLEL"

    local -a draft_args=()
    if [[ -n "$DRAFT_MODEL" ]]; then
        draft_args+=(--spec-draft-model "$DRAFT_MODEL" --spec-type "$SPEC_TYPE")
        if [[ -n "$DRAFT_N_MAX" ]]; then
            draft_args+=(--spec-draft-n-max "$DRAFT_N_MAX")
        fi
    fi

    "$SERVER" \
        --model "$DSV4_MODEL" \
        --alias dsv4-validation \
        --host 127.0.0.1 \
        --port "$port" \
        --ctx-size "$CTX_SIZE" \
        --parallel "$PARALLEL" \
        --seed 123 \
        --temp 0 \
        --flash-attn "$FLASH_ATTN" \
        --split-mode "$mode" \
        --tensor-split "$TENSOR_SPLIT" \
        --cache-prompt \
        --cache-reuse "$CACHE_REUSE" \
        "${draft_args[@]}" \
        >"$TMP_ROOT/server.log" 2>&1 &
    SERVER_PID=$!

    local ready=0
    for _ in $(seq 1 "${DSV4_STARTUP_RETRIES:-300}"); do
        if curl --silent --fail --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "error: llama-server exited while starting ($label):" >&2
            cat "$TMP_ROOT/server.log" >&2
            return 1
        fi
        sleep 1
    done
    if [[ "$ready" != 1 ]]; then
        echo "error: llama-server did not become ready ($label):" >&2
        cat "$TMP_ROOT/server.log" >&2
        return 1
    fi

    local first_prompt second_prompt
    first_prompt=$(DSV4_N_PREDICT="$N_PREDICT" python3 - "$PROMPT" <<'PY'
import json, sys
print(json.dumps({"model": "dsv4-validation", "prompt": sys.argv[1], "n_predict": int(__import__("os").environ["DSV4_N_PREDICT"]), "seed": 123, "temperature": 0, "cache_prompt": True, "id_slot": 0, "return_tokens": True}))
PY
)
    second_prompt=$(DSV4_N_PREDICT="$N_PREDICT" python3 - "$PROMPT$CONTINUATION" <<'PY'
import json, sys
print(json.dumps({"model": "dsv4-validation", "prompt": sys.argv[1], "n_predict": int(__import__("os").environ["DSV4_N_PREDICT"]), "seed": 123, "temperature": 0, "cache_prompt": True, "id_slot": 0, "return_tokens": True}))
PY
)
    request "$port" "$first_prompt" "$out_dir/first.json"
    request "$port" "$second_prompt" "$out_dir/continuation.json"
    request "$port" "$second_prompt" "$out_dir/replay.json"

    python3 - "$label" "$out_dir" <<'PY'
import json
import pathlib
import sys

label, directory = sys.argv[1:]
root = pathlib.Path(directory)
def load(name):
    with (root / name).open() as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise SystemExit(f"{label}: {name} is not a JSON object")
    return value

first = load("first.json")
continuation = load("continuation.json")
replay = load("replay.json")
for name, response in (("first", first), ("continuation", continuation), ("replay", replay)):
    if not response.get("content"):
        raise SystemExit(f"{label}: {name} returned no generated content")
    timings = response.get("timings")
    if not isinstance(timings, dict):
        raise SystemExit(f"{label}: {name} has no timings")
    if not isinstance(timings.get("predicted_n"), (int, float)) or timings["predicted_n"] <= 0:
        raise SystemExit(f"{label}: {name} predicted no tokens")

for name, response in (("continuation", continuation), ("replay", replay)):
    cache_n = response["timings"].get("cache_n", 0)
    if not isinstance(cache_n, (int, float)) or cache_n <= 0:
        raise SystemExit(f"{label}: {name} did not reuse KV (cache_n={cache_n!r})")

if continuation["content"] != replay["content"]:
    raise SystemExit(f"{label}: replay output differs from continuation output")
print(f"[{label}] prompt, continuation, replay, and KV reuse passed")
PY

    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}

# Keep the reference and tensor runs separate: each gets a fresh server and KV cache.
run_mode "$REFERENCE_SPLIT" reference "$BASE_PORT"
run_mode tensor tensor "$((BASE_PORT + 1))"

python3 - "$TMP_ROOT/reference" "$TMP_ROOT/tensor" <<'PY'
import json
import pathlib
import sys

reference_root, tensor_root = map(pathlib.Path, sys.argv[1:])
for name in ("first.json", "continuation.json", "replay.json"):
    with (reference_root / name).open() as f:
        reference = json.load(f)
    with (tensor_root / name).open() as f:
        tensor = json.load(f)
    if reference.get("content") != tensor.get("content"):
        raise SystemExit(f"reference and tensor-split {name} outputs differ")
print("[compare] reference and tensor-split deterministic outputs match")
PY

echo "DSv4 validation passed (model: $DSV4_MODEL)"
