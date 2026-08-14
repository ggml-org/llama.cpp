#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT="${LAGUNA_LLAMA_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
MODEL="${LAGUNA_MODEL:-/home/edwin/models/laguna/UD-IQ4_XS/Laguna-S-2.1-UD-IQ4_XS-00001-of-00003.gguf}"
PORT="${LAGUNA_VERIFY_PORT:-19090}"
TIMEOUT_SECONDS="${LAGUNA_VERIFY_TIMEOUT:-300}"
ARTIFACT_DIR="${LAGUNA_ARTIFACT_DIR:-/home/edwin/models/laguna/UD-IQ4_XS/diagnostics}"

SERVER="$ROOT/build/bin/llama-server"
TEST_META="$ROOT/build/bin/test-meta-split"
TEST_TENSOR="$ROOT/build/bin/test-tensor-split"
mkdir -p "$ARTIFACT_DIR"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
log="$ARTIFACT_DIR/laguna-verify-$stamp.log"
response="$ARTIFACT_DIR/laguna-verify-$stamp.json"
test_log="$ARTIFACT_DIR/laguna-verify-tests-$stamp.log"

for path in "$SERVER" "$TEST_META" "$TEST_TENSOR" "$MODEL"; do
    [[ -e "$path" ]] || { echo "missing required artifact: $path" >&2; exit 2; }
done
current_commit=$(git -C "$ROOT" rev-parse --short=9 HEAD)
[[ -z "$(git -C "$ROOT" status --porcelain)" ]] || { echo "source worktree is dirty" >&2; exit 2; }
"$SERVER" --version 2>&1 | grep -F "($current_commit)"

ulimit -s 8192
[[ "$(ulimit -s)" == 8192 ]] || { echo "could not enforce 8 MiB stack" >&2; exit 2; }

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_NO_SCRATCH_RECLAIM=1
export GGML_CUDA_DISABLE_GRAPHS=1
export GGML_CUDA_ALLREDUCE=nccl
export LD_LIBRARY_PATH=/opt/rocm/core-7.14/lib

(
    cd "$ROOT"
    env LD_LIBRARY_PATH="$ROOT/build/bin:$LD_LIBRARY_PATH" \
        ctest --test-dir build --verbose --output-on-failure -R 'test-(meta-split|tensor-split)'
) >"$test_log" 2>&1

env "$SERVER" \
    -m "$MODEL" \
    --jinja --parallel 2 --flash-attn on -lv 4 --port "$PORT" --host 127.0.0.1 \
    --threads 44 --batch-size 512 --ubatch-size 1024 --ctx-size 250000 \
    -ctk q8_0 -ctv q8_0 --fit-target 0 --split-mode tensor --main-gpu 1 \
    --tensor-split 1,1 -cram 32768 --no-warmup >"$log" 2>&1 &
pid=$!

cleanup() {
    if kill -0 "$pid" 2>/dev/null; then
        kill -INT "$pid" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 "$((TIMEOUT_SECONDS / 2))"); do
    if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" || rc=$?
        echo "server exited before ready: ${rc:-0}" >&2
        tail -n 120 "$log" >&2
        exit 1
    fi
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 2
done
[[ "$ready" == 1 ]] || { echo "health timeout" >&2; tail -n 120 "$log" >&2; exit 1; }

curl -fsS --max-time 240 -H 'Content-Type: application/json' \
    -d '{"prompt":"Complete this sentence concisely: The capital of France is","n_predict":8,"temperature":0,"cache_prompt":false}' \
    "http://127.0.0.1:$PORT/completion" >"$response"

python3 - "$response" <<'PY'
import json, sys
response = json.load(open(sys.argv[1]))
assert response.get("tokens_predicted") == 8, response
assert response.get("content"), response
print("content:", repr(response["content"]))
print("predicted_per_second:", response.get("timings", {}).get("predicted_per_second"))
PY

grep -F 'deep meta graph passed (2048 nodes)' "$test_log" >/dev/null
grep -F '100% tests passed, 0 tests failed out of 2' "$test_log" >/dev/null
grep -F 'graph nodes  = 4807' "$log" >/dev/null
grep -F 'model loaded' "$log" >/dev/null
grep -F 'listening on http://127.0.0.1:' "$log" >/dev/null
kill -0 "$pid"

echo "test_log=$test_log"
echo "server_log=$log"
echo "response=$response"
echo "LAGUNA_TENSOR_SPLIT_VERIFY_OK"
