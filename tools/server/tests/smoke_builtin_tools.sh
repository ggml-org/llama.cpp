#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

if [[ "${1:-}" == "--help" ]]; then
	cat <<'EOF'
Quick local smoke test for built-in server tools.

Environment overrides:
  PORT                 Server port (default: 8083)
  HOST                 Server host (default: 127.0.0.1)
  LLAMA_SERVER_BIN_PATH Override llama-server binary path
  MODEL_PATH           Use --model instead of --hf-repo/--hf-file
  HF_REPO              Hugging Face repo (default: ggml-org/models)
  HF_FILE              Hugging Face file (default: tinyllamas/stories260K.gguf)
  LOG_PATH             Server log path (default: /tmp/llama-server-builtin-tools-<port>.log)
  STARTUP_TIMEOUT_SEC  Wait time for /health (default: 180)
  KEEP_SERVER          Leave server running after test when set to 1
EOF
	exit 0
fi

PORT="${PORT:-8083}"
HOST="${HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-180}"
KEEP_SERVER="${KEEP_SERVER:-0}"
LOG_PATH="${LOG_PATH:-/tmp/llama-server-builtin-tools-${PORT}.log}"
SERVER_BIN="${LLAMA_SERVER_BIN_PATH:-$REPO_ROOT/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-}"
HF_REPO="${HF_REPO:-ggml-org/models}"
HF_FILE="${HF_FILE:-tinyllamas/stories260K.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-tinyllama-builtin-tools-smoke}"
TOOLS="${TOOLS:-question,todowrite,artifact_create,artifact_edit}"
TEMPERATURE="${TEMPERATURE:-0.8}"
SEED="${SEED:-42}"
CTX_SIZE="${CTX_SIZE:-2048}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PARALLEL="${PARALLEL:-1}"
CURL_BIN="${CURL_BIN:-curl}"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
	PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
else
	PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

pass() {
	printf '[PASS] %s\n' "$1"
}

fail() {
	printf '[FAIL] %s\n' "$1" >&2
	printf 'Server log: %s\n' "$LOG_PATH" >&2
	exit 1
}

cleanup() {
	local exit_code=$?

	if [[ "${KEEP_SERVER}" != "1" && -n "${SERVER_PID:-}" ]]; then
		if kill -0 "$SERVER_PID" 2>/dev/null; then
			kill "$SERVER_PID" 2>/dev/null || true
			wait "$SERVER_PID" 2>/dev/null || true
		fi
	fi

	if [[ $exit_code -eq 0 ]]; then
		if [[ "${KEEP_SERVER}" == "1" ]]; then
			printf 'Server left running at %s (pid %s)\n' "$BASE_URL" "$SERVER_PID"
		else
			printf 'Stopped validation server (pid %s)\n' "$SERVER_PID"
		fi
		printf 'Log kept at %s\n' "$LOG_PATH"
	fi
}

trap cleanup EXIT

if [[ ! -x "$SERVER_BIN" ]]; then
	fail "llama-server binary not found at $SERVER_BIN"
fi

if ! command -v "$CURL_BIN" >/dev/null 2>&1; then
	fail "curl not found: $CURL_BIN"
fi

if [[ "$PYTHON_BIN" == */* ]]; then
	[[ -x "$PYTHON_BIN" ]] || fail "Python not found: $PYTHON_BIN"
else
	command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python not found: $PYTHON_BIN"
fi

json_get() {
	"$CURL_BIN" -sS "$BASE_URL$1"
}

json_post() {
	local path="$1"
	local payload="$2"
	"$CURL_BIN" -sS -X POST "$BASE_URL$path" -H "Content-Type: application/json" -d "$payload"
}

SERVER_ARGS=(
	--host "$HOST"
	--port "$PORT"
	--temp "$TEMPERATURE"
	--seed "$SEED"
	--alias "$MODEL_ALIAS"
	--tools "$TOOLS"
	--ctx-size "$CTX_SIZE"
	--parallel "$PARALLEL"
	--batch-size "$BATCH_SIZE"
	--no-slots
	--no-jinja
)

if [[ -n "$MODEL_PATH" ]]; then
	SERVER_ARGS+=(--model "$MODEL_PATH")
else
	SERVER_ARGS+=(--hf-repo "$HF_REPO" --hf-file "$HF_FILE")
fi

printf 'Starting llama-server smoke target on %s\n' "$BASE_URL"
nohup "$SERVER_BIN" "${SERVER_ARGS[@]}" > "$LOG_PATH" 2>&1 &
SERVER_PID=$!
printf 'Server PID: %s\n' "$SERVER_PID"

for ((i = 0; i < STARTUP_TIMEOUT_SEC; i++)); do
	if json_get "/health" >/dev/null 2>&1; then
		pass "server became healthy"
		break
	fi

	if ! kill -0 "$SERVER_PID" 2>/dev/null; then
		fail "llama-server exited before becoming healthy"
	fi

	sleep 1

	if [[ $i -eq $((STARTUP_TIMEOUT_SEC - 1)) ]]; then
		fail "timed out waiting for /health"
	fi
done

tools_response=$(json_get "/tools")
JSON_INPUT="$tools_response" "$PYTHON_BIN" - <<'PY'
import json
import os

data = json.loads(os.environ["JSON_INPUT"])
names = {item["tool"] for item in data}
expected = {"question", "todowrite", "artifact_create", "artifact_edit"}
missing = sorted(expected - names)
if missing:
    raise SystemExit(f"missing tools: {', '.join(missing)}")
PY
pass "/tools advertised question/todo/artifact built-ins"

TODOWRITE_PAYLOAD=$(cat <<'JSON'
{"tool":"todowrite","params":{"todos":[{"content":"Inspect endpoints","status":"completed"},{"content":"Validate artifact flow","status":"in_progress"}]}}
JSON
)

todowrite_response=$(json_post "/tools" "$TODOWRITE_PAYLOAD")
JSON_INPUT="$todowrite_response" "$PYTHON_BIN" - <<'PY'
import json
import os

data = json.loads(os.environ["JSON_INPUT"])
assert data.get("status") == "completed", data
text = data.get("plain_text_response", "")
assert '"content": "Inspect endpoints"' in text, text
assert '"status": "in_progress"' in text, text
PY
pass "todowrite returned JSON todo snapshot"

ARTIFACT_CREATE_PAYLOAD=$(cat <<'JSON'
{"tool":"artifact_create","params":{"name":"notes.md","mime_type":"text/markdown","content":"# Notes\n\nLocal smoke test"},"context":{"conversation_id":"conv-smoke-1","tool_call_id":"call-smoke-1"}}
JSON
)

artifact_create_response=$(json_post "/tools" "$ARTIFACT_CREATE_PAYLOAD")
ARTIFACT_ID=$(JSON_INPUT="$artifact_create_response" "$PYTHON_BIN" - <<'PY'
import json
import os

data = json.loads(os.environ["JSON_INPUT"])
assert data.get("status") == "completed", data
artifact_id = data.get("artifact_id")
assert artifact_id, data
attachments = data.get("attachments") or []
assert len(attachments) == 1, attachments
attachment = attachments[0]
assert attachment.get("presentation") == "artifact", attachment
assert attachment.get("type") == "TEXT", attachment
assert attachment.get("artifactId") == artifact_id, attachment
assert "Local smoke test" in attachment.get("content", ""), attachment
print(artifact_id)
PY
)
pass "artifact_create returned artifact attachment"
printf 'Artifact ID: %s\n' "$ARTIFACT_ID"

ARTIFACT_EDIT_PAYLOAD=$(cat <<JSON
{"tool":"artifact_edit","params":{"artifact_id":"$ARTIFACT_ID","content":"# Notes\\n\\nEdited smoke test"},"context":{"conversation_id":"conv-smoke-1","tool_call_id":"call-smoke-2"}}
JSON
)

artifact_edit_response=$(json_post "/tools" "$ARTIFACT_EDIT_PAYLOAD")
JSON_INPUT="$artifact_edit_response" "$PYTHON_BIN" - <<'PY'
import json
import os

data = json.loads(os.environ["JSON_INPUT"])
assert data.get("status") == "completed", data
attachments = data.get("attachments") or []
assert len(attachments) == 1, attachments
attachment = attachments[0]
assert "Edited smoke test" in attachment.get("content", ""), attachment
PY
pass "artifact_edit updated the stored artifact"

QUESTION_PAYLOAD=$(cat <<'JSON'
{"tool":"question","params":{"questions":[{"question":"Pick one","header":"Choice","options":[{"label":"A","description":"Option A"},{"label":"B","description":"Option B"}]}]},"context":{"conversation_id":"conv-question-1","tool_call_id":"call-question-1"}}
JSON
)

question_response=$(json_post "/tools" "$QUESTION_PAYLOAD")
REQUEST_ID=$(JSON_INPUT="$question_response" "$PYTHON_BIN" - <<'PY'
import json
import os

data = json.loads(os.environ["JSON_INPUT"])
assert data.get("status") == "awaiting_user", data
assert data.get("kind") == "question", data
request_id = data.get("request_id")
assert request_id, data
print(request_id)
PY
)
pass "question entered awaiting_user state"
printf 'Question request ID: %s\n' "$REQUEST_ID"

pending_response=$(json_get "/tools/pending?conversation_id=conv-question-1")
JSON_INPUT="$pending_response" REQUEST_ID="$REQUEST_ID" "$PYTHON_BIN" - <<'PY'
import json
import os

request_id = os.environ["REQUEST_ID"]
data = json.loads(os.environ["JSON_INPUT"])
assert isinstance(data, list) and data, data
assert any(item.get("request_id") == request_id for item in data), data
PY
pass "/tools/pending returned the queued question"

QUESTION_REPLY_PAYLOAD=$(cat <<JSON
{"request_id":"$REQUEST_ID","conversation_id":"conv-question-1","answers":[["A"]]}
JSON
)

question_reply_response=$(json_post "/tools/reply" "$QUESTION_REPLY_PAYLOAD")
JSON_INPUT="$question_reply_response" "$PYTHON_BIN" - <<'PY'
import json
import os

data = json.loads(os.environ["JSON_INPUT"])
assert data.get("status") == "completed", data
text = data.get("plain_text_response", "")
assert "Pick one" in text, text
assert "A" in text, text
PY
pass "question reply completed successfully"

pending_after_response=$(json_get "/tools/pending?conversation_id=conv-question-1")
JSON_INPUT="$pending_after_response" "$PYTHON_BIN" - <<'PY'
import json
import os

data = json.loads(os.environ["JSON_INPUT"])
assert data == [], data
PY
pass "question queue cleared after reply"

printf 'All built-in tool smoke tests passed.\n'
