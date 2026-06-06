#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_SERVER_ENDPOINT="${LLAMA_SERVER_ENDPOINT:-http://localhost:11434/v1/chat/completions}"
LLAMA_SERVER_MODEL="${LLAMA_SERVER_MODEL:-gemma4:e4b}"
LLAMA_API_KEY="${LLAMA_API_KEY:-}"
AUDIO_QA_SYSTEM_PROMPT="${AUDIO_QA_SYSTEM_PROMPT:-あなたは音声内容に基づいて質問へ答えるアシスタントです。音声に含まれない情報は推測で補わず、不明な場合は不明と答えてください。回答は簡潔な日本語で返してください。}"
DEFAULT_AUDIO_QA_PROMPT="${DEFAULT_AUDIO_QA_PROMPT:-音声に含まれる話し言葉を質問または依頼として理解し、日本語で簡潔に答えてください。音声だけで答えられない場合は、不明と答えてください。}"
OUTPUT="${SCRIPT_DIR}/gemma4_audio_qa_input.wav"
DURATION="10"
DEVICE=":2"

usage() {
  cat <<'USAGE'
Usage:
  ./my_main/gemma4_audio_qa_llama.sh <audio-file> [instruction]
  ./my_main/gemma4_audio_qa_llama.sh --record [seconds]
  ./my_main/gemma4_audio_qa_llama.sh --record-with-text <question> [seconds]
  ./my_main/gemma4_audio_qa_llama.sh --list-devices

Environment variables:
  LLAMA_SERVER_ENDPOINT       llama-server chat endpoint.
                              Default: http://localhost:11434/v1/chat/completions
  LLAMA_SERVER_MODEL          Model alias used by the server. Default: gemma4:e4b
  LLAMA_API_KEY               Optional API key for llama-server
  AUDIO_QA_SYSTEM_PROMPT      System prompt for audio question answering
  DEFAULT_AUDIO_QA_PROMPT     Default user prompt when only audio is provided
  AUDIO_DEVICE                ffmpeg AVFoundation audio device. Default: :1
  AUDIO_OUTPUT                Output WAV for --record. Default: ./my_main/gemma4_audio_qa_input.wav

Examples:
  ./my_main/gemma4_audio_qa_llama.sh ./my_main/gemma4_test_ja.wav
  ./my_main/gemma4_audio_qa_llama.sh ./voice.wav "この音声の要点を2行でまとめてください。"
  ./my_main/gemma4_audio_qa_llama.sh --record 10
  AUDIO_DEVICE=':1' ./my_main/gemma4_audio_qa_llama.sh --record-with-text "この音声の要点を教えてください。" 8

Expected OpenAI-compatible request shape:
  POST /v1/chat/completions
  messages[1].content = [
    {"type":"text","text":"<question>"},
    {"type":"input_audio","input_audio":{"data":"<base64>","format":"wav|mp3"}}
  ]
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required." >&2
  exit 1
fi

DEVICE="${AUDIO_DEVICE:-$DEVICE}"
OUTPUT="${AUDIO_OUTPUT:-$OUTPUT}"

case "${1:-}" in
  --list-devices)
    if ! command -v ffmpeg >/dev/null 2>&1; then
      echo "ffmpeg is not installed. Install it with: brew install ffmpeg" >&2
      exit 1
    fi
    ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 || true
    exit 0
    ;;
  --record)
    if ! command -v ffmpeg >/dev/null 2>&1; then
      cat >&2 <<'MSG'
ffmpeg is not installed, so direct microphone recording is unavailable.

Use one of these options:
  1. Install ffmpeg: brew install ffmpeg
  2. Record with Voice Memos or QuickTime, export the file, then run:
     ./my_main/gemma4_audio_qa_llama.sh /path/to/your_recording.wav
MSG
      exit 1
    fi
    if [[ $# -ge 2 ]]; then
      DURATION="$2"
    fi
    QUESTION="$DEFAULT_AUDIO_QA_PROMPT"
    mkdir -p "$(dirname "$OUTPUT")"
    echo "Recording $DURATION seconds from AVFoundation device '$DEVICE'..."
    ffmpeg -hide_banner -y \
      -f avfoundation -i "$DEVICE" \
      -t "$DURATION" \
      -ac 1 -ar 16000 -c:a pcm_s16le \
      "$OUTPUT"
    echo "Created: $OUTPUT"
    AUDIO_FILE="$OUTPUT"
    ;;
  --record-with-text)
    if [[ $# -lt 2 ]]; then
      usage >&2
      exit 1
    fi
    if ! command -v ffmpeg >/dev/null 2>&1; then
      cat >&2 <<'MSG'
ffmpeg is not installed, so direct microphone recording is unavailable.

Use one of these options:
  1. Install ffmpeg: brew install ffmpeg
  2. Record with Voice Memos or QuickTime, export the file, then run:
     ./my_main/gemma4_audio_qa_llama.sh /path/to/your_recording.wav "<question>"
MSG
      exit 1
    fi
    QUESTION="$2"
    DURATION="${3:-$DURATION}"
    mkdir -p "$(dirname "$OUTPUT")"
    echo "Recording $DURATION seconds from AVFoundation device '$DEVICE'..."
    ffmpeg -hide_banner -y \
      -f avfoundation -i "$DEVICE" \
      -t "$DURATION" \
      -ac 1 -ar 16000 -c:a pcm_s16le \
      "$OUTPUT"
    echo "Created: $OUTPUT"
    AUDIO_FILE="$OUTPUT"
    ;;
  *)
    AUDIO_FILE="$1"
    QUESTION="${2:-$DEFAULT_AUDIO_QA_PROMPT}"
    ;;
esac

if [[ ! -f "$AUDIO_FILE" ]]; then
  echo "Audio file not found: $AUDIO_FILE" >&2
  exit 1
fi

case "${AUDIO_FILE##*.}" in
  wav|WAV)
    AUDIO_FORMAT="wav"
    ;;
  mp3|MP3)
    AUDIO_FORMAT="mp3"
    ;;
  *)
    echo "Unsupported audio format. Use .wav or .mp3" >&2
    exit 1
    ;;
esac

REQUEST_JSON="$(
  python3 - "$LLAMA_SERVER_MODEL" "$AUDIO_QA_SYSTEM_PROMPT" "$QUESTION" "$AUDIO_FORMAT" "$AUDIO_FILE" <<'PY'
import base64
import json
import pathlib
import sys

model, system_prompt, question, audio_format, audio_file = sys.argv[1:]
audio_bytes = pathlib.Path(audio_file).read_bytes()
audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

body = {
    "model": model,
    "messages": [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
                        "format": audio_format,
                    },
                },
            ],
        },
    ],
    "stream": False,
}

print(json.dumps(body, ensure_ascii=False))
PY
)"

curl_args=(
  -sS
  -H "Content-Type: application/json"
  -d "$REQUEST_JSON"
  "$LLAMA_SERVER_ENDPOINT"
)

if [[ -n "$LLAMA_API_KEY" ]]; then
  curl_args=(
    -sS
    -H "Content-Type: application/json"
    -H "Authorization: Bearer $LLAMA_API_KEY"
    -d "$REQUEST_JSON"
    "$LLAMA_SERVER_ENDPOINT"
  )
fi

if ! RESPONSE="$(curl "${curl_args[@]}")"; then
  cat >&2 <<MSG
Failed to connect to: $LLAMA_SERVER_ENDPOINT

If you are using Ollama, make sure it is running:
  ollama serve

If you are using llama-server instead, pass its endpoint explicitly:
  LLAMA_SERVER_ENDPOINT=http://localhost:8080/v1/chat/completions LLAMA_SERVER_MODEL=<model-alias> $0 ...
MSG
  exit 1
fi

python3 - "$RESPONSE" <<'PY'
import json
import sys

raw = sys.argv[1]
try:
    response = json.loads(raw)
except json.JSONDecodeError:
    print(raw)
    raise SystemExit(0)

if "choices" in response:
    message = response["choices"][0]["message"]["content"]
    if isinstance(message, str):
        print(message)
    else:
        print(json.dumps(message, ensure_ascii=False, indent=2))
elif "error" in response:
    print(json.dumps(response, ensure_ascii=False, indent=2))
else:
    print(json.dumps(response, ensure_ascii=False, indent=2))
PY
