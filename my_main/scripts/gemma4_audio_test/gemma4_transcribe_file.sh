#!/usr/bin/env bash
set -euo pipefail

MODEL="${OLLAMA_MODEL:-gemma4:e4b}"
ENDPOINT="${OLLAMA_TRANSCRIPTION_ENDPOINT:-http://localhost:11434/v1/audio/transcriptions}"

usage() {
  cat <<'USAGE'
Usage:
  ./my_main/gemma4_transcribe_file.sh <audio-file> [prompt]

Environment variables:
  OLLAMA_MODEL                    Model name. Default: gemma4:e4b
  OLLAMA_TRANSCRIPTION_ENDPOINT   Endpoint. Default: http://localhost:11434/v1/audio/transcriptions

Examples:
  ./my_main/gemma4_transcribe_file.sh ./my_main/gemma4_test_ja.wav
  OLLAMA_MODEL=gemma4:e4b ./my_main/gemma4_transcribe_file.sh ./voice.wav "日本語として正確に文字起こししてください。"
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

AUDIO_FILE="$1"
PROMPT="${2:-}"

if [[ ! -f "$AUDIO_FILE" ]]; then
  echo "Audio file not found: $AUDIO_FILE" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required." >&2
  exit 1
fi

curl_args=(
  -sS
  "$ENDPOINT"
  -F "model=$MODEL"
  -F "file=@$AUDIO_FILE"
  -F "response_format=json"
)

if [[ -n "$PROMPT" ]]; then
  curl_args+=(-F "prompt=$PROMPT")
fi

response="$(curl "${curl_args[@]}")"

if command -v jq >/dev/null 2>&1; then
  text="$(printf '%s' "$response" | jq -r '.text // empty')"
  if [[ -n "$text" ]]; then
    printf '%s\n' "$text"
  else
    printf '%s\n' "$response" | jq .
  fi
else
  printf '%s\n' "$response"
fi
