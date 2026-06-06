#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${1:-$SCRIPT_DIR/gemma4_test_ja.wav}"
TEXT="${2:-こんにちは。これはジェンマフォーの文字起こしテストです。}"
VOICE="${SAY_VOICE:-Kyoko}"

usage() {
  cat <<'USAGE'
Usage:
  ./my_main/gemma4_generate_test_wav.sh [output-wav] [text]

Environment variables:
  SAY_VOICE   macOS say voice. Default: Kyoko

Examples:
  ./my_main/gemma4_generate_test_wav.sh
  ./my_main/gemma4_generate_test_wav.sh ./my_main/custom.wav "今日はGemma4の音声認識を試します。"
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v say >/dev/null 2>&1; then
  echo "macOS say command is required." >&2
  exit 1
fi

if ! command -v afconvert >/dev/null 2>&1; then
  echo "macOS afconvert command is required." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"
tmp_aiff="$(mktemp "${TMPDIR:-/tmp}/gemma4-test.XXXXXX.aiff")"
trap 'rm -f "$tmp_aiff"' EXIT

say -v "$VOICE" -o "$tmp_aiff" "$TEXT"
afconvert -f WAVE -d LEI16@16000 "$tmp_aiff" "$OUTPUT"

echo "Created: $OUTPUT"
echo "Text: $TEXT"
