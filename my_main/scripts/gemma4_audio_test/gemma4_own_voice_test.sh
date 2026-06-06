#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSCRIBE_SCRIPT="$SCRIPT_DIR/gemma4_transcribe_file.sh"
OUTPUT="$SCRIPT_DIR/gemma4_own_voice.wav"
DURATION="10"
DEVICE=":1"
# PROMPT="日本語の音声として正確に文字起こししてください。あー、えー、あのー、えっと等のフィラーは出力せず、言い直しや詰まりは必要最小限に整えてください。出力は本文のみ。"
PROMPT="質問に答えてください。"

usage() {
  cat <<'USAGE'
Usage:
  ./my_main/gemma4_own_voice_test.sh <audio-file>
  ./my_main/gemma4_own_voice_test.sh --record [seconds]
  ./my_main/gemma4_own_voice_test.sh --list-devices

Modes:
  <audio-file>       Transcribe an audio file you recorded yourself.
  --record           Record from a microphone with ffmpeg, then transcribe.
  --list-devices     List macOS AVFoundation devices for ffmpeg.

Environment variables:
  OLLAMA_MODEL       Model name. Default inherited by gemma4_transcribe_file.sh: gemma4:e4b
  AUDIO_DEVICE       ffmpeg AVFoundation audio device. Default: :0
  AUDIO_OUTPUT       Output WAV for --record. Default: ./my_main/gemma4_own_voice.wav
  TRANSCRIBE_PROMPT  Prompt sent with the transcription request.

Examples:
  ./my_main/gemma4_own_voice_test.sh ~/Desktop/my_voice.wav
  ./my_main/gemma4_own_voice_test.sh --record 8
  AUDIO_DEVICE=':1' ./my_main/gemma4_own_voice_test.sh --record 10
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -eq 0 ]]; then
  usage
  exit 0
fi

if [[ ! -x "$TRANSCRIBE_SCRIPT" ]]; then
  echo "Missing executable transcription script: $TRANSCRIBE_SCRIPT" >&2
  echo "Run: chmod +x ./my_main/gemma4_transcribe_file.sh" >&2
  exit 1
fi

DEVICE="${AUDIO_DEVICE:-$DEVICE}"
OUTPUT="${AUDIO_OUTPUT:-$OUTPUT}"
PROMPT="${TRANSCRIBE_PROMPT:-$PROMPT}"

case "$1" in
  --list-devices)
    if ! command -v ffmpeg >/dev/null 2>&1; then
      echo "ffmpeg is not installed. Install it with: brew install ffmpeg" >&2
      exit 1
    fi
    ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 || true
    ;;
  --record)
    if ! command -v ffmpeg >/dev/null 2>&1; then
      cat >&2 <<'MSG'
ffmpeg is not installed, so direct microphone recording is unavailable.

Use one of these options:
  1. Install ffmpeg: brew install ffmpeg
  2. Record with Voice Memos or QuickTime, export the file, then run:
     ./my_main/gemma4_own_voice_test.sh /path/to/your_recording.wav
MSG
      exit 1
    fi
    DURATION="${2:-$DURATION}"
    mkdir -p "$(dirname "$OUTPUT")"
    echo "Recording $DURATION seconds from AVFoundation device '$DEVICE'..."
    ffmpeg -hide_banner -y \
      -f avfoundation -i "$DEVICE" \
      -t "$DURATION" \
      -ac 1 -ar 16000 -c:a pcm_s16le \
      "$OUTPUT"
    echo "Created: $OUTPUT"
    "$TRANSCRIBE_SCRIPT" "$OUTPUT" "$PROMPT"
    ;;
  *)
    AUDIO_FILE="$1"
    "$TRANSCRIBE_SCRIPT" "$AUDIO_FILE" "$PROMPT"
    ;;
esac
