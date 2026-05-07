#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"
BIN="$BUILD_DIR/bin/llama-cli"
DEFAULT_MODEL="$ROOT_DIR/models/model.gguf"
FALLBACK_MODEL="$ROOT_DIR/gemma-4-E4B-it-Q4_K_M.gguf"

MODEL_PATH="${1:-$DEFAULT_MODEL}"
shift $(( $# > 0 ? 1 : 0 ))

if [[ ! -x "$BIN" ]]; then
  echo "llama-cli not found. Building..."
  cmake -B "$BUILD_DIR"
  cmake --build "$BUILD_DIR" -j
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  if [[ "$MODEL_PATH" == "$DEFAULT_MODEL" && -f "$FALLBACK_MODEL" ]]; then
    MODEL_PATH="$FALLBACK_MODEL"
    echo "Using fallback model: $MODEL_PATH"
  else
    echo "Model file not found: $MODEL_PATH"
    echo "Usage: ./run-local.sh /path/to/model.gguf"
    exit 1
  fi
fi

echo "Starting llama-cli with model: $MODEL_PATH"
exec "$BIN" \
  -m "$MODEL_PATH" \
  -cnv \
  -ngl 999 \
  -c 4096 \
  --temp 0.7 \
  --reasoning off \
  "$@"
