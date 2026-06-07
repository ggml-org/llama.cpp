#!/usr/bin/env bash
# run_phase0.sh — build (if needed) and run the Phase 0 streaming script in Docker
#
# Usage:
#   ./my_main/docker/run_phase0.sh [phase0_streaming.py args...]
#
# Examples:
#   # WAV file (Silero-VAD, default)
#   ./my_main/docker/run_phase0.sh \
#       --audio my_main/sample/gemma4_audio_qa_input.wav
#
#   # WAV file + save detected turns for Phase 1 verification
#   ./my_main/docker/run_phase0.sh \
#       --audio my_main/sample/gemma4_audio_qa_input.wav \
#       --save-turns /workspace/turns
#
#   # WAV file + energy VAD (no silero-vad dep, useful for quick tests)
#   ./my_main/docker/run_phase0.sh \
#       --audio my_main/sample/gemma4_audio_qa_input.wav \
#       --vad-backend energy
#
# Environment variables:
#   MODEL    path to main GGUF  (default: auto-detected from HF cache)
#   MMPROJ   path to mmproj GGUF (default: auto-detected from HF cache)
#   GPUS     GPU selector passed to --gpus (default: all)
#   IMAGE    Docker image name   (default: gemma4-ua-phase0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"   # llama.cpp root
MY_MAIN="$REPO_ROOT/my_main"

IMAGE="${IMAGE:-gemma4-ua-phase0}"
GPUS="${GPUS:-all}"

# --------------------------------------------------------------------------- #
# Auto-detect GGUF model paths from HF cache
# --------------------------------------------------------------------------- #
HF_CACHE="${HOME}/.cache/huggingface/hub"
GGUF_SNAP="${HF_CACHE}/models--ggml-org--gemma-4-12B-it-GGUF/snapshots"

_find_gguf() {
    local pattern="$1"
    find "$GGUF_SNAP" -name "$pattern" 2>/dev/null | sort | tail -1
}

MODEL="${MODEL:-$(_find_gguf 'gemma-4-12B-it-Q4_K_M.gguf')}"
MMPROJ="${MMPROJ:-$(_find_gguf 'mmproj-gemma-4-12B-it-Q8_0.gguf')}"

if [[ -z "$MODEL" || -z "$MMPROJ" ]]; then
    echo "ERROR: Could not auto-detect model files." >&2
    echo "  Set MODEL and MMPROJ env vars explicitly:" >&2
    echo "    MODEL=/path/to/main.gguf MMPROJ=/path/to/mmproj.gguf $0 ..." >&2
    exit 1
fi

echo "[run_phase0] model  : $MODEL"
echo "[run_phase0] mmproj : $MMPROJ"

# --------------------------------------------------------------------------- #
# Build image if not already built
# --------------------------------------------------------------------------- #
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo "[run_phase0] Building Docker image '$IMAGE' (first run only, takes ~10min)..."
    docker build -t "$IMAGE" "$SCRIPT_DIR"
fi

# --------------------------------------------------------------------------- #
# Compute host-side turns directory if --save-turns is requested
# --------------------------------------------------------------------------- #
TURNS_VOL=""
TURNS_MOUNT=""
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    if [[ "${ARGS[$i]}" == "--save-turns" && $((i+1)) -lt ${#ARGS[@]} ]]; then
        HOST_TURNS="${ARGS[$((i+1))]}"
        mkdir -p "$HOST_TURNS"
        TURNS_VOL="-v $(realpath "$HOST_TURNS"):/workspace/turns"
        ARGS[$((i+1))]="/workspace/turns"
    fi
done

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
docker run --rm -it \
    --gpus "$GPUS" \
    -v "$MY_MAIN":/workspace/my_main \
    -v "$(dirname "$MODEL")":/models:ro \
    -v "$HF_CACHE":/root/.cache/huggingface:ro \
    $TURNS_VOL \
    "$IMAGE" \
    --model  "/models/$(basename "$MODEL")" \
    --mmproj "/models/$(basename "$MMPROJ")" \
    "${ARGS[@]}"
