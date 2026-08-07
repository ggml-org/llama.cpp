#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
PREFIX="${PREFIX:-$HOME/.local}"
JOBS="${JOBS:-$(nproc)}"
FORCE_MMQ="${FORCE_MMQ:-ON}"

if [[ -z "${CUDA_ARCH:-}" ]]; then
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ' || true)"
fi
CUDA_ARCH="${CUDA_ARCH:-75}"

for command in cmake git python3 nvcc nvidia-smi; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "error: required command not found: $command" >&2
        exit 1
    fi
done

cd "$ROOT"

python3 scripts/apply-tiered-dram-pinned-fallback.py
python3 scripts/apply-tiered-dram-matmul-staging.py
python3 scripts/apply-tiered-no-prompt-echo.py
python3 scripts/apply-tiered-hotpath-optimizations.py

python3 - <<'PY'
from pathlib import Path

path = Path("ggml/src/ggml-cuda/tiered.cu")
text = path.read_text(encoding="utf-8")
phrases = (
    "using a mapped pinned copy",
    "tiered-memory: SSD tensor %s is used by unsupported op %s",
    "tiered-memory: failed to stage DRAM weight %s: %s",
    "tiered-memory: failed to stream %s: %s",
)

malformed = []
for phrase in phrases:
    start = text.find(phrase)
    if start < 0:
        malformed.append(f"missing generated phrase: {phrase}")
        continue
    tail = text[start + len(phrase):]
    if not tail.startswith(r'\n"'):
        malformed.append(f"malformed escaped newline after: {phrase}")

if malformed:
    raise SystemExit("\n".join(malformed))

print(f"validated generated CUDA source: {path}")
PY

cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FORCE_MMQ="$FORCE_MMQ" \
    -DGGML_BACKEND_DL=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_TESTS=OFF

cmake --build "$BUILD_DIR" --target llama-tiered -j"$JOBS"

install -Dm755 "$BUILD_DIR/bin/llama-tiered" "$PREFIX/bin/llama-tiered"
rm -f "$PREFIX/bin/Summer.CPP" "$PREFIX/bin/summer"
mkdir -p "$HOME/models"

cat <<EOF

Summer.cpp installation complete.

  CUDA architecture : $CUDA_ARCH
  FORCE_MMQ          : $FORCE_MMQ
  llama-tiered       : $PREFIX/bin/llama-tiered
  model directory    : $HOME/models
  DRAM matmul        : temporary VRAM staging for cuBLAS compatibility
  prompt output      : generated tokens only

Add this line to your shell configuration when $PREFIX/bin is not in PATH:

  export PATH="$PREFIX/bin:\$PATH"

Then place a GGUF file in $HOME/models and run:

  llama-tiered \
    -m "$HOME/models/model.gguf" \
    --vram-mib 3800 \
    --dram-mib 6500 \
    -n 128 \
    "こんにちは"
EOF