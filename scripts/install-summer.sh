#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
PREFIX="${PREFIX:-$HOME/.local}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
FORCE_MMQ="${FORCE_MMQ:-ON}"

for command in cmake install nvidia-smi nvcc python3; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "error: required command not found: $command" >&2
        exit 1
    fi
done

if [[ -z "${JOBS:-}" ]]; then
    if command -v nproc >/dev/null 2>&1; then
        JOBS="$(nproc)"
    else
        JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')"
    fi
fi
if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: JOBS must be a positive integer: $JOBS" >&2
    exit 1
fi

if [[ -z "${CUDA_ARCH:-}" ]]; then
    CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ' || true)"
fi
if ! [[ "${CUDA_ARCH:-}" =~ ^[0-9]+(;[0-9]+)*$ ]]; then
    echo "error: could not detect a valid CUDA architecture; set CUDA_ARCH explicitly, for example CUDA_ARCH=75" >&2
    exit 1
fi

FORCE_MMQ="${FORCE_MMQ^^}"
case "$FORCE_MMQ" in
    ON|OFF) ;;
    *)
        echo "error: FORCE_MMQ must be ON or OFF: $FORCE_MMQ" >&2
        exit 1
        ;;
esac

cd "$ROOT"

python3 scripts/apply-tiered-dram-pinned-fallback.py
python3 scripts/apply-tiered-dram-matmul-staging.py
python3 scripts/apply-tiered-no-prompt-echo.py

python3 - <<'PYCODE'
from pathlib import Path

cuda_path = Path("ggml/src/ggml-cuda/tiered.cu")
cuda_text = cuda_path.read_text(encoding="utf-8")
phrases = (
    "using a mapped pinned copy",
    "tiered-memory: SSD tensor %s is used by unsupported op %s",
    "tiered-memory: failed to stage DRAM weight %s: %s",
    "tiered-memory: failed to stream %s: %s",
)

malformed = []
for phrase in phrases:
    start = cuda_text.find(phrase)
    if start < 0:
        malformed.append(f"missing generated phrase: {phrase}")
        continue
    tail = cuda_text[start + len(phrase):]
    if not tail.startswith(r'\\n"'):
        malformed.append(f"malformed escaped newline after: {phrase}")

cli_path = Path("examples/tiered-memory/tiered.cpp")
cli_text = cli_path.read_text(encoding="utf-8")
for marker in ("LLAMA_ASCII_LOGO", "Summer.cpp tiered-memory CLI", "print_banner();"):
    if marker not in cli_text:
        malformed.append(f"missing CLI banner marker: {marker}")

if malformed:
    raise SystemExit("\\n".join(malformed))

print(f"validated generated CUDA source: {cuda_path}")
print(f"validated Summer.cpp CLI banner: {cli_path}")
PYCODE

cmake -S . -B "$BUILD_DIR" \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \\
    -DGGML_CUDA=ON \\
    -DGGML_CUDA_FORCE_MMQ="$FORCE_MMQ" \\
    -DGGML_BACKEND_DL=OFF \\
    -DBUILD_SHARED_LIBS=OFF \\
    -DLLAMA_BUILD_EXAMPLES=ON \\
    -DLLAMA_BUILD_TESTS=OFF

cmake --build "$BUILD_DIR" --target llama-tiered --parallel "$JOBS"

install -Dm755 "$BUILD_DIR/bin/llama-tiered" "$PREFIX/bin/llama-tiered"
if ! "$PREFIX/bin/llama-tiered" --help >/dev/null 2>&1; then
    echo "error: installed llama-tiered failed its --help smoke test" >&2
    exit 1
fi

rm -f "$PREFIX/bin/Summer.CPP" "$PREFIX/bin/summer"
mkdir -p "$MODEL_DIR"

cat <<EOF

Summer.cpp installation complete.

  CUDA architecture : $CUDA_ARCH
  FORCE_MMQ          : $FORCE_MMQ
  build jobs         : $JOBS
  llama-tiered       : $PREFIX/bin/llama-tiered
  model directory    : $MODEL_DIR
  CLI banner         : Summer.cpp enabled
  DRAM matmul        : temporary VRAM staging for cuBLAS compatibility
  prompt output      : generated tokens only

Add this line to your shell configuration when $PREFIX/bin is not in PATH:

  export PATH="$PREFIX/bin:\\$PATH"

Then place a GGUF file in $MODEL_DIR and run:

  llama-tiered \\
    -m "$MODEL_DIR/model.gguf" \\
    --vram-mib 3800 \\
    --dram-mib 6500 \\
    -n 128 \\
    "こんにちは"
EOF
