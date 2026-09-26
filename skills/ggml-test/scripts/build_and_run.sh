#!/usr/bin/env bash
# Compile and run a single-file standalone ggml test program against an already-built llama.cpp tree, without touching the project's CMakeLists.txt.
#
# Usage:
#   skills/ggml-test/scripts/build_and_run.sh tmp/test_foo.cpp [-- program-args...]
#
# Env:
#   GGML_TEST_BUILD_DIR - path to the CMake build dir to link against (default: <repo root>/build). Must already contain bin/libggml*, i.e. the project must be built once.
set -euo pipefail

SRC="$1"; shift || true

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
BUILD_DIR="${GGML_TEST_BUILD_DIR:-$ROOT/build}"
BIN_DIR="$BUILD_DIR/bin"
OUT="${SRC%.cpp}"

if [ ! -d "$BIN_DIR" ]; then
    echo "error: $BIN_DIR not found - build llama.cpp once first, e.g.:" >&2
    echo "  cmake -B $BUILD_DIR && cmake --build $BUILD_DIR --target ggml -j" >&2
    exit 1
fi

g++ -std=c++17 -O0 -g \
    -I "$ROOT/ggml/include" \
    "$SRC" \
    -L "$BIN_DIR" -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,"$BIN_DIR" \
    -o "$OUT"

echo "compiled -> $OUT"
exec "$OUT" "$@"
