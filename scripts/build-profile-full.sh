#!/usr/bin/env bash
# Compile-time profiling using clang -ftime-trace + ClangBuildAnalyzer.
#
# Requires ClangBuildAnalyzer:
# Install mac: brew install clang-build-analyzer
# Install linux: https://github.com/aras-p/ClangBuildAnalyzer.git

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-profile-full"
OUTPUT_BIN="${BUILD_DIR}/clang_analysis.bin"
JOBS="${1:--j$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"

if ! command -v clang++ &>/dev/null; then
    echo "error: clang++ not found" >&2
    exit 1
fi

if ! command -v ClangBuildAnalyzer &>/dev/null; then
    echo "error: ClangBuildAnalyzer not found" >&2
    echo "  brew install clangbuildanalyzer  (macOS)" >&2
    echo "  or: https://github.com/aras-p/ClangBuildAnalyzer/releases" >&2
    exit 1
fi

CLANG_VER=$(clang++ --version | head -1)
echo "compiler : ${CLANG_VER}"
echo "build dir: ${BUILD_DIR}"
echo "output   : ${OUTPUT_BIN}"
echo "jobs     : ${JOBS}"
echo

if command -v ccache &>/dev/null; then
    echo "clearing ccache..."
    ccache -C -z
fi

export CCACHE_DISABLE=1

cmake --fresh \
    -S "${ROOT_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_FLAGS="-ftime-trace" \
    -DCMAKE_CXX_FLAGS="-ftime-trace" \
    -DGGML_CCACHE=OFF \
    -DGGML_OPENMP=ON \
    -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_APP=OFF

echo

echo "Initializing ClangBuildAnalyzer..."
ClangBuildAnalyzer --start "${BUILD_DIR}"
echo

echo "building..."
echo

START=$(date +%s)

cmake --build "${BUILD_DIR}" --clean-first "${JOBS}"

END=$(date +%s)
ELAPSED=$((END - START))

echo
printf "build time: %ds (%dm %ds)\n" "${ELAPSED}" "$((ELAPSED / 60))" "$((ELAPSED % 60))"
echo

echo "Aggregating profile metrics..."
ClangBuildAnalyzer --stop "${BUILD_DIR}" "${OUTPUT_BIN}" > /dev/null

echo
echo "================================================================================"
ClangBuildAnalyzer --analyze "${OUTPUT_BIN}" | tee "${BUILD_DIR}/profile-report-full.txt"

echo
echo "largest trace files (top 20 by size):"
find "${BUILD_DIR}" -name "*.json" ! -name "compile_commands.json" \
    | xargs ls -l 2>/dev/null \
    | awk 'NF>5 {print $5, $NF}' \
    | sort -rn \
    | awk 'NR<=20 {printf "%8.1f KB  %s\n", $1/1024, $2}'

echo
echo "ClangBuildAnalyzer report was generated: ${BUILD_DIR}/profile-report-full.txt"
