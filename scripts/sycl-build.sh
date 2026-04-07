#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/sycl-build.sh [options] [target] [-- <extra build args>]

Configure and build the SYCL backend with Ninja in build-sycl/.

Options:
  -r, --reconfigure       Force CMake reconfigure before building
  -c, --clean             Remove build-sycl/ and configure from scratch
  -B, --build-dir <dir>   Override build directory (default: build-sycl)
  -h, --help              Show this help

Examples:
  ./scripts/sycl-build.sh
  ./scripts/sycl-build.sh llama-completion
  ./scripts/sycl-build.sh -r llama-bench
  ./scripts/sycl-build.sh -c
  ./scripts/sycl-build.sh llama-completion -- -v
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build-sycl"

force_reconfigure=0
clean_build=0
target=""
extra_build_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--reconfigure)
            force_reconfigure=1
            shift
            ;;
        -c|--clean)
            clean_build=1
            force_reconfigure=1
            shift
            ;;
        -B|--build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            extra_build_args=("$@")
            break
            ;;
        *)
            if [[ -z "${target}" ]]; then
                target="$1"
            else
                extra_build_args+=("$1")
            fi
            shift
            ;;
    esac
done

if [[ ! -f /opt/intel/oneapi/setvars.sh ]]; then
    echo "error: /opt/intel/oneapi/setvars.sh not found" >&2
    exit 1
fi

# shellcheck disable=SC1091
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null
set -u

if ! command -v cmake >/dev/null 2>&1; then
    echo "error: cmake not found in PATH" >&2
    exit 1
fi

if ! command -v ninja >/dev/null 2>&1; then
    echo "error: ninja not found in PATH" >&2
    exit 1
fi

if ! command -v icx >/dev/null 2>&1 || ! command -v icpx >/dev/null 2>&1; then
    echo "error: icx/icpx not found after sourcing oneAPI" >&2
    exit 1
fi

if (( clean_build )); then
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

cmake_input_changed() {
    local stamp="${BUILD_DIR}/build.ninja"

    if [[ ! -f "${stamp}" ]]; then
        return 0
    fi

    find "${ROOT_DIR}" \
        -path "${BUILD_DIR}" -prune -o \
        \( -name 'CMakeLists.txt' -o -path "${ROOT_DIR}/cmake/*.cmake" \) \
        -newer "${stamp}" -print -quit | grep -q .
}

needs_configure=0

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" || ! -f "${BUILD_DIR}/build.ninja" ]]; then
    needs_configure=1
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]] && ! grep -q '^CMAKE_GENERATOR:INTERNAL=Ninja$' "${BUILD_DIR}/CMakeCache.txt"; then
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    needs_configure=1
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]] && ! grep -q '^CMAKE_C_COMPILER:FILEPATH=.*/icx$' "${BUILD_DIR}/CMakeCache.txt"; then
    needs_configure=1
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]] && ! grep -q '^CMAKE_CXX_COMPILER:FILEPATH=.*/icpx$' "${BUILD_DIR}/CMakeCache.txt"; then
    needs_configure=1
fi

if (( force_reconfigure )) || cmake_input_changed; then
    needs_configure=1
fi

configure_args=(
    -S "${ROOT_DIR}"
    -B "${BUILD_DIR}"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_SYCL=ON
    -DGGML_SYCL_TARGET=INTEL
    -DGGML_SYCL_ONECCL=ON
    -DCMAKE_C_COMPILER=icx
    -DCMAKE_CXX_COMPILER=icpx
)

if command -v ccache >/dev/null 2>&1; then
    configure_args+=(
        -DCMAKE_C_COMPILER_LAUNCHER=ccache
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    )
fi

if (( needs_configure )); then
    echo "[sycl-build] configuring ${BUILD_DIR}"
    cmake "${configure_args[@]}"
fi

jobs="${CMAKE_BUILD_PARALLEL_LEVEL:-}"
if [[ -z "${jobs}" ]]; then
    jobs="$(nproc)"
fi

build_cmd=(cmake --build "${BUILD_DIR}" --config Release -j "${jobs}")

if [[ -n "${target}" ]]; then
    build_cmd+=(--target "${target}")
fi

if [[ ${#extra_build_args[@]} -gt 0 ]]; then
    build_cmd+=(-- "${extra_build_args[@]}")
fi

echo "[sycl-build] building${target:+ target ${target}} with Ninja in ${BUILD_DIR}"
"${build_cmd[@]}"
