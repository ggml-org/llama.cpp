#!/usr/bin/env bash
# Build a separate native ROCm binary for one qualified RDNA architecture.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-${HIP_PATH:-}}"
TARGET_ARCH="${TARGET_ARCH:-}"
BUILD_DIR="${BUILD_DIR:-}"
JOBS="${JOBS:-}"
GGML_HIP_RCCL="${GGML_HIP_RCCL:-OFF}"
BUILD_TESTS="${BUILD_TESTS:-ON}"
BUILD_SIDECARS="${BUILD_SIDECARS:-ON}"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
usage() {
    cat <<'EOF'
Usage: scripts/build-rdna-unified.sh [--arch gfx1030|gfx1100] [--rocm PATH]
                                     [--build-dir PATH] [--jobs N]

Environment overrides: TARGET_ARCH, ROCM_PATH, BUILD_DIR, JOBS,
GGML_HIP_RCCL (default OFF), BUILD_TESTS, BUILD_SIDECARS.

Build gfx1030 and gfx1100 in separate directories. Mixed-architecture HIP
objects are intentionally rejected because their tuned compile definitions differ.
EOF
}

while (($#)); do
    case "$1" in
        --arch|--target-arch) [[ $# -ge 2 ]] || fail "$1 requires a value"; TARGET_ARCH=$2; shift 2 ;;
        --rocm|--rocm-path)   [[ $# -ge 2 ]] || fail "$1 requires a value"; ROCM_PATH=$2; shift 2 ;;
        --build-dir)          [[ $# -ge 2 ]] || fail "$1 requires a value"; BUILD_DIR=$2; shift 2 ;;
        --jobs)               [[ $# -ge 2 ]] || fail "$1 requires a value"; JOBS=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "unknown argument: $1" ;;
    esac
done

command -v cmake >/dev/null || fail "cmake is required"
command -v ninja >/dev/null || fail "ninja is required"

if [[ -z $ROCM_PATH ]]; then
    candidates=()
    while IFS= read -r path; do candidates+=("$path"); done < <(
        find /opt/rocm -maxdepth 1 -mindepth 1 -type d -name 'core-*' -print 2>/dev/null | sort -Vr)
    candidates+=(/opt/rocm)
    for path in "${candidates[@]}"; do
        if [[ -x $path/llvm/bin/clang++ && -x $path/bin/rocminfo ]]; then
            ROCM_PATH=$path
            break
        fi
    done
fi
[[ -n $ROCM_PATH && -x $ROCM_PATH/llvm/bin/clang++ ]] || fail "ROCm clang++ not found; set ROCM_PATH"
[[ -x $ROCM_PATH/bin/rocminfo ]] || fail "rocminfo not found under $ROCM_PATH"

if [[ -z $TARGET_ARCH ]]; then
    mapfile -t arches < <("$ROCM_PATH/bin/rocminfo" 2>/dev/null |
        awk '$1 == "Name:" && $2 ~ /^gfx(1030|1100)$/ { print $2 }' | sort -u)
    ((${#arches[@]} == 1)) || fail "expected one supported GPU architecture; detected: ${arches[*]:-(none)}"
    TARGET_ARCH=${arches[0]}
fi
case "$TARGET_ARCH" in
    gfx1030|gfx1100) ;;
    *) fail "unqualified target '$TARGET_ARCH'; use a separate reviewed port for other architectures" ;;
esac

if [[ $TARGET_ARCH == gfx1100 && -n ${HSA_OVERRIDE_GFX_VERSION:-} ]]; then
    fail "HSA_OVERRIDE_GFX_VERSION must be unset for native gfx1100"
fi

BUILD_DIR=${BUILD_DIR:-$ROOT_DIR/build-$TARGET_ARCH-unified}
if [[ -z $JOBS ]]; then
    ncpu=$(nproc)
    mem_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    mem_jobs=$((mem_kib / 4 / 1024 / 1024))
    ((mem_jobs >= 1)) || mem_jobs=1
    ((mem_jobs < ncpu)) && JOBS=$mem_jobs || JOBS=$ncpu
fi
[[ $JOBS =~ ^[1-9][0-9]*$ ]] || fail "JOBS must be a positive integer"

export ROCM_PATH HIP_PATH=$ROCM_PATH
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake_flags=(
    -G Ninja
    -DGGML_HIP=ON
    -DGGML_HIP_RCCL="$GGML_HIP_RCCL"
    -DGGML_HIP_GRAPHS=ON
    -DGGML_HIP_NO_VMM=ON
    -DGGML_HIP_MMQ_MFMA=ON
    -DGGML_NATIVE=ON
    -DGGML_BACKEND_DL=OFF
    -DGPU_TARGETS="$TARGET_ARCH"
    -DCMAKE_HIP_ARCHITECTURES="$TARGET_ARCH"
    -DCMAKE_HIP_COMPILER="$ROCM_PATH/llvm/bin/clang++"
    -DCMAKE_PREFIX_PATH="$ROCM_PATH"
    -DCMAKE_HIP_FLAGS=-mllvm\ --amdgpu-unroll-threshold-local=600
    -DLLAMA_BUILD_SERVER=ON
    -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_EXAMPLES=ON
    -DLLAMA_BUILD_TESTS="$BUILD_TESTS"
    -DLLAMA_BUILD_SPEC_SIDECARS="$BUILD_SIDECARS"
    -DLLAMA_SPEC_SIDECAR_HIP_ARCHITECTURES="$TARGET_ARCH"
    -DLLAMA_BUILD_UI=OFF
    -DCMAKE_BUILD_TYPE=Release
)

printf 'Configuring unified RDNA build\n  source: %s\n  build:  %s\n  ROCm:   %s\n  arch:   %s\n  jobs:   %s\n' \
    "$ROOT_DIR" "$BUILD_DIR" "$ROCM_PATH" "$TARGET_ARCH" "$JOBS"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" "${cmake_flags[@]}"

targets=(llama-server)
if [[ $BUILD_SIDECARS == ON ]]; then targets+=(spec-sidecar-hip-mtp spec-sidecar-hip-dflash); fi
if [[ $BUILD_TESTS == ON ]]; then targets+=(test-backend-ops); fi
cmake --build "$BUILD_DIR" --target "${targets[@]}" --parallel "$JOBS"

SERVER=$BUILD_DIR/bin/llama-server
HIP_LIB=$BUILD_DIR/bin/libggml-hip.so
[[ -x $SERVER && -e $HIP_LIB ]] || fail "expected server/HIP artifacts were not produced"
if ldd "$SERVER" "$HIP_LIB" | grep -q 'not found'; then
    ldd "$SERVER" "$HIP_LIB" | grep 'not found' >&2
    fail "unresolved shared library"
fi
grep -Eq "^CMAKE_HIP_ARCHITECTURES:[^=]*=$TARGET_ARCH$" "$BUILD_DIR/CMakeCache.txt" ||
    fail "CMake cache architecture mismatch"

{
    printf 'built_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'source_commit=%s\n' "$(git -C "$ROOT_DIR" rev-parse HEAD)"
    printf 'source_dirty=%s\n' "$([[ -n $(git -C "$ROOT_DIR" status --porcelain) ]] && echo yes || echo no)"
    printf 'rocm_path=%s\narch=%s\n' "$ROCM_PATH" "$TARGET_ARCH"
    sha256sum "$SERVER" "$HIP_LIB"
} > "$BUILD_DIR/rdna-build-manifest.txt"

printf 'Build complete: %s\nManifest: %s\n' "$SERVER" "$BUILD_DIR/rdna-build-manifest.txt"
