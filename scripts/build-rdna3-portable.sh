#!/usr/bin/env bash
# Build the reviewed RDNA3/gfx11 branch with user-portable ROCm discovery.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build-gfx1100-portable}"
ROCM_PATH="${ROCM_PATH:-${HIP_PATH:-}}"
TARGET_ARCH="${TARGET_ARCH:-${AMDGPU_TARGETS:-${ROCM_DOCKER_ARCH:-}}}"
HIPCXX="${HIPCXX:-}"
JOBS="${JOBS:-}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS:-ON}"
GGML_NATIVE="${GGML_NATIVE:-ON}"
GGML_BACKEND_DL="${GGML_BACKEND_DL:-OFF}"
GGML_CPU_ALL_VARIANTS="${GGML_CPU_ALL_VARIANTS:-OFF}"
GGML_HIP_RCCL="${GGML_HIP_RCCL:-ON}"
BUILD_TESTS="${BUILD_TESTS:-OFF}"
BUILD_SIDECARS="${BUILD_SIDECARS:-ON}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: scripts/build-rdna3-portable.sh [options]

Discovers ROCm, clang, RCCL, and the GPU architecture where possible. Values
may also be supplied as environment variables before the command.

Options:
  --rocm-path PATH       ROCm installation root (or ROCM_PATH=PATH)
  --target-arch ARCH     AMD GPU target, normally gfx1100 (or TARGET_ARCH=ARCH)
  --build-dir PATH       CMake build directory (or BUILD_DIR=PATH)
  --jobs N               Parallel build jobs (or JOBS=N)

Environment: GGML_HIP_RCCL (default ON), BUILD_TESTS (default OFF),
BUILD_SIDECARS (default ON), GGML_BACKEND_DL (default OFF), and
GGML_CPU_ALL_VARIANTS (default OFF) control optional build features.
  -h, --help             Show this help

Examples:
  ./scripts/build-rdna3-portable.sh
  ROCM_PATH=/opt/rocm TARGET_ARCH=gfx1100 ./scripts/build-rdna3-portable.sh
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rocm-path)
            [[ $# -ge 2 ]] || fail "--rocm-path requires a value"
            ROCM_PATH=$2
            shift 2
            ;;
        --target-arch|--arch)
            [[ $# -ge 2 ]] || fail "$1 requires a value"
            TARGET_ARCH=$2
            shift 2
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || fail "--build-dir requires a value"
            BUILD_DIR=$2
            shift 2
            ;;
        --jobs)
            [[ $# -ge 2 ]] || fail "--jobs requires a value"
            JOBS=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1 (use --help)"
            ;;
    esac
done

command -v cmake >/dev/null 2>&1 || fail "cmake is not installed"

add_candidate() {
    local candidate=${1:-}
    [[ -n "$candidate" && -d "$candidate" ]] || return 0
    candidate=${candidate%/}
    local existing
    for existing in "${ROCM_CANDIDATES[@]:-}"; do
        [[ "$existing" != "$candidate" ]] || return 0
    done
    ROCM_CANDIDATES+=("$candidate")
}

find_clang() {
    local root=$1 candidate
    for candidate in \
        "$root/llvm/bin/clang" \
        "$root/lib/llvm/bin/clang" \
        "$root/bin/clang"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

has_rccl() {
    local root=$1 candidate
    for candidate in \
        "$root/lib/librccl.so" \
        "$root/lib64/librccl.so"; do
        [[ -e "$candidate" ]] && return 0
    done
    compgen -G "$root/lib/librccl.so.*" >/dev/null 2>&1 && return 0
    compgen -G "$root/lib64/librccl.so.*" >/dev/null 2>&1 && return 0
    return 1
}

rccl_requested() {
    case "$GGML_HIP_RCCL" in
        ON|1|TRUE|on|true) return 0 ;;
        *) return 1 ;;
    esac
}

ROCM_CANDIDATES=()
if [[ -n "$ROCM_PATH" ]]; then
    add_candidate "$ROCM_PATH"
else
    if command -v hipconfig >/dev/null 2>&1; then
        for hip_arg in -R -p; do
            hip_root=$(hipconfig "$hip_arg" 2>/dev/null || true)
            [[ -n "$hip_root" ]] && add_candidate "$hip_root"
        done
    fi
    add_candidate "/opt/rocm"
    for candidate in /opt/rocm-*; do
        add_candidate "$candidate"
    done
fi

[[ ${#ROCM_CANDIDATES[@]} -gt 0 ]] || fail "ROCm was not found; set ROCM_PATH=/path/to/rocm"

selected_rocm=""
for candidate in "${ROCM_CANDIDATES[@]}"; do
    if { [[ -n "$HIPCXX" ]] || find_clang "$candidate" >/dev/null 2>&1; } &&
            { ! rccl_requested || has_rccl "$candidate"; }; then
        selected_rocm=$candidate
        break
    fi
done

if [[ -z "$selected_rocm" ]]; then
    if [[ -n "${ROCM_PATH:-}" ]]; then
        fail "ROCm compiler or RCCL was not found under $ROCM_PATH"
    fi
    fail "ROCm clang with RCCL was not found; set ROCM_PATH explicitly or install RCCL"
fi
ROCM_PATH=$selected_rocm

if [[ -z "$HIPCXX" ]]; then
    HIPCXX=$(find_clang "$ROCM_PATH") || fail "ROCm clang was not found under $ROCM_PATH"
fi
[[ -x "$HIPCXX" ]] || fail "HIP compiler is not executable: $HIPCXX"

if [[ -z "$TARGET_ARCH" ]]; then
    detected_arches=()
    rocminfo_cmd="$ROCM_PATH/bin/rocminfo"
    if [[ ! -x "$rocminfo_cmd" ]] && command -v rocminfo >/dev/null 2>&1; then
        rocminfo_cmd=$(command -v rocminfo)
    fi
    if [[ -x "$rocminfo_cmd" ]]; then
        while IFS= read -r arch; do
            [[ -n "$arch" ]] && detected_arches+=("$arch")
        done < <("$rocminfo_cmd" 2>/dev/null | awk '$1 == "Name:" && $2 ~ /^gfx11[0-9]+$/ { print $2 }' | sort -u)
    fi
    if [[ ${#detected_arches[@]} -eq 0 ]] && [[ -x "$ROCM_PATH/bin/hipconfig" ]]; then
        while IFS= read -r arch; do
            [[ -n "$arch" ]] && detected_arches+=("$arch")
        done < <("$ROCM_PATH/bin/hipconfig" --gpus 2>/dev/null | grep -oE 'gfx11[0-9]+' | sort -u || true)
    fi
    if [[ ${#detected_arches[@]} -eq 1 ]]; then
        TARGET_ARCH=${detected_arches[0]}
    elif [[ ${#detected_arches[@]} -gt 1 ]]; then
        fail "multiple RDNA3 GPU architectures detected (${detected_arches[*]}); set TARGET_ARCH explicitly"
    else
        TARGET_ARCH=gfx1100
        echo "WARNING: RDNA3 GPU architecture was not detected; defaulting to gfx1100. Set TARGET_ARCH for another gfx11 target." >&2
    fi
fi
[[ "$TARGET_ARCH" =~ ^gfx11[0-9]+$ ]] ||
    fail "unqualified RDNA3 target '$TARGET_ARCH'; use build-rdna2-portable.sh for gfx1030"

if [[ "$GGML_BACKEND_DL" != "ON" && "$GGML_BACKEND_DL" != "1" && "$GGML_BACKEND_DL" != "TRUE" ]]; then
    case "$GGML_CPU_ALL_VARIANTS" in
        ON|1|TRUE|on|true) fail "GGML_CPU_ALL_VARIANTS requires GGML_BACKEND_DL=ON" ;;
    esac
fi
if [[ "$GGML_BACKEND_DL" == "ON" || "$GGML_BACKEND_DL" == "1" || "$GGML_BACKEND_DL" == "TRUE" ]]; then
    case "$BUILD_SHARED_LIBS" in
        ON|1|TRUE|on|true) ;;
        *) fail "GGML_BACKEND_DL=ON requires BUILD_SHARED_LIBS=ON" ;;
    esac
fi

if [[ "$TARGET_ARCH" == gfx11* && -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
    fail "HSA_OVERRIDE_GFX_VERSION must be unset for native RDNA3 targets"
fi
export HIP_PATH="$ROCM_PATH"
export HIPCXX
for libdir in "$ROCM_PATH/lib" "$ROCM_PATH/lib64"; do
    if [[ -d "$libdir" ]]; then
        export LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
done

if [[ -z "$JOBS" ]]; then
    JOBS=$(command -v nproc >/dev/null 2>&1 && nproc || printf '1')
fi
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || fail "JOBS must be a positive integer"

if pgrep -f "^$BUILD_DIR/bin/llama-server" >/dev/null 2>&1; then
    fail "llama-server from this build is running; stop it before rebuilding"
fi

CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-$ROCM_PATH}"
CMAKE_FLAGS=(
    -DGGML_HIP=ON
    -DGGML_HIP_RCCL="$GGML_HIP_RCCL"
    -DGGML_HIP_GRAPHS=ON
    -DGGML_HIP_NO_VMM=ON
    -DGGML_HIP_MMQ_MFMA=ON
    -DGGML_NATIVE="$GGML_NATIVE"
    -DAMDGPU_TARGETS="$TARGET_ARCH"
    -DGPU_TARGETS="$TARGET_ARCH"
    -DCMAKE_HIP_ARCHITECTURES="$TARGET_ARCH"
    -DCMAKE_HIP_COMPILER="$HIPCXX"
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"
    -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"
    -DGGML_BACKEND_DL="$GGML_BACKEND_DL"
    -DGGML_CPU_ALL_VARIANTS="$GGML_CPU_ALL_VARIANTS"
    -DLLAMA_BUILD_SERVER=ON
    -DLLAMA_BUILD_EXAMPLES=ON
    -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_TESTS="$BUILD_TESTS"
    -DLLAMA_BUILD_SPEC_SIDECARS="$BUILD_SIDECARS"
    -DLLAMA_SPEC_SIDECAR_HIP_ARCHITECTURES="$TARGET_ARCH"
    -DLLAMA_BUILD_UI=OFF
    -DLLAMA_USE_PREBUILT_UI=OFF
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

if rccl_requested; then
    has_rccl "$ROCM_PATH" || fail "RCCL was requested but was not found under $ROCM_PATH"
fi

echo "Configuring portable RDNA3 ROCm build"
echo "  source: $ROOT_DIR"
echo "  build:  $BUILD_DIR"
echo "  ROCm:   $ROCM_PATH"
echo "  clang:  $HIPCXX"
echo "  target: $TARGET_ARCH"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" "${CMAKE_FLAGS[@]}"
# When enabled, llama-server depends on the complete provider set through
# tools/spec-sidecar/CMakeLists.txt. Do not duplicate that list here.
targets=(llama-server)
if [[ "$BUILD_TESTS" == ON || "$BUILD_TESTS" == 1 || "$BUILD_TESTS" == TRUE || "$BUILD_TESTS" == on || "$BUILD_TESTS" == true ]]; then
    targets+=(test-backend-ops)
fi
cmake --build "$BUILD_DIR" --target "${targets[@]}" --config "$BUILD_TYPE" --parallel "$JOBS"

SERVER="$BUILD_DIR/bin/llama-server"
HIP_LIB="$BUILD_DIR/bin/libggml-hip.so"
[[ -x "$SERVER" && -e "$HIP_LIB" ]] || fail "server/HIP backend artifacts were not produced"
if [[ "$BUILD_SIDECARS" == ON || "$BUILD_SIDECARS" == 1 || "$BUILD_SIDECARS" == TRUE || "$BUILD_SIDECARS" == on || "$BUILD_SIDECARS" == true ]]; then
    for so in spec_hip_sidecar.so spec_dflash_sidecar.so spec_qwen35moe_mtp_sidecar.so spec_qwen4exp_mtp_sidecar.so; do
        [[ -f "$BUILD_DIR/bin/$so" ]] || fail "speculative sidecar was not produced: $so"
    done
fi
if ldd "$SERVER" "$HIP_LIB" | grep -q 'not found'; then
    ldd "$SERVER" "$HIP_LIB" | grep 'not found' >&2
    fail "unresolved shared library"
fi

{
    printf 'built_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'source_commit=%s\n' "$(git -C "$ROOT_DIR" rev-parse HEAD)"
    printf 'source_dirty=%s\n' "$([[ -n $(git -C "$ROOT_DIR" status --porcelain) ]] && echo yes || echo no)"
    printf 'rocm_path=%s\narch=%s\n' "$ROCM_PATH" "$TARGET_ARCH"
    sha256sum "$SERVER" "$HIP_LIB"
} > "$BUILD_DIR/rdna-build-manifest.txt"

echo
echo "Build complete: $SERVER"
echo "For the qualified native gfx1100 multi-GPU profile, build with RCCL (default) and launch with GGML_HIP_RDNA3_AUTO=1."
