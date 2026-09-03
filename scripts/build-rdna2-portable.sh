#!/usr/bin/env bash
# Build the RDNA2/V620 branch with user-portable ROCm discovery.
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
ROCM_PATH="${ROCM_PATH:-${HIP_PATH:-}}"
TARGET_ARCH="${TARGET_ARCH:-${AMDGPU_TARGETS:-${ROCM_DOCKER_ARCH:-}}}"
HIPCXX="${HIPCXX:-}"
JOBS="${JOBS:-}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS:-ON}"
GGML_NATIVE="${GGML_NATIVE:-ON}"
GGML_BACKEND_DL="${GGML_BACKEND_DL:-ON}"
GGML_CPU_ALL_VARIANTS="${GGML_CPU_ALL_VARIANTS:-ON}"
GGML_HIP_RCCL="${GGML_HIP_RCCL:-ON}"
BUILD_SPEC_SIDECARS="${BUILD_SPEC_SIDECARS:-ON}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: scripts/build-rdna2-portable.sh [options]

Discovers ROCm, clang, RCCL, and the GPU architecture where possible. Values
may also be supplied as environment variables before the command.

Options:
  --rocm-path PATH       ROCm installation root (or ROCM_PATH=PATH)
  --target-arch ARCH     AMD GPU target, normally gfx1030 (or TARGET_ARCH=ARCH)
  --build-dir PATH       CMake build directory (or BUILD_DIR=PATH)
  --jobs N               Parallel build jobs (or JOBS=N)
  --no-spec-sidecars     Skip the optional speculative sidecar libraries
                         (or BUILD_SPEC_SIDECARS=OFF)
  -h, --help             Show this help

Examples:
  ./scripts/build-rdna2-portable.sh
  ROCM_PATH=/opt/rocm TARGET_ARCH=gfx1030 ./scripts/build-rdna2-portable.sh
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
        --no-spec-sidecars)
            BUILD_SPEC_SIDECARS=OFF
            shift
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
    if command -v rocminfo >/dev/null 2>&1; then
        while IFS= read -r arch; do
            [[ -n "$arch" ]] && detected_arches+=("$arch")
        done < <(rocminfo 2>/dev/null | awk '$1 == "Name:" && $2 ~ /^gfx[0-9]+$/ { print $2 }' | sort -u)
    fi
    if [[ ${#detected_arches[@]} -eq 0 ]] && command -v hipconfig >/dev/null 2>&1; then
        while IFS= read -r arch; do
            [[ -n "$arch" ]] && detected_arches+=("$arch")
        done < <(hipconfig --gpus 2>/dev/null | grep -oE 'gfx[0-9]+' | sort -u || true)
    fi
    if [[ ${#detected_arches[@]} -eq 1 ]]; then
        TARGET_ARCH=${detected_arches[0]}
    elif [[ ${#detected_arches[@]} -gt 1 ]]; then
        fail "multiple GPU architectures detected (${detected_arches[*]}); set TARGET_ARCH explicitly"
    else
        TARGET_ARCH=gfx1030
        echo "WARNING: GPU architecture was not detected; defaulting to gfx1030. Set TARGET_ARCH for another GPU." >&2
    fi
fi

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

if [[ "$TARGET_ARCH" == "gfx1030" ]]; then
    export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}"
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
    -DGGML_NATIVE="$GGML_NATIVE"
    -DAMDGPU_TARGETS="$TARGET_ARCH"
    -DCMAKE_HIP_ARCHITECTURES="$TARGET_ARCH"
    -DCMAKE_HIP_COMPILER="$HIPCXX"
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH"
    -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS"
    -DGGML_BACKEND_DL="$GGML_BACKEND_DL"
    -DGGML_CPU_ALL_VARIANTS="$GGML_CPU_ALL_VARIANTS"
    -DLLAMA_BUILD_SERVER=ON
    -DLLAMA_BUILD_EXAMPLES=ON
    -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_SPEC_SIDECARS="$BUILD_SPEC_SIDECARS"
    -DLLAMA_SPEC_SIDECAR_HIP_ARCHITECTURES="$TARGET_ARCH"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

if rccl_requested; then
    has_rccl "$ROCM_PATH" || fail "RCCL was requested but was not found under $ROCM_PATH"
fi

echo "Configuring RDNA2 ROCm build"
echo "  source: $ROOT_DIR"
echo "  build:  $BUILD_DIR"
echo "  ROCm:   $ROCM_PATH"
echo "  clang:  $HIPCXX"
echo "  target: $TARGET_ARCH"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" "${CMAKE_FLAGS[@]}"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" --parallel "$JOBS"

SERVER="$BUILD_DIR/bin/llama-server"
[[ -x "$SERVER" ]] || fail "server binary was not produced: $SERVER"
if ! compgen -G "$BUILD_DIR/bin/*ggml-hip*.so*" >/dev/null 2>&1; then
    fail "HIP backend library was not produced under $BUILD_DIR/bin"
fi
case "$BUILD_SPEC_SIDECARS" in
    ON|1|TRUE|on|true)
        for so in spec_hip_sidecar.so spec_dflash_sidecar.so spec_qwen35moe_mtp_sidecar.so spec_qwen4exp_mtp_sidecar.so; do
            [[ -f "$BUILD_DIR/bin/$so" ]] || fail "speculative sidecar was not produced: $so"
        done
        ;;
esac

echo
echo "Build complete: $SERVER"
echo "For V620 runtime optimizations set HSA_OVERRIDE_GFX_VERSION=10.3.0 before launching."
case "$BUILD_SPEC_SIDECARS" in
    ON|1|TRUE|on|true)
        echo "Speculative sidecars built (dormant unless SPEC_SIDECAR=1 at runtime)."
        echo "For automatic discovery, place prepared bundles at:"
        echo "  $BUILD_DIR/bin/spec-sidecar-mtp and $BUILD_DIR/bin/spec-sidecar-dflash"
        echo "See docs/spec-sidecars.md. Disable at build time with BUILD_SPEC_SIDECARS=OFF."
        ;;
esac
