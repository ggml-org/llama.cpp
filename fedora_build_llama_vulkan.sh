#!/usr/bin/env bash

# Build llama.cpp with the Vulkan backend and create a relocatable runtime bundle.
# This script intentionally does NOT run git pull. Update the repository manually before running it.

set -Eeuo pipefail
IFS=$'\n\t'

# -----------------------------
# User-configurable variables
# -----------------------------

# Existing llama.cpp Git repository. The script assumes you update this repo manually.
REPO_DIR="/home/tmatkovic/repos/llama.cpp"

# CMake build directory inside the repository.
BUILD_DIR="$REPO_DIR/build"

# Destination for versioned standalone Vulkan bundles.
OUTPUT_ROOT="/home/tmatkovic/llama.cpp/vulkan"

# Release is the appropriate configuration for normal llama.cpp inference use.
CMAKE_BUILD_TYPE="Release"

# Start from a clean CMake build tree. ccache still preserves compiled-object cache outside this directory.
CLEAN_BUILD=true

# Refuse to replace an already existing bundle with the same version/build name unless set to true.
OVERWRITE_BUNDLE=false

# Number of parallel compile jobs. nproc uses all available logical CPU threads.
JOBS="$(nproc)"

# Extra CMake options can be added here, one option per array element.
# Example: EXTRA_CMAKE_ARGS+=("-DGGML_NATIVE=OFF")
EXTRA_CMAKE_ARGS=()

# -----------------------------
# Helper functions
# -----------------------------

# Print an informational message.
info() {
    printf '\n==> %s\n' "$*"
}

# Print an error and terminate the script.
die() {
    printf '\nERROR: %s\n' "$*" >&2
    exit 1
}

# Check that a required command exists before starting a potentially long build.
require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command '$1' was not found in PATH."
}

# Remove the temporary staging directory if the script exits before publishing the bundle.
cleanup() {
    if [[ -n "${STAGING_DIR:-}" && -d "${STAGING_DIR:-}" ]]; then
        rm -rf -- "$STAGING_DIR"
    fi
}
trap cleanup EXIT

# -----------------------------
# Pre-flight checks
# -----------------------------

info "Checking required build/runtime tools"

# CMake configures llama.cpp.
require_command cmake
# Ninja is the selected CMake generator.
require_command ninja
# GCC/G++ compile the C and C++ code.
require_command gcc
require_command g++
# glslc compiles Vulkan shaders used by the ggml Vulkan backend.
require_command glslc
# vulkaninfo confirms that the host Vulkan stack is usable.
require_command vulkaninfo
# patchelf writes a relative RPATH so bundled binaries find bundled shared libraries.
require_command patchelf
# file is used to identify ELF binaries/libraries before copying and patching them.
require_command file
# ldd is used to verify that key runtime binaries have no unresolved shared-library dependencies.
require_command ldd
# sed/grep are used for version parsing and validation.
require_command sed
require_command grep

# Verify that the repository exists and looks like a llama.cpp checkout.
[[ -d "$REPO_DIR/.git" ]] || die "REPO_DIR is not a Git repository: $REPO_DIR"
[[ -f "$REPO_DIR/CMakeLists.txt" ]] || die "CMakeLists.txt was not found in: $REPO_DIR"

# Verify Vulkan before spending time compiling. Warnings from unrelated ICDs do not necessarily make this fail.
info "Checking Vulkan availability"
vulkaninfo --summary >/dev/null || die "vulkaninfo failed. Fix the Vulkan driver/loader setup before building."

# -----------------------------
# Configure and build
# -----------------------------

cd "$REPO_DIR"

if [[ "$CLEAN_BUILD" == "true" ]]; then
    info "Removing previous CMake build directory: $BUILD_DIR"
    # A clean tree avoids accidentally packaging stale binaries from an older configuration.
    rm -rf -- "$BUILD_DIR"
fi

info "Configuring llama.cpp: Vulkan=ON, build type=$CMAKE_BUILD_TYPE"
# Configure a native Release build with the Vulkan backend and Ninja generator.
cmake -S "$REPO_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -DGGML_VULKAN=ON \
    "${EXTRA_CMAKE_ARGS[@]}"

info "Building llama.cpp with $JOBS parallel jobs"
# Build all default targets so the bundle contains llama-server, llama-cli, llama-bench and other built tools.
cmake --build "$BUILD_DIR" -j "$JOBS"

# The bundle name is derived from the actual binary that was just built, not from Git assumptions.
SERVER_BIN="$BUILD_DIR/bin/llama-server"
[[ -x "$SERVER_BIN" ]] || die "Build completed but llama-server was not found at: $SERVER_BIN"

info "Reading llama.cpp version from the freshly built llama-server"
VERSION_OUTPUT="$($SERVER_BIN --version 2>&1)"
printf '%s\n' "$VERSION_OUTPUT"

# llama-server currently writes --version output to stderr, so stderr is captured above as well.
# Parse output such as: version: 0.5.0-dev (build 11369, commit 4ebdf2c74)
LLAMA_VERSION="$(printf '%s\n' "$VERSION_OUTPUT" | sed -n 's/^version: \([^ ]*\).*/\1/p' | head -n1)"
BUILD_NUMBER="$(printf '%s\n' "$VERSION_OUTPUT" | sed -n 's/.*(build \([0-9][0-9]*\),.*/\1/p' | head -n1)"
COMMIT="$(printf '%s\n' "$VERSION_OUTPUT" | sed -n 's/.*commit \([^)]*\)).*/\1/p' | head -n1)"

[[ -n "$LLAMA_VERSION" ]] || die "Could not parse llama.cpp version from llama-server --version."

# Prefer version + numeric build number. Fall back to the commit if master changes its version text format.
if [[ -n "$BUILD_NUMBER" ]]; then
    BUNDLE_NAME="llama-${LLAMA_VERSION}-${BUILD_NUMBER}"
elif [[ -n "$COMMIT" ]]; then
    BUNDLE_NAME="llama-${LLAMA_VERSION}-${COMMIT}"
else
    die "Could not parse either build number or commit from llama-server --version."
fi

FINAL_DIR="$OUTPUT_ROOT/$BUNDLE_NAME"
STAGING_DIR="$OUTPUT_ROOT/.${BUNDLE_NAME}.tmp.$$"

# Create the root directory that holds all versioned Vulkan bundles.
mkdir -p -- "$OUTPUT_ROOT"

if [[ -e "$FINAL_DIR" ]]; then
    if [[ "$OVERWRITE_BUNDLE" == "true" ]]; then
        info "Removing existing bundle because OVERWRITE_BUNDLE=true: $FINAL_DIR"
        rm -rf -- "$FINAL_DIR"
    else
        die "Bundle already exists: $FINAL_DIR (set OVERWRITE_BUNDLE=true only if you really want to replace it)"
    fi
fi

# Stage the new bundle in a temporary directory so a failed copy/patch/verification does not leave a final-looking bundle.
mkdir -p -- "$STAGING_DIR"

# -----------------------------
# Create standalone llama.cpp runtime bundle
# -----------------------------

info "Copying built ELF executables, shared libraries and their symlinks into the bundle"

# Copy every top-level ELF output from build/bin. This includes llama-server, llama-cli, llama-bench,
# other llama.cpp tools/tests, and the shared libraries needed by those binaries.
while IFS= read -r -d '' path; do
    if [[ -L "$path" ]]; then
        # Preserve shared-library symlinks (for example libllama.so -> libllama.so.0).
        cp -a -- "$path" "$STAGING_DIR/"
    elif [[ -f "$path" ]] && file -Lb "$path" | grep -q '^ELF '; then
        # Preserve ELF file permissions/timestamps while copying the actual binary/library.
        cp -a -- "$path" "$STAGING_DIR/"
    fi
done < <(find "$BUILD_DIR/bin" -maxdepth 1 \( -type f -o -type l \) -print0)

[[ -x "$STAGING_DIR/llama-server" ]] || die "llama-server was not copied into the staging bundle."
[[ -x "$STAGING_DIR/llama-cli" ]] || die "llama-cli was not copied into the staging bundle."
[[ -x "$STAGING_DIR/llama-bench" ]] || die "llama-bench was not copied into the staging bundle."

info "Setting RPATH to \$ORIGIN for relocatable llama.cpp ELF files"

# Patch each real ELF file that supports an ELF dynamic section. $ORIGIN means "the directory containing this file".
while IFS= read -r -d '' path; do
    if file -Lb "$path" | grep -q '^ELF ' && patchelf --print-rpath "$path" >/dev/null 2>&1; then
        patchelf --set-rpath '$ORIGIN' "$path"
    fi
done < <(find "$STAGING_DIR" -maxdepth 1 -type f -print0)

# -----------------------------
# Verification
# -----------------------------

info "Verifying key bundled executables"

for program in llama-server llama-cli llama-bench; do
    binary="$STAGING_DIR/$program"

    # Confirm that each important tool starts successfully from the bundle itself.
    "$binary" --version >/dev/null

    # Check dynamic dependencies and fail if any required shared library cannot be resolved.
    LDD_OUTPUT="$(ldd "$binary" 2>&1 || true)"
    if printf '%s\n' "$LDD_OUTPUT" | grep -q 'not found'; then
        printf '%s\n' "$LDD_OUTPUT" >&2
        die "$program has unresolved shared-library dependencies."
    fi
done

# Confirm that the packaged CLI can enumerate the Vulkan backend/devices.
DEVICE_OUTPUT="$($STAGING_DIR/llama-cli --list-devices)"
printf '%s\n' "$DEVICE_OUTPUT"
printf '%s\n' "$DEVICE_OUTPUT" | grep -q 'Vulkan' || die "The bundled llama-cli did not report any Vulkan device."

# Record useful build metadata inside the bundle for later identification.
{
    printf 'bundle_name=%s\n' "$BUNDLE_NAME"
    printf 'llama_version=%s\n' "$LLAMA_VERSION"
    printf 'build_number=%s\n' "${BUILD_NUMBER:-}"
    printf 'commit=%s\n' "${COMMIT:-}"
    printf 'cmake_build_type=%s\n' "$CMAKE_BUILD_TYPE"
    printf 'ggml_vulkan=ON\n'
    printf 'built_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'source_repo=%s\n' "$REPO_DIR"
} > "$STAGING_DIR/BUILD_INFO.txt"

# Publish the bundle only after every verification above has succeeded.
mv -- "$STAGING_DIR" "$FINAL_DIR"
STAGING_DIR=""

info "Standalone llama.cpp Vulkan bundle created successfully"
printf 'Bundle name: %s\n' "$BUNDLE_NAME"
printf 'Bundle path: %s\n' "$FINAL_DIR"
printf 'Server:      %s/llama-server\n' "$FINAL_DIR"
printf 'CLI:         %s/llama-cli\n' "$FINAL_DIR"
printf 'Bench:       %s/llama-bench\n' "$FINAL_DIR"
printf '\nUpdate LLAMA_VULKAN_DIR in ~/.bashrc to:\n'
printf 'LLAMA_VULKAN_DIR="%s"\n' "$FINAL_DIR"
