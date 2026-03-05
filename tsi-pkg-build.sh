#!/usr/bin/env bash
# Safe to source: does not leak -e/-u into the interactive shell.
#
# ============================
# USAGE (source is recommended)
# ============================
#
#   source tsi-pkg-build.sh [build-mode] [flags...] [MLIR_COMPILER_DIR] [TOOLBOX_DIR]
#
# Build modes (optional):
#   release | debug | debug-detail
#     - release      : GGML_PERF_RELEASE
#     - debug        : GGML_PERF_DETAIL
#     - debug-detail : GGML_PERF_DETAIL + TMU_DEBUG_VALIDATE
#
# Submodules:
#   - First run in a fresh repo checkout: auto "git submodule update --init --recursive"
#   - Later runs: submodule update is OPTIONAL unless you pass:
#       git-submodule-pull
#   - If ggml-tsi-kernel is missing, script forces submodule init even without the flag.
#
# Blob build (OFF by default):
#   build-fpga-blobs     : build blobs in ggml-tsi-kernel/fpga-kernel only
#   build-posix-blobs    : build blobs in ggml-tsi-kernel/posix-kernel only
#   build-all-blobs      : build blobs for both fpga+posix kernels
#
# Python virtual env:
#   overwrite-venv       : delete blob-creation venv and recreate it (installs deps)
#                          NOTE: this alone does NOT build blobs unless blob flag is also set.
#   git-submodule-pull   : ALSO forces overwrite-venv AND build-all-blobs (as requested)
#
# Build selection:
#   Default (no build-selection flags): build-posix + build-fpga + package
#   build-posix          : only build posix C/C++
#   build-fpga           : only build fpga target
#   package              : only package fpga bundle (requires fpga already built)
#   (You can combine them: e.g. build-posix build-fpga)
#
# Cleanup:
#   clean                : rm -rf build-posix build-fpga (llama.cpp) and kernel build dirs in ggml-tsi-kernel
#   clean-all            : clean + remove python venv blob-creation
#
# Coverage:
#   enable_coverage      : adds -DENABLE_COVERAGE=ON
#
# Help:
#   help | -h | --help
#
# Examples:
#   source tsi-pkg-build.sh debug-detail
#   source tsi-pkg-build.sh debug git-submodule-pull
#   source tsi-pkg-build.sh debug build-all-blobs overwrite-venv
#   source tsi-pkg-build.sh release build-posix
#   source tsi-pkg-build.sh debug build-fpga package
#   source tsi-pkg-build.sh clean
#

log_error(){ echo "ERROR: $*" >&2; }
log_info(){  echo "INFO:  $*"; }

# detect sourced vs executed
__TSI_SOURCED=0
(return 0 2>/dev/null) && __TSI_SOURCED=1 || true
__TSI_OLD_SET="$(set +o)"
__OLD_VIRTUAL_ENV=""

run() { "$@"; local rc=$?; [ $rc -eq 0 ] || log_error "cmd failed ($rc): $*"; return $rc; }
absdir() { (cd "$1" 2>/dev/null && pwd); }
tolower(){ echo "$1" | tr '[:upper:]' '[:lower:]'; }

die() {
  log_error "$*"
  if [ "$__TSI_SOURCED" -eq 1 ]; then return 1; else exit 1; fi
}

cleanup() {
  # leave/restore venv
  if [ -n "${__OLD_VIRTUAL_ENV:-}" ] && [ -f "${__OLD_VIRTUAL_ENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${__OLD_VIRTUAL_ENV}/bin/activate" >/dev/null 2>&1 || true
  else
    type deactivate >/dev/null 2>&1 && deactivate || true
  fi

  # restore caller shell behavior
  eval "${__TSI_OLD_SET}" >/dev/null 2>&1 || true
  stty sane 2>/dev/null || true

  # don't leak traps into parent shell
  trap - RETURN EXIT 2>/dev/null || true
}

usage() {
  sed -n '1,120p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//'
}

select_arch() {
  local m; m="$(uname -m)"
  case "$m" in
    x86_64|amd64) echo "x86_64" ;;
    aarch64|arm64) echo "aarch64" ;;
    *) log_error "Unsupported host arch from uname -m: $m"; return 2 ;;
  esac
}

# -------------------------
# Submodule init/update logic
# -------------------------
submodule_marker_file() { echo ".tsi_submodules_initialized"; }

is_first_time_repo() { [ ! -f "$(submodule_marker_file)" ]; }
mark_repo_initialized() { : > "$(submodule_marker_file)" || true; }

submodule_self_heal_if_needed() {
  # Fix stale/bad state for ggml-tsi-kernel specifically:
  # directory exists and non-empty but not a proper submodule checkout.
  if [ -e "ggml-tsi-kernel" ]; then
    if [ ! -d "ggml-tsi-kernel/.git" ] && [ -n "$(ls -A ggml-tsi-kernel 2>/dev/null || true)" ]; then
      log_info "ggml-tsi-kernel exists and is non-empty; cleaning stale submodule state"
      run git submodule deinit -f -- ggml-tsi-kernel || true
      run rm -rf ggml-tsi-kernel || return 1
      run rm -rf .git/modules/ggml-tsi-kernel || return 1
    fi
  fi
  return 0
}

ensure_submodules() {
  # want_update: 0/1 (user flag)
  local want_update="$1"
  local first_time=0
  is_first_time_repo && first_time=1 || true

  # If submodule directory is missing, we must init regardless of optional policy.
  if [ ! -d "ggml-tsi-kernel" ]; then
    log_info "ggml-tsi-kernel missing; forcing submodule init"
    want_update=1
  fi

  if [ "$first_time" -eq 1 ]; then
    log_info "First-time repo setup detected; initializing submodules automatically"
    want_update=1
  fi

  if [ "$want_update" -eq 1 ]; then
    submodule_self_heal_if_needed || return 1
    run git submodule update --recursive --init || return 1
    if [ "$first_time" -eq 1 ]; then
      mark_repo_initialized
      log_info "Submodules initialized; future runs will skip by default (use git-submodule-pull to force update)."
    fi
  else
    log_info "Skipping git submodule update (already initialized). Use git-submodule-pull to enable."
  fi
  return 0
}

# -------------------------
# Args/flags
# -------------------------
parse_args() {
  # Defaults from env
  BUILD_TYPE=""
  MLIR_COMPILER_DIR_IN="${MLIR_COMPILER_DIR:-}"
  TOOLBOX_DIR_IN="${TOOLBOX_DIR:-}"

  ENABLE_COVERAGE_FLAG=""

  # Submodules
  GIT_SUBMODULE_PULL=0

  # Blobs (default OFF)
  DO_BLOB_FPGA=0
  DO_BLOB_POSIX=0

  # Python venv
  OVERWRITE_VENV=0

  # Build selection (default: posix+fpga+package)
  DO_BUILD_POSIX=1
  DO_BUILD_FPGA=1
  DO_PACKAGE_FPGA=1
  __USER_BUILD_SELECT=0

  # Cleanup
  DO_CLEAN=0
  DO_CLEAN_ALL=0

  local a
  for a in "$@"; do
    case "$(tolower "$a")" in
      help|-h|--help)
        usage
        if [ "$__TSI_SOURCED" -eq 1 ]; then return 0; else exit 0; fi
        ;;
      release|debug|debug-detail)
        [ -z "${BUILD_TYPE}" ] && BUILD_TYPE="$a"
        ;;
      enable_coverage)
        ENABLE_COVERAGE_FLAG="-DENABLE_COVERAGE=ON"
        log_info "enable_coverage detected"
        ;;
      git-submodule-pull)
        GIT_SUBMODULE_PULL=1
        log_info "git-submodule-pull detected (will update submodules)"
        ;;
      build-fpga-blobs)
        DO_BLOB_FPGA=1
        log_info "build-fpga-blobs detected"
        ;;
      build-posix-blobs)
        DO_BLOB_POSIX=1
        log_info "build-posix-blobs detected"
        ;;
      build-all-blobs)
        DO_BLOB_FPGA=1
        DO_BLOB_POSIX=1
        log_info "build-all-blobs detected"
        ;;
      overwrite-venv)
        OVERWRITE_VENV=1
        log_info "overwrite-venv detected"
        ;;
      build-posix|posix)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_FPGA=0; DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_BUILD_POSIX=1
        log_info "build-posix selected"
        ;;
      build-fpga|fpga)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_FPGA=0; DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_BUILD_FPGA=1
        log_info "build-fpga selected"
        ;;
      package|bundle)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_FPGA=0; DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_PACKAGE_FPGA=1
        log_info "package selected"
        ;;
      clean)
        DO_CLEAN=1
        log_info "clean selected"
        ;;
      clean-all)
        DO_CLEAN_ALL=1
        log_info "clean-all selected"
        ;;
      *)
        # positional paths (only if not already set)
        if [ -z "${MLIR_COMPILER_DIR_IN}" ]; then
          MLIR_COMPILER_DIR_IN="$a"
        elif [ -z "${TOOLBOX_DIR_IN}" ]; then
          TOOLBOX_DIR_IN="$a"
        fi
        ;;
    esac
  done

  # Required behavior: git-submodule-pull ALSO deletes+recreates venv and builds all blobs.
  if [ "${GIT_SUBMODULE_PULL}" -eq 1 ]; then
    OVERWRITE_VENV=1
    DO_BLOB_FPGA=1
    DO_BLOB_POSIX=1
    log_info "git-submodule-pull => forcing overwrite-venv + build-all-blobs"
  fi
}

resolve_paths() {
  local arch="$1"

  if [ -z "${MLIR_COMPILER_DIR_IN}" ]; then
    MLIR_SDK_VERSION="${MLIR_SDK_VERSION:-/proj/rel/sw/sdk-r.0.2.5/${arch}}"
    MLIR_COMPILER_DIR_IN="${MLIR_SDK_VERSION}/compiler"
    log_info "Using default MLIR_COMPILER_DIR: ${MLIR_COMPILER_DIR_IN}"
  fi

  if [ -z "${TOOLBOX_DIR_IN}" ]; then
    MLIR_SDK_VERSION="${MLIR_SDK_VERSION:-$(dirname "${MLIR_COMPILER_DIR_IN}")}"
    TOOLBOX_DIR_IN="${MLIR_SDK_VERSION}/toolbox/build/install-fpga"
    log_info "Using default TOOLBOX_DIR: ${TOOLBOX_DIR_IN}"
  fi

  MLIR_COMPILER_DIR="$(absdir "${MLIR_COMPILER_DIR_IN}")" || die "MLIR_COMPILER_DIR not found: ${MLIR_COMPILER_DIR_IN}"
  TOOLBOX_DIR="$(absdir "${TOOLBOX_DIR_IN}")" || die "TOOLBOX_DIR not found: ${TOOLBOX_DIR_IN}"

  export MLIR_SDK_VERSION="${MLIR_SDK_VERSION:-$(dirname "${MLIR_COMPILER_DIR}")}"
  export MLIR_COMPILER_DIR
  export COMPILER_INSTALL_DIR="${MLIR_COMPILER_DIR}"
  export TOOLBOX_DIR
  export FAU_LOOKUP_TABLE_PATH="${MLIR_SDK_VERSION}/ffm/txe-ffm-cpp/third-party/FAU/include/"

  log_info "MLIR_COMPILER_DIR: ${MLIR_COMPILER_DIR}"
  log_info "TOOLBOX_DIR: ${TOOLBOX_DIR}"
}

setup_toolchain() {
  export CC="/proj/local/gcc-13.3.0/bin/gcc"
  export CXX="/proj/local/gcc-13.3.0/bin/g++"
  export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:${LD_LIBRARY_PATH:-}"
}

# -------------------------
# Python env (only when requested)
# -------------------------
setup_python() {
  log_info "python venv: ensuring blob-creation exists"
  __OLD_VIRTUAL_ENV="${VIRTUAL_ENV:-}"

  if [ "${OVERWRITE_VENV}" -eq 1 ] && [ -d "blob-creation" ]; then
    log_info "overwrite-venv: removing existing blob-creation venv"
    rm -rf blob-creation || return 1
  fi

  # If venv exists, activate
  if [ -d "blob-creation" ] && [ -f "blob-creation/bin/activate" ]; then
    log_info "blob-creation virtual env already exists; activating"
    # shellcheck disable=SC1091
    run bash -c 'source blob-creation/bin/activate && python -V >/dev/null' || return 1
    # shellcheck disable=SC1091
    source blob-creation/bin/activate || return 1
    return 0
  fi

  # Otherwise create + install deps
  run /proj/local/Python-3.11.12/bin/python3 -m venv blob-creation || return 1
  # shellcheck disable=SC1091
  run bash -c 'source blob-creation/bin/activate && python -V >/dev/null' || return 1
  # shellcheck disable=SC1091
  source blob-creation/bin/activate || return 1

  log_info "installing mlir and python dependencies"
  run pip install --upgrade pip || return 1
  run pip install torch==2.7.0 || return 1
  run pip install -r "${MLIR_COMPILER_DIR}/python/requirements-common.txt" || return 1

  local MLIR_WHL
  MLIR_WHL="$(ls "${MLIR_COMPILER_DIR}/python"/mlir_external_packages-*.whl 2>/dev/null | head -1 || true)"
  if [ -n "${MLIR_WHL}" ]; then
    run pip install "${MLIR_WHL}" || return 1
  else
    log_info "WARNING: mlir_external_packages wheel not found in ${MLIR_COMPILER_DIR}/python/"
  fi

  run pip install onnxruntime-training || return 1
  return 0
}

# -------------------------
# Blob build steps (explicit only)
# -------------------------
build_fpga_blobs() {
  log_info "BLOB: creating fpga kernels/blobs"
  # In ggml-tsi-kernel/fpga-kernel
  run cmake -B build-fpga -DTOOLBOX_DIR="${TOOLBOX_DIR}" -DCOMPILER_INSTALL_DIR="${MLIR_COMPILER_DIR}" || return 1
  run ./create-all-kernels.sh || return 1
  return 0
}

build_posix_blobs() {
  log_info "BLOB: creating posix kernels/blobs"
  # In ggml-tsi-kernel/posix-kernel
  run ./create-all-kernels.sh || return 1
  return 0
}

# -------------------------
# Build and package (llama.cpp root)
# -------------------------
build_posix() {
  log_info "building llama.cpp/ggml for posix"
  local bt; bt="$(tolower "${BUILD_TYPE}")"
  local common="-DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DGGML_NATIVE=ON -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF"
  local cflags_base="-DGGML_TARGET_POSIX -DGGML_TSAVORITE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"
  local perf="-DGGML_PERF"
  local tmu_debug=""

  [ "$bt" = "release" ] && perf="-DGGML_PERF_RELEASE"
  [ "$bt" = "debug" ] && perf="-DGGML_PERF_DETAIL"
  if [ "$bt" = "debug-detail" ]; then
    perf="-DGGML_PERF_DETAIL"
    tmu_debug="-DTMU_DEBUG_VALIDATE"
    log_info "TMU_DEBUG_VALIDATE ENABLED (debug-detail build)"
  fi

  run cmake -B build-posix ${common} \
    -DCMAKE_C_COMPILER="${CC}" -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="${perf} ${tmu_debug} ${cflags_base}" \
    -DCMAKE_CXX_FLAGS="${perf} ${tmu_debug} ${cflags_base}" \
    ${ENABLE_COVERAGE_FLAG} || return 1

  run cmake --build build-posix --config Release || return 1
  return 0
}

wrap_glibc_bins() {
  log_info "fixing GLIBC compatibility for TSI binaries"

  [ -f build-posix/bin/simple-backend-tsi ] || return 0
  [ -f build-posix/bin/simple-backend-tsi-original ] || mv build-posix/bin/simple-backend-tsi build-posix/bin/simple-backend-tsi-original || return 1
  cat > build-posix/bin/simple-backend-tsi <<'EOL'
#!/bin/bash
export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:$LD_LIBRARY_PATH"
exec "$(dirname "$0")/simple-backend-tsi-original" "$@"
EOL
  chmod +x build-posix/bin/simple-backend-tsi || return 1

  [ -f build-posix/bin/llama-cli ] || return 0
  [ -f build-posix/bin/llama-cli-original ] || mv build-posix/bin/llama-cli build-posix/bin/llama-cli-original || return 1
  cat > build-posix/bin/llama-cli <<'EOL'
#!/bin/bash
export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:$LD_LIBRARY_PATH"
exec "$(dirname "$0")/llama-cli-original" "$@"
EOL
  chmod +x build-posix/bin/llama-cli || return 1
  return 0
}

build_fpga() {
  log_info "building llama.cpp/ggml for fpga"
  local bt; bt="$(tolower "${BUILD_TYPE}")"
  local ARM_TOOLCHAIN_FILE="${TOOLBOX_DIR}/lib/cmake/toolchains/arm.cmake"
  local perf="-DGGML_PERF"
  [ "$bt" = "release" ] && perf="-DGGML_PERF_RELEASE"
  [ "$bt" = "debug" ] && perf="-DGGML_PERF_DETAIL"
  [ "$bt" = "debug-detail" ] && perf="-DGGML_PERF_DETAIL"

  run cmake -B build-fpga \
    -DCMAKE_TOOLCHAIN_FILE="${ARM_TOOLCHAIN_FILE}" \
    -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=fpga -DLLAMA_CURL=OFF \
    -DCMAKE_C_FLAGS="${perf} -DGGML_TSAVORITE" \
    -DCMAKE_CXX_FLAGS="${perf} -DGGML_TSAVORITE" \
    ${ENABLE_COVERAGE_FLAG} || return 1

  run cmake --build build-fpga --config Release || return 1
  return 0
}

bundle_fpga() {
  log_info "creating tar bundle for fpga"
  local TSI_GGML_VERSION=0.2.5
  local TSI_GGML_BUNDLE_INSTALL_DIR=tsi-ggml
  local GGML_TSI_INSTALL_DIR=ggml-tsi-kernel
  local TSI_GGML_RELEASE_DIR=/proj/rel/sw/ggml
  local TSI_BLOB_INSTALL_DIR
  TSI_BLOB_INSTALL_DIR="$(pwd)/${GGML_TSI_INSTALL_DIR}/fpga-kernel/build-fpga"

  # Sanity checks: package requires built fpga binaries
  [ -f "build-fpga/bin/llama-cli" ] || die "package requested but build-fpga/bin/llama-cli not found. Run with build-fpga first."

  mkdir -p "${TSI_GGML_BUNDLE_INSTALL_DIR}"
  rm -f "${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh"

  cat > "./${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh" <<'EOL'
#!/bin/bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(pwd)

tsi_kernels=("add" "sub" "mult" "div" "abs" "inv" "neg" "sin" "sqrt" "sqr" "sigmoid" "silu" "rms_norm" "swiglu" \
"add_16" "sub_16" "mult_16" "div_16" "abs_16" "inv_16" "neg_16" "sin_16" "sqrt_16" "sqr_16" "sigmoid_16" "silu_16" "rms_norm_16" "swiglu_16")

for kernel in "${tsi_kernels[@]}"; do
  mkdir -p __TSI_BLOB_INSTALL_DIR__/txe_${kernel}
  cp blobs __TSI_BLOB_INSTALL_DIR__/txe_${kernel}/ -r
done
EOL

  sed -i "s|__TSI_BLOB_INSTALL_DIR__|${TSI_BLOB_INSTALL_DIR}|g" "./${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh"
  chmod +x "./${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh" || return 1

  # These are repo-provided fpga blobs directory (not the generated per-kernel build dirs)
  cp "${GGML_TSI_INSTALL_DIR}/fpga/blobs" "${TSI_GGML_BUNDLE_INSTALL_DIR}/" -r || return 1

  cp build-fpga/bin/llama-cli "${TSI_GGML_BUNDLE_INSTALL_DIR}/" || return 1
  cp build-fpga/bin/libggml*.so "${TSI_GGML_BUNDLE_INSTALL_DIR}/" || return 1
  cp build-fpga/bin/libllama*.so "${TSI_GGML_BUNDLE_INSTALL_DIR}/" || return 1
  cp build-fpga/bin/simple-backend-tsi "${TSI_GGML_BUNDLE_INSTALL_DIR}/" || return 1

  tar -cvzf "${TSI_GGML_BUNDLE_INSTALL_DIR}-${TSI_GGML_VERSION}.tz" "${TSI_GGML_BUNDLE_INSTALL_DIR}"/* || return 1

  if [ "$(tolower "$BUILD_TYPE")" = "release" ]; then
    cp "${TSI_GGML_BUNDLE_INSTALL_DIR}-${TSI_GGML_VERSION}.tz" "${TSI_GGML_RELEASE_DIR}/" || return 1

    local LATEST_TZ="${TSI_GGML_BUNDLE_INSTALL_DIR}-${TSI_GGML_VERSION}.tz"
    local LATEST_FULL_PATH="${TSI_GGML_RELEASE_DIR}/$(basename "$LATEST_TZ")"
    rm -f "${TSI_GGML_RELEASE_DIR}/tsi-ggml-aws-latest.tz" "${TSI_GGML_RELEASE_DIR}/tsi-ggml-latest.tz"
    ln -s "/aws${LATEST_FULL_PATH}" "${TSI_GGML_RELEASE_DIR}/tsi-ggml-aws-latest.tz"
    ln -s "${LATEST_FULL_PATH}" "${TSI_GGML_RELEASE_DIR}/tsi-ggml-latest.tz"
    log_info "Symlinks updated to point to $(basename "$LATEST_FULL_PATH")"
  fi
  return 0
}

# -------------------------
# Cleanup
# -------------------------
do_clean() {
  log_info "clean: removing build directories"
  rm -rf build-posix build-fpga 2>/dev/null || true
  if [ -d ggml-tsi-kernel ]; then
    rm -rf ggml-tsi-kernel/fpga-kernel/build-fpga 2>/dev/null || true
    rm -rf ggml-tsi-kernel/posix-kernel/build-posix 2>/dev/null || true
  fi
  return 0
}

do_clean_all() {
  do_clean || return 1
  if [ -d ggml-tsi-kernel/blob-creation ]; then
    log_info "clean-all: removing python venv blob-creation"
    rm -rf ggml-tsi-kernel/blob-creation || true
  fi
  return 0
}

main() {
  set -o pipefail

  local arch; arch="$(select_arch)" || return $?
  parse_args "$@"

  # Cleanup-only mode: do cleanup and exit early
  if [ "${DO_CLEAN_ALL}" -eq 1 ]; then
    do_clean_all
    return 0
  fi
  if [ "${DO_CLEAN}" -eq 1 ]; then
    do_clean
    return 0
  fi

  resolve_paths "$arch" || return $?

  # First run auto-inits; later optional unless git-submodule-pull is provided
  ensure_submodules "${GIT_SUBMODULE_PULL}" || return 1

  # Blob/venv work happens in ggml-tsi-kernel only when requested
  local need_python=0
  if [ "${OVERWRITE_VENV}" -eq 1 ] || [ "${DO_BLOB_FPGA}" -eq 1 ] || [ "${DO_BLOB_POSIX}" -eq 1 ]; then
    need_python=1
  fi

  if [ "${need_python}" -eq 1 ] || [ "${DO_BLOB_FPGA}" -eq 1 ] || [ "${DO_BLOB_POSIX}" -eq 1 ]; then
    cd ggml-tsi-kernel/ || die "ggml-tsi-kernel not found (submodule). Try git-submodule-pull."

    setup_toolchain || return 1

    # Only set up python env when requested (overwrite or blob build)
    if [ "${need_python}" -eq 1 ]; then
      setup_python || return 1
    fi

    # Build blobs ONLY when explicitly requested
    if [ "${DO_BLOB_FPGA}" -eq 1 ]; then
      cd fpga-kernel || return 1
      build_fpga_blobs || return 1
      cd .. || return 1
    fi

    if [ "${DO_BLOB_POSIX}" -eq 1 ]; then
      cd posix-kernel || return 1
      build_posix_blobs || return 1
      cd .. || return 1
    fi

    cd .. || return 1
  else
    setup_toolchain || return 1
  fi

  # Build selection in llama.cpp root
  if [ "${DO_BUILD_POSIX}" -eq 1 ]; then
    build_posix || return 1
    wrap_glibc_bins || return 1
  fi

  if [ "${DO_BUILD_FPGA}" -eq 1 ]; then
    build_fpga || return 1
  fi

  if [ "${DO_PACKAGE_FPGA}" -eq 1 ]; then
    bundle_fpga || return 1
  fi

  return 0
}

# Ensure cleanup runs even on errors; do not leak trap in sourced shell
if [ "$__TSI_SOURCED" -eq 1 ]; then
  trap cleanup RETURN
else
  trap cleanup EXIT
fi

main "$@"; __rc=$?

if [ "$__TSI_SOURCED" -eq 1 ]; then
  return "$__rc"
else
  exit "$__rc"
fi
