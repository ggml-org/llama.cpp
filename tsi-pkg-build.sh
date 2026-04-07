#!/usr/bin/env bash
# ==============================================================================
# tsi-pkg-build.sh (source-safe)
#
# USAGE (source is recommended)
# ============================
#
# source tsi-pkg-build.sh [build-mode] [flags...] [MLIR_COMPILER_DIR] [TOOLBOX_DIR]
#
# Build modes (optional):
#   release
#   debug
#   debug-tmu
#   debug-tmu-detail
#
# - release        : GGML_PERF_RELEASE
# - debug          : POSIX => GGML_PERF_DETAIL
#                   FPGA  => GGML_PERF  (GGML_PERF_DETAIL disabled to avoid expensive file logging)
# - debug-tmu      : GGML_PERF_DETAIL + TMU_DEBUG
# - debug-tmu-detail : GGML_PERF_DETAIL + TMU_DEBUG + TMU_DEBUG_VALIDATE
#
# Submodules:
# - First run in a fresh repo checkout: auto "git submodule update --init --recursive"
# - Later runs: submodule update is OPTIONAL unless pass:
#     git-submodule-pull
# - If ggml-tsi-kernel is missing, script forces submodule init even without the flag.
#
# Blob build (OFF by default):
#   build-fpga-blobs  : build blobs in ggml-tsi-kernel/fpga-kernel only
#   build-posix-blobs : build blobs in ggml-tsi-kernel/posix-kernel only
#   build-all-blobs   : build blobs for both fpga+posix kernels
#
# Auto blob safeguards (ON by default):
# - If deleted ggml-tsi-kernel (rm -rf) or host objects are missing:
#   * POSIX build auto-builds POSIX blobs if required for link
#   * FPGA build auto-builds FPGA blobs if required for link
# Disable both with:
#   no-auto-blobs
#
# Python virtual env (only used for blob generation):
#   overwrite-venv : delete blob-creation venv and recreate it (installs deps)
# NOTE: this alone does NOT build blobs unless blob flag is also set.
# git-submodule-pull : ALSO forces overwrite-venv AND build-all-blobs (as requested)
#
# Build selection:
# Default (no build-selection flags): build-posix + build-fpga + package
#
#   build-posix
#     * Build POSIX ggml/llama.cpp with TMU + TVU enabled
#     * Output directory: ./build-posix
#
#   build-posix-tmu-only
#     * Build POSIX ggml/llama.cpp with TMU enabled and TVU disabled
#     * Output directory: ./build-posix-tmu-only
#
#   build-posix-tmu-disable
#     * Build POSIX ggml/llama.cpp with TVU enabled and TMU disabled
#     * Output directory: ./build-posix-tmu-disable
#
#   build-fpga
#     * Build FPGA ggml/llama.cpp with TMU + TVU enabled
#     * Output directory: ./build-fpga
#
#   build-fpga-tmu-only
#     * Build FPGA ggml/llama.cpp with TMU enabled and TVU disabled
#     * Output directory: ./build-fpga-tmu-only
#
#   build-fpga-tmu-disable
#     * Build FPGA ggml/llama.cpp with TVU enabled and TMU disabled
#     * Output directory: ./build-fpga-tmu-disable
#
#   package
#     * Package FPGA bundle (requires an FPGA build dir already built)
#
# Incremental build:
#   incremental : do not rm -rf build dirs (both llama.cpp + kernels)
#
# Cleanup:
#   clean     : rm -rf build-* (llama.cpp) and kernel build dirs in ggml-tsi-kernel
#   clean-all : clean + remove python venv blob-creation
#
# Coverage:
#   enable_coverage : adds -DENABLE_COVERAGE=ON
#
# Help:
#   help  | -h | --help | -help
#
# ==============================================================================
#
# EXAMPLES
# ========
#
# 1) Default (posix + fpga + package) with default build-type (debug):
#    source tsi-pkg-build.sh
#
# 2) POSIX only:
#    source tsi-pkg-build.sh debug build-posix
#
# 3) POSIX TMU-only:
#    source tsi-pkg-build.sh debug build-posix-tmu-only
#
# 4) POSIX TMU disabled (TVU-only):
#    source tsi-pkg-build.sh debug build-posix-tmu-disable
#
# 5) FPGA only (TMU+TVU):
#    source tsi-pkg-build.sh debug build-fpga
#
# 6) FPGA TMU-only:
#    source tsi-pkg-build.sh debug build-fpga-tmu-only
#
# 7) FPGA TMU disabled (TVU-only):
#    source tsi-pkg-build.sh debug build-fpga-tmu-disable
#
# 8) Debug TMU:
#    source tsi-pkg-build.sh debug-tmu build-fpga
#
# 9) Debug TMU detail (adds TMU_DEBUG_VALIDATE):
#    source tsi-pkg-build.sh debug-tmu-detail build-posix build-fpga
#
# 10) Build blobs explicitly:
#    source tsi-pkg-build.sh build-all-blobs
#    source tsi-pkg-build.sh build-fpga-blobs
#    source tsi-pkg-build.sh build-posix-blobs
#
# 11) Incremental builds (do not delete build dirs):
#    source tsi-pkg-build.sh incremental build-posix build-fpga
#
# 12) Provide explicit paths:
#    source tsi-pkg-build.sh debug build-fpga /path/to/compiler /path/to/toolbox/install-fpga
#
# ==============================================================================

log_error(){ echo "ERROR: $*" >&2; }
log_info(){ echo "INFO: $*"; }

__TSI_SOURCED=0
(return 0 2>/dev/null) && __TSI_SOURCED=1

__TSI_OLD_SET="$(set +o)"
__TSI_SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"

# --- VENV TRACKING (FIX) ---
# If script activates blob-creation venv, restore previous env when sourced.
__OLD_VIRTUAL_ENV=""
__TSI_CHANGED_VENV=0

run() {
  "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    log_error "cmd failed ($rc): $*"
    return $rc
  fi
  return 0
}

absdir() { (cd "$1" 2>/dev/null && pwd); }
tolower(){ echo "$1" | tr '[:upper:]' '[:lower:]'; }

die() {
  log_error "$*"
  if [ "$__TSI_SOURCED" -eq 1 ]; then return 1; else exit 1; fi
}

cleanup() {
  # --- VENV RESTORE (FIX) ---
  if [ "${__TSI_CHANGED_VENV:-0}" -eq 1 ]; then
    if declare -F deactivate >/dev/null 2>&1; then
      deactivate >/dev/null 2>&1 || true
    else
      unset VIRTUAL_ENV 2>/dev/null || true
    fi

    # If user was in a previous venv before we activated blob-creation, restore it.
    if [ -n "${__OLD_VIRTUAL_ENV}" ] && [ -f "${__OLD_VIRTUAL_ENV}/bin/activate" ]; then
      # shellcheck disable=SC1090
      source "${__OLD_VIRTUAL_ENV}/bin/activate" >/dev/null 2>&1 || true
    fi
  fi

  # restore caller shell behavior
  eval "${__TSI_OLD_SET}" >/dev/null 2>&1 || true
  stty sane 2>/dev/null || true
  trap - RETURN EXIT 2>/dev/null || true
}

usage() {
  local p="${__TSI_SCRIPT_PATH}"
  if [ -r "$p" ]; then
    sed -n '1,320p' "$p" 2>/dev/null | sed 's/^# \{0,1\}//'
    return 0
  fi
  # fallback (should rarely happen)
  cat <<'EOF'
tsi-pkg-build.sh: unable to read script header for help output.
Try:  cat tsi-pkg-build.sh | sed -n '1,320p'
EOF
  return 0
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
# Submodule logic (robust)
# -------------------------
MARKER_FILE=".tsi_submodules_initialized"
SUBMODULE_DIR="ggml-tsi-kernel"

submodule_self_heal_if_needed() {
  # If the path exists but is not a proper submodule checkout and is non-empty, wipe it.
  if [ -e "${SUBMODULE_DIR}" ]; then
    if [ ! -d "${SUBMODULE_DIR}/.git" ] && [ -n "$(ls -A "${SUBMODULE_DIR}" 2>/dev/null || true)" ]; then
      log_info "${SUBMODULE_DIR} exists and is non-empty (stale). Cleaning to allow submodule clone."
      run git submodule deinit -f -- "${SUBMODULE_DIR}" || true
      run rm -rf "${SUBMODULE_DIR}" || true
      run rm -rf ".git/modules/${SUBMODULE_DIR}" || true
      return 1
    fi
  fi
  return 0
}

ensure_submodules() {
  local want_update="$1" # 0/1 from user flag
  local force=0

  # If submodule directory missing, ALWAYS force init.
  if [ ! -d "${SUBMODULE_DIR}" ]; then
    log_info "${SUBMODULE_DIR} missing; forcing submodule init"
    force=1
  fi
  # If marker missing, treat as first-time repo.
  if [ ! -f "${MARKER_FILE}" ]; then
    force=1
  fi
  # User asked explicitly.
  if [ "${want_update}" -eq 1 ]; then
    force=1
  fi

  if [ "${force}" -eq 1 ]; then
    submodule_self_heal_if_needed || true
    run git submodule update --init --recursive || die "git submodule update failed"
    : > "${MARKER_FILE}" || true
  else
    log_info "Skipping git submodule update (already initialized). Use git-submodule-pull to refresh."
  fi

  [ -d "${SUBMODULE_DIR}" ] || die "${SUBMODULE_DIR} still missing after submodule init"
  return 0
}

# -------------------------
# Args/flags
# -------------------------
parse_args() {
  SHOW_HELP=0

  BUILD_TYPE=""
  MLIR_COMPILER_DIR_IN="${MLIR_COMPILER_DIR:-}"
  TOOLBOX_DIR_IN="${TOOLBOX_DIR:-}"
  ENABLE_COVERAGE_FLAG=""

  # submodules
  GIT_SUBMODULE_PULL=0

  # blobs
  DO_BLOB_FPGA=0
  DO_BLOB_POSIX=0

  # python venv
  OVERWRITE_VENV=0

  # build selection (default: posix+fpga+package)
  DO_BUILD_POSIX=1
  DO_BUILD_POSIX_TMU_ONLY=0
  DO_BUILD_POSIX_TMU_DISABLE=0

  DO_BUILD_FPGA=1
  DO_BUILD_FPGA_TMU_ONLY=0
  DO_BUILD_FPGA_TMU_DISABLE=0

  DO_PACKAGE_FPGA=1
  __USER_BUILD_SELECT=0

  # cleanup
  DO_CLEAN=0
  DO_CLEAN_ALL=0

  # cleaning build dirs before build (default ON)
  DO_CLEAN_BUILD_DIRS=1
  INCREMENTAL=0

  # auto blobs (default ON; applies to POSIX+FPGA host object link safety)
  AUTO_BLOBS=1

  # packaging selection
  PACKAGE_FPGA_BUILD_DIR=""

  local a
  for a in "$@"; do
    case "$(tolower "$a")" in
      help|-h|--help|-help)
        SHOW_HELP=1
        return 0
        ;;
      release|debug|debug-tmu|debug-tmu-detail)
        [ -z "${BUILD_TYPE}" ] && BUILD_TYPE="$a"
        ;;
      enable_coverage)
        ENABLE_COVERAGE_FLAG="-DENABLE_COVERAGE=ON"
        log_info "enable_coverage detected"
        ;;
      git-submodule-pull)
        GIT_SUBMODULE_PULL=1
        log_info "git-submodule-pull detected"
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
      no-auto-blobs)
        AUTO_BLOBS=0
        log_info "no-auto-blobs detected"
        ;;
      incremental)
        INCREMENTAL=1
        DO_CLEAN_BUILD_DIRS=0
        log_info "incremental build selected (no rm -rf build dirs)"
        ;;
      build-posix|posix)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_POSIX_TMU_ONLY=0; DO_BUILD_POSIX_TMU_DISABLE=0
          DO_BUILD_FPGA=0; DO_BUILD_FPGA_TMU_ONLY=0; DO_BUILD_FPGA_TMU_DISABLE=0
          DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_BUILD_POSIX=1
        log_info "build-posix selected"
        ;;
      build-posix-tmu-only)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_POSIX_TMU_ONLY=0; DO_BUILD_POSIX_TMU_DISABLE=0
          DO_BUILD_FPGA=0; DO_BUILD_FPGA_TMU_ONLY=0; DO_BUILD_FPGA_TMU_DISABLE=0
          DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_BUILD_POSIX_TMU_ONLY=1
        log_info "build-posix-tmu-only selected"
        ;;
      build-posix-tmu-disable)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_POSIX_TMU_ONLY=0; DO_BUILD_POSIX_TMU_DISABLE=0
          DO_BUILD_FPGA=0; DO_BUILD_FPGA_TMU_ONLY=0; DO_BUILD_FPGA_TMU_DISABLE=0
          DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_BUILD_POSIX_TMU_DISABLE=1
        log_info "build-posix-tmu-disable selected"
        ;;
      build-fpga|fpga)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_POSIX_TMU_ONLY=0; DO_BUILD_POSIX_TMU_DISABLE=0
          DO_BUILD_FPGA=0; DO_BUILD_FPGA_TMU_ONLY=0; DO_BUILD_FPGA_TMU_DISABLE=0
          DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_BUILD_FPGA=1
        PACKAGE_FPGA_BUILD_DIR="build-fpga"
        log_info "build-fpga selected"
        ;;
      build-fpga-tmu-only)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_POSIX_TMU_ONLY=0; DO_BUILD_POSIX_TMU_DISABLE=0
          DO_BUILD_FPGA=0; DO_BUILD_FPGA_TMU_ONLY=0; DO_BUILD_FPGA_TMU_DISABLE=0
          DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_BUILD_FPGA_TMU_ONLY=1
        PACKAGE_FPGA_BUILD_DIR="build-fpga-tmu-only"
        log_info "build-fpga-tmu-only selected"
        ;;
      build-fpga-tmu-disable)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_POSIX_TMU_ONLY=0; DO_BUILD_POSIX_TMU_DISABLE=0
          DO_BUILD_FPGA=0; DO_BUILD_FPGA_TMU_ONLY=0; DO_BUILD_FPGA_TMU_DISABLE=0
          DO_PACKAGE_FPGA=0
          __USER_BUILD_SELECT=1
        fi
        DO_BUILD_FPGA_TMU_DISABLE=1
        PACKAGE_FPGA_BUILD_DIR="build-fpga-tmu-disable"
        log_info "build-fpga-tmu-disable selected"
        ;;
      package|bundle)
        if [ "$__USER_BUILD_SELECT" -eq 0 ]; then
          DO_BUILD_POSIX=0; DO_BUILD_POSIX_TMU_ONLY=0; DO_BUILD_POSIX_TMU_DISABLE=0
          DO_BUILD_FPGA=0; DO_BUILD_FPGA_TMU_ONLY=0; DO_BUILD_FPGA_TMU_DISABLE=0
          DO_PACKAGE_FPGA=0
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
        # positional paths
        if [ -z "${MLIR_COMPILER_DIR_IN}" ]; then
          MLIR_COMPILER_DIR_IN="$a"
        elif [ -z "${TOOLBOX_DIR_IN}" ]; then
          TOOLBOX_DIR_IN="$a"
        fi
        ;;
    esac
  done

  # git-submodule-pull ALSO deletes+recreates venv and builds all blobs.
  if [ "${GIT_SUBMODULE_PULL}" -eq 1 ]; then
    OVERWRITE_VENV=1
    DO_BLOB_FPGA=1
    DO_BLOB_POSIX=1
    log_info "git-submodule-pull => forcing overwrite-venv + build-all-blobs"
  fi

  # Default build type if none provided
  if [ -z "${BUILD_TYPE}" ]; then
    BUILD_TYPE="debug"
  fi

  return 0
}

resolve_paths() {
  local arch="$1"

  if [ -z "${MLIR_COMPILER_DIR_IN}" ]; then
    MLIR_SDK_VERSION="${MLIR_SDK_VERSION:-/proj/rel/sw/sdk-r.0.3.1/${arch}}"
    MLIR_COMPILER_DIR_IN="${MLIR_SDK_VERSION}/compiler"
  fi
  if [ -z "${TOOLBOX_DIR_IN}" ]; then
    MLIR_SDK_VERSION="${MLIR_SDK_VERSION:-$(dirname "${MLIR_COMPILER_DIR_IN}")}"
    TOOLBOX_DIR_IN="${MLIR_SDK_VERSION}/toolbox/build/install-fpga"
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
# Python venv (only when needed for blob generation)
# -------------------------
setup_python() {
  # Save the caller's current VIRTUAL_ENV (if any). This allows restore.
  __OLD_VIRTUAL_ENV="${VIRTUAL_ENV:-}"

  if [ "${OVERWRITE_VENV}" -eq 1 ] && [ -d "blob-creation" ]; then
    log_info "overwrite-venv: removing existing blob-creation venv"
    rm -rf blob-creation || return 1
  fi

  if [ -d "blob-creation" ] && [ -f "blob-creation/bin/activate" ]; then
    run bash -c 'source blob-creation/bin/activate && python -V >/dev/null' || return 1
    # shellcheck disable=SC1091
    source blob-creation/bin/activate || return 1
    [ "${VIRTUAL_ENV:-}" != "${__OLD_VIRTUAL_ENV:-}" ] && __TSI_CHANGED_VENV=1 || true
    return 0
  fi

  run /proj/local/Python-3.11.12/bin/python3 -m venv blob-creation || return 1
  run bash -c 'source blob-creation/bin/activate && python -V >/dev/null' || return 1
  # shellcheck disable=SC1091
  source blob-creation/bin/activate || return 1
  [ "${VIRTUAL_ENV:-}" != "${__OLD_VIRTUAL_ENV:-}" ] && __TSI_CHANGED_VENV=1 || true

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
# Blob presence + build helpers
# -------------------------
posix_host_objs_present() {
  [ -d "posix-kernel/build-posix" ] || return 1
  find "posix-kernel/build-posix" -name "host.o" -print -quit 2>/dev/null | grep -q . || return 1
  return 0
}

fpga_host_objs_present() {
  [ -d "fpga-kernel/build-fpga" ] || return 1
  find "fpga-kernel/build-fpga" -name "host.o" -print -quit 2>/dev/null | grep -q . || return 1
  return 0
}

build_fpga_blobs() {
  log_info "BLOB: building FPGA kernels/blobs"
  cd fpga-kernel || return 1
  run cmake -B build-fpga -DTOOLBOX_DIR="${TOOLBOX_DIR}" -DCOMPILER_INSTALL_DIR="${MLIR_COMPILER_DIR}" || return 1
  run ./create-all-kernels.sh || return 1
  cd .. || return 1
  return 0
}

build_posix_blobs() {
  log_info "BLOB: building POSIX kernels/blobs"
  cd posix-kernel || return 1
  run ./create-all-kernels.sh || return 1
  cd .. || return 1
  return 0
}

# -------------------------
# PERF/DEBUG defs
# -------------------------
compute_perf_and_debug_defs() {
  local target="$1" # posix|fpga
  local bt; bt="$(tolower "${BUILD_TYPE}")"

  PERF_DEF="-DGGML_PERF"
  DBG_DEFS=""

  if [ "$bt" = "release" ]; then
    PERF_DEF="-DGGML_PERF_RELEASE"
    DBG_DEFS=""
    return 0
  fi

  if [ "$bt" = "debug" ]; then
    if [ "$target" = "fpga" ]; then
      # FPGA debug: disable GGML_PERF_DETAIL (expensive file logging on FPGA)
      PERF_DEF="-DGGML_PERF"
    else
      PERF_DEF="-DGGML_PERF_DETAIL"
    fi
    DBG_DEFS=""
    return 0
  fi

  if [ "$bt" = "debug-tmu" ]; then
    PERF_DEF="-DGGML_PERF_DETAIL"
    DBG_DEFS="-DTMU_DEBUG"
    return 0
  fi

  if [ "$bt" = "debug-tmu-detail" ]; then
    PERF_DEF="-DGGML_PERF_DETAIL"
    DBG_DEFS="-DTMU_DEBUG -DTMU_DEBUG_VALIDATE"
    return 0
  fi

  return 0
}

# -------------------------
# POSIX build (clean rebuild by default)
# -------------------------
build_posix_impl() {
  local build_dir="$1"  # build-posix / build-posix-tmu-only / build-posix-tmu-disable
  local want_tmu="$2"   # 1/0
  local want_tvu="$3"   # 1/0

  log_info "building llama.cpp/ggml for posix (${build_dir})"
  if [ "${DO_CLEAN_BUILD_DIRS}" -eq 1 ]; then
    log_info "clean rebuild: rm -rf ./${build_dir}"
    rm -rf "${build_dir}" || return 1
  fi

  compute_perf_and_debug_defs "posix"

  local common="-DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DGGML_NATIVE=ON -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF"

  local supported=""
  [ "${want_tmu}" -eq 1 ] && supported="${supported} -DTMU_SUPPORTED"
  [ "${want_tvu}" -eq 1 ] && supported="${supported} -DTVU_SUPPORTED"

  local cflags_base="-DGGML_TARGET_POSIX -DGGML_TSAVORITE ${supported} -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"

  run cmake -B "${build_dir}" ${common} \
    -DCMAKE_C_COMPILER="${CC}" -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="${PERF_DEF} ${DBG_DEFS} ${cflags_base}" \
    -DCMAKE_CXX_FLAGS="${PERF_DEF} ${DBG_DEFS} ${cflags_base}" \
    ${ENABLE_COVERAGE_FLAG} || return 1

  run cmake --build "${build_dir}" --config Release || return 1
  return 0
}

build_posix() { build_posix_impl "build-posix" 1 1; }
build_posix_tmu_only() { build_posix_impl "build-posix-tmu-only" 1 0; }
build_posix_tmu_disable() { build_posix_impl "build-posix-tmu-disable" 0 1; }

wrap_glibc_bins() {
  local build_dir="$1"
  log_info "fixing GLIBC compatibility for TSI binaries (${build_dir})"

  if [ -f "${build_dir}/bin/simple-backend-tsi" ] && [ ! -f "${build_dir}/bin/simple-backend-tsi-original" ]; then
    mv "${build_dir}/bin/simple-backend-tsi" "${build_dir}/bin/simple-backend-tsi-original" || return 1
    cat > "${build_dir}/bin/simple-backend-tsi" <<'EOL'
#!/bin/bash
export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:$LD_LIBRARY_PATH"
exec "$(dirname "$0")/simple-backend-tsi-original" "$@"
EOL
    chmod +x "${build_dir}/bin/simple-backend-tsi" || return 1
  fi

  if [ -f "${build_dir}/bin/llama-cli" ] && [ ! -f "${build_dir}/bin/llama-cli-original" ]; then
    mv "${build_dir}/bin/llama-cli" "${build_dir}/bin/llama-cli-original" || return 1
    cat > "${build_dir}/bin/llama-cli" <<'EOL'
#!/bin/bash
export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:$LD_LIBRARY_PATH"
exec "$(dirname "$0")/llama-cli-original" "$@"
EOL
    chmod +x "${build_dir}/bin/llama-cli" || return 1
  fi

  return 0
}

# -------------------------
# FPGA build (clean rebuild by default)
# -------------------------
build_fpga_impl() {
  local build_dir="$1"  # build-fpga / build-fpga-tmu-only / build-fpga-tmu-disable
  local want_tmu="$2"   # 1/0
  local want_tvu="$3"   # 1/0

  log_info "building llama.cpp/ggml for fpga (${build_dir})"
  if [ "${DO_CLEAN_BUILD_DIRS}" -eq 1 ]; then
    log_info "clean rebuild: rm -rf ./${build_dir}"
    rm -rf "${build_dir}" || return 1
  fi

  compute_perf_and_debug_defs "fpga"

  local ARM_TOOLCHAIN_FILE="${TOOLBOX_DIR}/lib/cmake/toolchains/arm.cmake"

  local supported=""
  [ "${want_tmu}" -eq 1 ] && supported="${supported} -DTMU_SUPPORTED"
  [ "${want_tvu}" -eq 1 ] && supported="${supported} -DTVU_SUPPORTED"

  run cmake -B "${build_dir}" \
    -DCMAKE_TOOLCHAIN_FILE="${ARM_TOOLCHAIN_FILE}" \
    -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=fpga -DLLAMA_CURL=OFF \
    -DCMAKE_C_FLAGS="${PERF_DEF} ${DBG_DEFS} -DGGML_TSAVORITE ${supported}" \
    -DCMAKE_CXX_FLAGS="${PERF_DEF} ${DBG_DEFS} -DGGML_TSAVORITE ${supported}" \
    ${ENABLE_COVERAGE_FLAG} || return 1

  run cmake --build "${build_dir}" --config Release || return 1
  return 0
}

build_fpga() { build_fpga_impl "build-fpga" 1 1; }
build_fpga_tmu_only() { build_fpga_impl "build-fpga-tmu-only" 1 0; }
build_fpga_tmu_disable() { build_fpga_impl "build-fpga-tmu-disable" 0 1; }

choose_existing_fpga_build_dir_for_package() {
  # If user explicitly selected a package build dir, prefer it.
  if [ -n "${PACKAGE_FPGA_BUILD_DIR}" ] && [ -f "${PACKAGE_FPGA_BUILD_DIR}/bin/llama-cli" ]; then
    echo "${PACKAGE_FPGA_BUILD_DIR}"
    return 0
  fi

  # Otherwise, pick the first viable build dir in priority order.
  local d
  for d in build-fpga build-fpga-tmu-only build-fpga-tmu-disable; do
    if [ -f "${d}/bin/llama-cli" ]; then
      echo "${d}"
      return 0
    fi
  done

  # None found.
  echo ""
  return 0
}

bundle_fpga() {
  local build_dir="$1"

  log_info "creating tar bundle for fpga (${build_dir})"

  local TSI_GGML_VERSION=0.3.1
  local TSI_GGML_BUNDLE_INSTALL_DIR=tsi-ggml
  local GGML_TSI_INSTALL_DIR=ggml-tsi-kernel
  local TSI_GGML_RELEASE_DIR=/proj/rel/sw/ggml
  local TSI_BLOB_INSTALL_DIR

  TSI_BLOB_INSTALL_DIR="$(pwd)/${GGML_TSI_INSTALL_DIR}/fpga-kernel/build-fpga"

  [ -f "${build_dir}/bin/llama-cli" ] || die "package requested but ${build_dir}/bin/llama-cli not found. Run an FPGA build first."

  mkdir -p "${TSI_GGML_BUNDLE_INSTALL_DIR}"
  rm -f "${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh"

  cat > "./${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh" <<'EOL'
#!/bin/bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(pwd)
tsi_kernels=("add" "sub" "mult" "div" "abs" "inv" "neg" "sin" "sqrt" "sqr" "sigmoid" "silu" "rms_norm" "swiglu" \
"add_16" "sub_16" "mult_16" "div_16" "abs_16" "inv_16" "neg_16" "sin_16" "sqrt_16" "sqr_16" "sigmoid_16" "silu_16" "rms_norm_16" "swiglu_16" "mul_mat_tile_f32_k32" "mul_mat_tile_f32_k64" "mul_mat_tile_f32_k128")
for kernel in "${tsi_kernels[@]}"; do
  mkdir -p __TSI_BLOB_INSTALL_DIR__/txe_${kernel}
  cp blobs __TSI_BLOB_INSTALL_DIR__/txe_${kernel}/ -r
done
EOL

  sed -i "s|__TSI_BLOB_INSTALL_DIR__|${TSI_BLOB_INSTALL_DIR}|g" "./${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh"
  chmod +x "./${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh" || return 1

  cp "${GGML_TSI_INSTALL_DIR}/fpga/blobs" "${TSI_GGML_BUNDLE_INSTALL_DIR}/" -r || return 1
  cp "${build_dir}/bin/llama-cli" "${TSI_GGML_BUNDLE_INSTALL_DIR}/" || return 1
  cp "${build_dir}/bin/libggml"*.so "${TSI_GGML_BUNDLE_INSTALL_DIR}/" || return 1
  cp "${build_dir}/bin/libllama"*.so "${TSI_GGML_BUNDLE_INSTALL_DIR}/" || return 1
  cp "${build_dir}/bin/simple-backend-tsi" "${TSI_GGML_BUNDLE_INSTALL_DIR}/" || return 1

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
# Cleanup commands
# -------------------------
do_clean() {
  log_info "clean: removing build directories"
  rm -rf \
    build-posix build-posix-tmu-only build-posix-tmu-disable \
    build-fpga build-fpga-tmu-only build-fpga-tmu-disable 2>/dev/null || true

  if [ -d "${SUBMODULE_DIR}" ]; then
    rm -rf "${SUBMODULE_DIR}/fpga-kernel/build-fpga" 2>/dev/null || true
    rm -rf "${SUBMODULE_DIR}/posix-kernel/build-posix" 2>/dev/null || true
  fi
  return 0
}

do_clean_all() {
  do_clean || return 1
  if [ -d "${SUBMODULE_DIR}/blob-creation" ]; then
    log_info "clean-all: removing python venv blob-creation"
    rm -rf "${SUBMODULE_DIR}/blob-creation" || true
  fi
  return 0
}

main() {
  set -o pipefail

  local arch
  arch="$(select_arch)" || return $?

  parse_args "$@" || return $?
  if [ "${SHOW_HELP}" -eq 1 ]; then
    usage
    return 0
  fi

  if [ "${DO_CLEAN_ALL}" -eq 1 ]; then do_clean_all; return 0; fi
  if [ "${DO_CLEAN}" -eq 1 ]; then do_clean; return 0; fi

  resolve_paths "$arch" || return $?
  setup_toolchain || return 1
  ensure_submodules "${GIT_SUBMODULE_PULL}" || return 1

  local need_python=0
  if [ "${OVERWRITE_VENV}" -eq 1 ] || [ "${DO_BLOB_FPGA}" -eq 1 ] || [ "${DO_BLOB_POSIX}" -eq 1 ]; then
    need_python=1
  fi

  local auto_posix_blob=0
  local auto_fpga_blob=0

  if [ "${AUTO_BLOBS}" -eq 1 ]; then
    cd "${SUBMODULE_DIR}" || return 1

    if [ "${DO_BUILD_POSIX}" -eq 1 ] || [ "${DO_BUILD_POSIX_TMU_ONLY}" -eq 1 ] || [ "${DO_BUILD_POSIX_TMU_DISABLE}" -eq 1 ]; then
      if ! posix_host_objs_present; then
        auto_posix_blob=1
        log_info "POSIX host objects missing => auto-building POSIX blobs to avoid undefined _mlir_ciface_*_host"
        need_python=1
        DO_BLOB_POSIX=1
      fi
    fi

    if [ "${DO_BUILD_FPGA}" -eq 1 ] || [ "${DO_BUILD_FPGA_TMU_ONLY}" -eq 1 ] || [ "${DO_BUILD_FPGA_TMU_DISABLE}" -eq 1 ]; then
      if ! fpga_host_objs_present; then
        auto_fpga_blob=1
        log_info "FPGA host objects missing => auto-building FPGA blobs to avoid undefined _mlir_ciface_*_host"
        need_python=1
        DO_BLOB_FPGA=1
      fi
    fi

    cd .. || return 1
  fi

  if [ "${need_python}" -eq 1 ] && ( [ "${DO_BLOB_FPGA}" -eq 1 ] || [ "${DO_BLOB_POSIX}" -eq 1 ] ); then
    cd "${SUBMODULE_DIR}" || die "cannot enter ggml-tsi-kernel"
    setup_python || return 1
    [ "${DO_BLOB_FPGA}" -eq 1 ] && build_fpga_blobs || return 1
    [ "${DO_BLOB_POSIX}" -eq 1 ] && build_posix_blobs || return 1
    cd .. || return 1
  fi

  if [ "${DO_BUILD_POSIX}" -eq 1 ]; then
    build_posix || return 1
    wrap_glibc_bins "build-posix" || return 1
  fi

  if [ "${DO_BUILD_POSIX_TMU_ONLY}" -eq 1 ]; then
    build_posix_tmu_only || return 1
    wrap_glibc_bins "build-posix-tmu-only" || return 1
  fi

  if [ "${DO_BUILD_POSIX_TMU_DISABLE}" -eq 1 ]; then
    build_posix_tmu_disable || return 1
    wrap_glibc_bins "build-posix-tmu-disable" || return 1
  fi

  if [ "${DO_BUILD_FPGA}" -eq 1 ]; then
    build_fpga || return 1
    PACKAGE_FPGA_BUILD_DIR="${PACKAGE_FPGA_BUILD_DIR:-build-fpga}"
  fi

  if [ "${DO_BUILD_FPGA_TMU_ONLY}" -eq 1 ]; then
    build_fpga_tmu_only || return 1
    PACKAGE_FPGA_BUILD_DIR="${PACKAGE_FPGA_BUILD_DIR:-build-fpga-tmu-only}"
  fi

  if [ "${DO_BUILD_FPGA_TMU_DISABLE}" -eq 1 ]; then
    build_fpga_tmu_disable || return 1
    PACKAGE_FPGA_BUILD_DIR="${PACKAGE_FPGA_BUILD_DIR:-build-fpga-tmu-disable}"
  fi

  if [ "${DO_PACKAGE_FPGA}" -eq 1 ]; then
    local pkg_dir
    pkg_dir="$(choose_existing_fpga_build_dir_for_package)"
    [ -n "${pkg_dir}" ] || die "package requested but no FPGA build output found (expected build-fpga / build-fpga-tmu-only / build-fpga-tmu-disable)."
    bundle_fpga "${pkg_dir}" || return 1
  fi

  if [ "${auto_posix_blob}" -eq 1 ]; then
    log_info "NOTE: POSIX blobs were auto-built because they are required for linking _mlir_ciface_*_host."
  fi
  if [ "${auto_fpga_blob}" -eq 1 ]; then
    log_info "NOTE: FPGA blobs were auto-built because they are required for linking _mlir_ciface_*_host."
  fi

  return 0
}

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
