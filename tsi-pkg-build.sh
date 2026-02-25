#!/usr/bin/env bash
# Safe to source: does not leak -e/-u into the interactive shell.
#
# Build modes:
#   Release build:
#     source tsi-pkg-build.sh release
#
#   Debug (detailed performance) build:
#     source tsi-pkg-build.sh debug
#
#   Default (developer build):
#     source tsi-pkg-build.sh

log_error(){ echo "ERROR: $*" >&2; }
log_info(){ echo "INFO:  $*"; }

# detect sourced vs executed
__TSI_SOURCED=0
(return 0 2>/dev/null) && __TSI_SOURCED=1 || true
__TSI_OLD_SET="$(set +o)"

run() { "$@"; local rc=$?; [ $rc -eq 0 ] || log_error "cmd failed ($rc): $*"; return $rc; }
absdir() { (cd "$1" 2>/dev/null && pwd); }

die() {
  log_error "$*"
  if [ "$__TSI_SOURCED" -eq 1 ]; then return 1; else exit 1; fi
}

tolower(){ echo "$1" | tr '[:upper:]' '[:lower:]'; }

select_arch() {
  local m; m="$(uname -m)"
  case "$m" in
    x86_64|amd64) echo "x86_64" ;;
    aarch64|arm64) echo "aarch64" ;;
    *) log_error "Unsupported host arch from uname -m: $m"; return 2 ;;
  esac
}

parse_args() {
  BUILD_TYPE="${1:-}"
  MLIR_COMPILER_DIR_IN="${2:-${MLIR_COMPILER_DIR:-}}"
  TOOLBOX_DIR_IN="${3:-${TOOLBOX_DIR:-}}"

  ENABLE_COVERAGE_FLAG=""
  local a
  for a in "$@"; do
    if [ "$(tolower "$a")" = "enable_coverage" ]; then
      ENABLE_COVERAGE_FLAG="-DENABLE_COVERAGE=ON"
      log_info "enable_coverage detected"
      break
    fi
  done
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

setup_python() {
  log_info "creating python virtual env"
  __OLD_VIRTUAL_ENV="${VIRTUAL_ENV:-}"
  run /proj/local/Python-3.11.12/bin/python3 -m venv blob-creation || return 1
  # shellcheck disable=SC1091
  run bash -c 'source blob-creation/bin/activate && python -V >/dev/null' || return 1
  # activate for current shell
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
}

build_kernels() {
  log_info "creating fpga kernel"
  run cmake -B build-fpga -DTOOLBOX_DIR="${TOOLBOX_DIR}" -DCOMPILER_INSTALL_DIR="${MLIR_COMPILER_DIR}" || return 1
  run ./create-all-kernels.sh || return 1

  log_info "creating posix kernel"
  cd ../posix-kernel/ || return 1
  run ./create-all-kernels.sh || return 1
}

build_posix() {
  log_info "building llama.cpp/ggml for posix"
  local bt; bt="$(tolower "${BUILD_TYPE}")"
  local common="-DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=posix -DGGML_NATIVE=ON -DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF"
  local cflags_base="-DGGML_TARGET_POSIX -DGGML_TSAVORITE -mno-amx-tile -mno-amx-int8 -mno-amx-bf16 -mno-avx512bf16 -mno-avxvnni"
  local perf="-DGGML_PERF"
  [ "$bt" = "release" ] && perf="-DGGML_PERF_RELEASE"
  [ "$bt" = "debug" ]   && perf="-DGGML_PERF_DETAIL"

  run cmake -B build-posix ${common} \
    -DCMAKE_C_COMPILER="${CC}" -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_C_FLAGS="${perf} ${cflags_base}" \
    -DCMAKE_CXX_FLAGS="${perf} ${cflags_base}" \
    ${ENABLE_COVERAGE_FLAG} || return 1

  run cmake --build build-posix --config Release || return 1
}

wrap_glibc_bins() {
  log_info "fixing GLIBC compatibility for TSI binaries"

  mv build-posix/bin/simple-backend-tsi build-posix/bin/simple-backend-tsi-original || return 1
  cat > build-posix/bin/simple-backend-tsi <<'EOL'
#!/bin/bash
export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:$LD_LIBRARY_PATH"
exec "$(dirname "$0")/simple-backend-tsi-original" "$@"
EOL
  chmod +x build-posix/bin/simple-backend-tsi || return 1

  mv build-posix/bin/llama-cli build-posix/bin/llama-cli-original || return 1
  cat > build-posix/bin/llama-cli <<'EOL'
#!/bin/bash
export LD_LIBRARY_PATH="/proj/local/gcc-13.3.0/lib64:$LD_LIBRARY_PATH"
exec "$(dirname "$0")/llama-cli-original" "$@"
EOL
  chmod +x build-posix/bin/llama-cli || return 1
}

build_fpga() {
  log_info "building llama.cpp/ggml for fpga"
  local bt; bt="$(tolower "${BUILD_TYPE}")"
  local ARM_TOOLCHAIN_FILE="${TOOLBOX_DIR}/lib/cmake/toolchains/arm.cmake"
  local perf="-DGGML_PERF"
  [ "$bt" = "release" ] && perf="-DGGML_PERF_RELEASE"
  [ "$bt" = "debug" ]   && perf="-DGGML_PERF_DETAIL"

  run cmake -B build-fpga \
    -DCMAKE_TOOLCHAIN_FILE="${ARM_TOOLCHAIN_FILE}" \
    -DGGML_TSAVORITE=ON -DGGML_TSAVORITE_TARGET=fpga -DLLAMA_CURL=OFF \
    -DCMAKE_C_FLAGS="${perf} -DGGML_TSAVORITE" \
    -DCMAKE_CXX_FLAGS="${perf} -DGGML_TSAVORITE" \
    ${ENABLE_COVERAGE_FLAG} || return 1

  run cmake --build build-fpga --config Release || return 1
}

bundle_fpga() {
  log_info "creating tar bundle for fpga"
  local TSI_GGML_VERSION=0.2.5
  local TSI_GGML_BUNDLE_INSTALL_DIR=tsi-ggml
  local GGML_TSI_INSTALL_DIR=ggml-tsi-kernel
  local TSI_GGML_RELEASE_DIR=/proj/rel/sw/ggml
  local TSI_BLOB_INSTALL_DIR
  TSI_BLOB_INSTALL_DIR="$(pwd)/${GGML_TSI_INSTALL_DIR}/fpga-kernel/build-fpga"

  mkdir -p "${TSI_GGML_BUNDLE_INSTALL_DIR}"
  rm -f "${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh"

  cat > "./${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh" <<EOL
#!/bin/bash
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:\$(pwd)

tsi_kernels=("add" "sub" "mult" "div" "abs" "inv" "neg" "sin" "sqrt" "sqr" "sigmoid" "silu" "rms_norm" "swiglu" \
"add_16" "sub_16" "mult_16" "div_16" "abs_16" "inv_16" "neg_16" "sin_16" "sqrt_16" "sqr_16" "sigmoid_16" "silu_16" "rms_norm_16" "swiglu_16")

for kernel in "\${tsi_kernels[@]}"; do
  mkdir -p ${TSI_BLOB_INSTALL_DIR}/txe_\$kernel
  cp blobs ${TSI_BLOB_INSTALL_DIR}/txe_\$kernel/ -r
done
EOL
  chmod +x "${TSI_GGML_BUNDLE_INSTALL_DIR}/ggml.sh" || return 1

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
}

main() {
  set -o pipefail
  local arch; arch="$(select_arch)" || return $?
  parse_args "$@"
  resolve_paths "$arch" || return $?
  run git submodule update --recursive --init || return 1

  cd ggml-tsi-kernel/ || return 1
  setup_toolchain || return 1
  setup_python || return 1

  cd fpga-kernel || return 1
  build_kernels || return 1

  cd ../../ || return 1
  build_posix || return 1
  wrap_glibc_bins || return 1
  build_fpga || return 1
  bundle_fpga || return 1
  return 0
}

main "$@"; __rc=$?

# restore caller shell behavior (prevents TAB/prompt issues)
eval "${__TSI_OLD_SET}" >/dev/null 2>&1 || true
stty sane 2>/dev/null || true

if [ "$__TSI_SOURCED" -eq 1 ]; then
  return "$__rc"
else
  exit "$__rc"
fi


# Exit venv (or restore previous venv if one was active)
if [ -n "${__OLD_VIRTUAL_ENV}" ]; then
  # shellcheck disable=SC1091
  source "${__OLD_VIRTUAL_ENV}/bin/activate" || true
else
  deactivate || true
fi
