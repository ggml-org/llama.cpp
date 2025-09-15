#!/usr/bin/env bash
set -euo pipefail

# Run llama.cpp tests inside a container (podman or docker)
#
# Environment variables:
#   ENGINE        : container runtime (podman|docker). Default: prefer podman, else docker
#   IMAGE         : base image. Default: docker.io/library/fedora:41
#   BUILD_DIR     : CMake build dir. Default: build-container
#   BUILD_TYPE    : CMake build type. Default: Release
#   JOBS          : parallel jobs for build/ctest. Default: nproc
#   CMAKE_ARGS    : extra cmake args (for (re)configure if needed). Example: "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86;89"
#   BUILD_IF_NEEDED: If build dir missing, configure+build first (1|0). Default: 1
#   CTEST_ARGS    : extra ctest args. Example: "-R test-tokenizer -VV"
#   CTEST_LABEL   : ctest label filter, e.g. "main" or "model". Empty = all tests. Default: main
#   CTEST_EXCLUDE : ctest exclude regex passed via -E
#   LLAMACPP_TEST_MODELFILE : path to a gguf model for tests labeled "model" (optional)
#
# Usage examples:
#   scripts/test-in-container.sh                   # run label=main tests in container
#   CTEST_LABEL=                                   # run all tests
#   CMAKE_ARGS='-DGGML_CUDA=ON' scripts/test-in-container.sh
#   ENGINE=docker IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 CMAKE_ARGS='-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86;89' scripts/test-in-container.sh

echo "[test-in-container] starting"

# choose engine
if [[ -n ${ENGINE:-} ]]; then
  engine="$ENGINE"
else
  if command -v podman >/dev/null 2>&1; then
    engine=podman
  elif command -v docker >/dev/null 2>&1; then
    engine=docker
  else
    echo "Error: neither podman nor docker found in PATH" >&2
    exit 1
  fi
fi

image="${IMAGE:-docker.io/library/fedora:41}"
build_dir="${BUILD_DIR:-build-container}"
build_type="${BUILD_TYPE:-Release}"
jobs="${JOBS:-}"
if [[ -z "$jobs" ]]; then
  if command -v nproc >/dev/null 2>&1; then jobs=$(nproc); else jobs=8; fi
fi

build_if_needed="${BUILD_IF_NEEDED:-1}"
ctest_label_default="main"
ctest_label="${CTEST_LABEL-${ctest_label_default}}"

# selinux-friendly volume flag for podman; plain for docker
vol_suffix=""
if [[ "$engine" == "podman" ]]; then
  vol_suffix=":Z"
fi

proj_root=$(pwd)

echo "[test-in-container] engine=$engine image=$image build_dir=$build_dir build_type=$build_type jobs=$jobs label=${ctest_label:-all}"

# GPU passthrough (docker) when CUDA is requested via CMAKE_ARGS
gpu_args=()
if [[ "$engine" == "docker" ]]; then
  if [[ "${CMAKE_ARGS:-}" == *"-DGGML_CUDA=ON"* ]]; then
    nvvis="${NVIDIA_VISIBLE_DEVICES:-all}"
    if [[ "$nvvis" != "all" ]]; then
      gpu_args+=("--gpus" "device=${nvvis}")
    else
      gpu_args+=("--gpus" "all")
    fi
    gpu_args+=("-e" "NVIDIA_VISIBLE_DEVICES=${nvvis}" "-e" "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}")
  fi
fi

# propagate selected envs
env_args=()
[[ -n ${CMAKE_ARGS:-} ]] && env_args+=("-e" "CMAKE_ARGS=${CMAKE_ARGS}")
[[ -n ${CTEST_ARGS:-} ]] && env_args+=("-e" "CTEST_ARGS=${CTEST_ARGS}")
[[ -n ${CTEST_LABEL:-} ]] && env_args+=("-e" "CTEST_LABEL=${CTEST_LABEL}")
[[ -n ${CTEST_EXCLUDE:-} ]] && env_args+=("-e" "CTEST_EXCLUDE=${CTEST_EXCLUDE}")
[[ -n ${LLAMACPP_TEST_MODELFILE:-} ]] && env_args+=("-e" "LLAMACPP_TEST_MODELFILE=${LLAMACPP_TEST_MODELFILE}")

"$engine" run --rm "${gpu_args[@]}" "${env_args[@]}" \
  -v "$proj_root:/src${vol_suffix}" \
  -v "$proj_root/.ccache:/src/.ccache${vol_suffix}" \
  -w /src \
  "$image" \
  bash -lc "\
    set -euo pipefail; \
    echo '[container] installing test deps...'; \
    if command -v dnf >/dev/null 2>&1; then \
      dnf -y install --setopt=install_weak_deps=False gcc-c++ cmake make git git-lfs libcurl-devel ccache >/dev/null; \
      git lfs install --system >/dev/null 2>&1 || true; \
    elif command -v apt-get >/dev/null 2>&1; then \
      export DEBIAN_FRONTEND=noninteractive; \
      apt-get update -qq >/dev/null; \
      apt-get install -y -qq build-essential cmake make git git-lfs libcurl4-openssl-dev ccache >/dev/null; \
      git lfs install --system >/dev/null 2>&1 || true; \
    else \
      echo 'Unsupported base image: no dnf or apt-get'; exit 1; \
    fi; \
    git config --global --add safe.directory /src || true; \
    export CCACHE_DIR=/src/.ccache; mkdir -p "\$CCACHE_DIR"; \
    \
    if [[ ! -f '$build_dir/CTestTestfile.cmake' ]]; then \
      if [[ '${build_if_needed}' == '1' ]]; then \
        echo '[container] configuring (not found)...'; \
        extra=(); if [[ -n \${CMAKE_ARGS:-} ]]; then read -r -a extra <<< "\$CMAKE_ARGS"; fi; \
        cmake -S . -B '$build_dir' -DCMAKE_BUILD_TYPE='$build_type' "\${extra[@]}"; \
        echo '[container] building (tests)...'; \
        cmake --build '$build_dir' -j '$jobs'; \
      else \
        echo 'Error: no build dir found and BUILD_IF_NEEDED=0'; exit 1; \
      fi; \
    fi; \
    echo '[container] running ctest...'; \
    cd '$build_dir'; \
    label_arg=(); if [[ -n \${CTEST_LABEL:-${ctest_label_default}} ]]; then label_arg+=( -L "\${CTEST_LABEL:-${ctest_label_default}}" ); fi; \
    exclude_arg=(); if [[ -n \${CTEST_EXCLUDE:-} ]]; then exclude_arg+=( -E "\$CTEST_EXCLUDE" ); fi; \
    extra_ctest=(); if [[ -n \${CTEST_ARGS:-} ]]; then read -r -a extra_ctest <<< "\$CTEST_ARGS"; fi; \
    ctest --output-on-failure -j '$jobs' "\${label_arg[@]}" "\${exclude_arg[@]}" "\${extra_ctest[@]}"; \
    echo '[container] tests done.' \
  "

echo "[test-in-container] finished. See $build_dir/Testing for reports."
