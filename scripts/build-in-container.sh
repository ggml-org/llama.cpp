#!/usr/bin/env bash
set -euo pipefail

# Reproducible containerized build for llama.cpp using Fedora toolchain
#
# Defaults can be overridden via environment variables:
#   ENGINE    : container runtime to use (podman|docker). Default: prefer podman, else docker
#   IMAGE     : base image. Default: docker.io/library/fedora:41
#   BUILD_DIR : CMake build dir inside project. Default: build-container
#   BUILD_TYPE: CMake build type. Default: Release
#   JOBS      : parallel build jobs. Default: nproc
#   CMAKE_ARGS: extra CMake args, e.g. "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86"
#
# Usage examples:
#   scripts/build-in-container.sh
#   IMAGE=fedora:41 BUILD_TYPE=Debug scripts/build-in-container.sh
#   CMAKE_ARGS='-DGGML_CUDA=ON' scripts/build-in-container.sh
#   ENGINE=docker scripts/build-in-container.sh

echo "[build-in-container] starting"

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

# selinux-friendly volume flag for podman; plain for docker
vol_suffix=""
if [[ "$engine" == "podman" ]]; then
  vol_suffix=":Z"
fi

proj_root=$(pwd)

echo "[build-in-container] engine=$engine image=$image build_dir=$build_dir build_type=$build_type jobs=$jobs"

# GPU passthrough (docker) when building CUDA
gpu_args=()
if [[ "$engine" == "docker" ]]; then
  if [[ "${CMAKE_ARGS:-}" == *"-DGGML_CUDA=ON"* ]]; then
    gpu_args+=("--gpus" "all" "-e" "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}" "-e" "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}")
  fi
fi

"$engine" run --rm "${gpu_args[@]}" \
  -v "$proj_root:/src${vol_suffix}" \
  -w /src \
  "$image" \
  bash -lc "\
    set -euo pipefail; \
    echo '[container] installing toolchain...'; \
    if command -v dnf >/dev/null 2>&1; then \
      dnf -y install --setopt=install_weak_deps=False gcc-c++ cmake make libcurl-devel git >/dev/null; \
    elif command -v apt-get >/dev/null 2>&1; then \
      export DEBIAN_FRONTEND=noninteractive; \
      apt-get update -qq >/dev/null; \
      apt-get install -y -qq build-essential cmake make git libcurl4-openssl-dev >/dev/null; \
    else \
      echo 'Unsupported base image: no dnf or apt-get'; exit 1; \
    fi; \
    echo '[container] configuring CMake...'; \
    cmake -S . -B '$build_dir' -DCMAKE_BUILD_TYPE='$build_type' ${CMAKE_ARGS:-}; \
    echo '[container] building...'; \
    cmake --build '$build_dir' -j '$jobs'; \
    echo '[container] done. binaries in $build_dir/bin' \
  "

echo "[build-in-container] finished. See $build_dir/bin for outputs."
