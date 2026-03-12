# Development and Testing

## Development

### Code Generation

The backend uses code generation from YAML configuration:

```bash
# Regenerate protocol code
cd ggml-virtgpu/
python regenerate_remoting.py
```

### Adding New Operations

1. Add function definition to `ggmlremoting_functions.yaml`
2. Regenerate code with `regenerate_remoting.py`
3. Implement guest-side forwarding in `virtgpu-forward-*.cpp`
4. Implement host-side handling in `backend-dispatched-*.cpp`

## Testing

This document provides instructions for building and testing the GGML-VirtGPU backend on macOS with containers.

### Prerequisites

The testing setup requires:

- macOS host system
- Container runtime with `libkrun` provider (podman machine)
- Access to development patchset for VirglRenderer

### Required Patchsets

The backend requires patches that are currently under review:

- **Virglrenderer APIR upstream PR**: https://gitlab.freedesktop.org/virgl/virglrenderer/-/merge_requests/1590 (for reference)
- **MacOS Virglrenderer (for krunkit)**: https://gitlab.freedesktop.org/kpouget/virglrenderer/-/tree/main-macos
- **Linux Virglrenderer (for krun)**: https://gitlab.freedesktop.org/kpouget/virglrenderer/-/tree/main-linux

### Build Instructions

#### 1. Build ggml-virtgpu-backend (Host-side, macOS)

```bash
# Build the backend that runs natively on macOS
mkdir llama.cpp
cd llama.cpp
git clone https://github.com/ggml-org/llama.cpp.git src
cd src

LLAMA_MAC_BUILD=$PWD/build/ggml-virtgpu-backend

cmake -S . -B $LLAMA_MAC_BUILD \
      -DGGML_NATIVE=OFF \
      -DLLAMA_CURL=ON \
      -DGGML_REMOTINGBACKEND=ONLY \
      -DGGML_METAL=ON

TARGETS="ggml-metal"
cmake --build $LLAMA_MAC_BUILD --parallel 8 --target $TARGETS

# Build additional tools for native benchmarking
EXTRA_TARGETS="llama-run llama-bench"
cmake --build $LLAMA_MAC_BUILD --parallel 8 --target $EXTRA_TARGETS
```

#### 2. Build virglrenderer (Host-side, macOS)

```bash
# Build virglrenderer with APIR support
mkdir virglrenderer
git clone https://gitlab.freedesktop.org/kpouget/virglrenderer -b main-macos src
cd src

VIRGL_BUILD_DIR=$PWD/build

# -Dvenus=true and VIRGL_ROUTE_VENUS_TO_APIR=1 route the APIR requests via the Venus backend, for easier testing without a patched hypervisor

meson setup $VIRGL_BUILD_DIR \
      -Dvenus=true \
      -Dapir=true

ninja -C $VIRGL_BUILD_DIR
```

#### 3. Build ggml-virtgpu (Guest-side, Linux)

Option A: Build from a script:

```bash
# Inside a Linux container
mkdir llama.cpp
git clone https://github.com/ggml-org/llama.cpp.git src
cd src

LLAMA_LINUX_BUILD=$PWD//build-virtgpu

cmake -S . -B $LLAMA_LINUX_BUILD \
      -DGGML_VIRTGPU=ON

ninja -C $LLAMA_LINUX_BUILD
```

Option B: Build container image with frontend:

```bash
cat << EOF > remoting.containerfile
FROM quay.io/fedora/fedora:43
USER 0

WORKDIR /app/remoting

ARG LLAMA_CPP_REPO="https://github.com/ggml-org/llama.cpp.git"
ARG LLAMA_CPP_VERSION="master"
ARG LLAMA_CPP_CMAKE_FLAGS="-DGGML_VIRTGPU=ON"
ARG LLAMA_CPP_CMAKE_BUILD_FLAGS="--parallel 4"

RUN dnf install -y git cmake gcc gcc-c++ libcurl-devel libdrm-devel

RUN git clone "\${LLAMA_CPP_REPO}" src \\
 && git -C src fetch origin \${LLAMA_CPP_VERSION} \\
 && git -C src reset --hard FETCH_HEAD

RUN mkdir -p build \\
 && cd src \\
 && set -o pipefail \\
 && cmake -S . -B ../build \${LLAMA_CPP_CMAKE_FLAGS} \\
 && cmake --build ../build/ \${LLAMA_CPP_CMAKE_BUILD_FLAGS}

ENTRYPOINT ["/app/remoting/src/build/bin/llama-server"]
EOF

mkdir -p empty_dir
podman build -f remoting.containerfile ./empty_dir -t localhost/llama-cpp.virtgpu
```

### Environment Setup

#### Set krunkit Environment Variables

```bash
# Define the base directories (adapt these paths to your system)
VIRGL_BUILD_DIR=$HOME/remoting/virglrenderer/build
LLAMA_MAC_BUILD=$HOME/remoting/llama.cpp/build-backend

# For krunkit to load the custom virglrenderer library
export DYLD_LIBRARY_PATH=$VIRGL_BUILD_DIR/src

# For Virglrenderer to load the ggml-remotingbackend library
export VIRGL_APIR_BACKEND_LIBRARY="$LLAMA_MAC_BUILD/bin/libggml-virtgpu-backend.dylib"

# For llama.cpp remotingbackend to load the ggml-metal backend
export APIR_LLAMA_CPP_GGML_LIBRARY_PATH="$LLAMA_MAC_BUILD/bin/libggml-metal.dylib"
export APIR_LLAMA_CPP_GGML_LIBRARY_REG=ggml_backend_metal_reg
```

#### Launch Container Environment

```bash
# Set container provider to libkrun
export CONTAINERS_MACHINE_PROVIDER=libkrun
podman machine start
```

#### Verify Environment

Confirm that krunkit is using the correct virglrenderer library:

```bash
lsof -c krunkit | grep virglrenderer
# Expected output:
# krunkit 50574 user  txt  REG  1,14  2273912  10849442 ($VIRGL_BUILD_DIR/src)/libvirglrenderer.1.dylib
```

### Running Tests

#### Launch Test Container

```bash
# Optional model caching
mkdir -p models
PODMAN_CACHE_ARGS="-v models:/models --user root:root --cgroupns host --security-opt label=disable -w /models"

podman run $PODMAN_CACHE_ARGS -it --rm --device /dev/dri localhost/llama-cpp.virtgpu
```

#### Test llama.cpp in Container

```bash

# Run performance benchmark
/app/remoting/build/bin/llama-bench -m ./llama3.2
```

Expected output (performance may vary):
```
| model                          |       size |     params | backend    | ngl |          test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------------: | -------------------: |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ggml-virtgpu |  99 |         pp512 |        991.30 ± 0.66 |
| llama 3B Q4_K - Medium         |   1.87 GiB |     3.21 B | ggml-virtgpu |  99 |         tg128 |         85.71 ± 0.11 |
```

### Troubleshooting

#### SSH Environment Variable Issues

⚠️ **Warning**: Setting `DYLD_LIBRARY_PATH` from SSH doesn't work on macOS. Here is a workaround:

**Workaround 1: Replace system library**
```bash
VIRGL_BUILD_DIR=$HOME/remoting/virglrenderer/build  # ⚠️ adapt to your system
BREW_VIRGL_DIR=/opt/homebrew/Cellar/virglrenderer/0.10.4d/lib
VIRGL_LIB=libvirglrenderer.1.dylib

cd $BREW_VIRGL_DIR
mv $VIRGL_LIB ${VIRGL_LIB}.orig
ln -s $VIRGL_BUILD_DIR/src/$VIRGL_LIB
```
