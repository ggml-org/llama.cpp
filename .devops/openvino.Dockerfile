ARG OPENVINO_VERSION_MAJOR=2026.2.1
ARG OPENVINO_VERSION_FULL=2026.2.1.21919.ede283a88e3
ARG UBUNTU_VERSION=24.04

# Intel GPU driver versions. https://github.com/intel/compute-runtime/releases
ARG IGC_VERSION=v2.36.3
ARG IGC_VERSION_FULL=2_2.36.3+21719
ARG COMPUTE_RUNTIME_VERSION=26.22.38646.4
ARG COMPUTE_RUNTIME_VERSION_FULL=26.22.38646.4-0
ARG IGDGMM_VERSION=22.10.0

# Intel NPU driver versions. https://github.com/intel/linux-npu-driver/releases
ARG NPU_DRIVER_VERSION=v1.33.0
ARG NPU_DRIVER_FULL=v1.33.0.20260529-26625960453
ARG LIBZE1_VERSION=1.27.0-1~24.04~ppa2

# Optional proxy build arguments
ARG http_proxy=
ARG https_proxy=

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A

ARG NODE_VERSION=24

FROM docker.io/node:$NODE_VERSION AS web

ARG APP_VERSION

WORKDIR /app/tools/ui

COPY tools/ui/package.json tools/ui/package-lock.json ./
RUN npm ci

COPY tools/ui/ ./
RUN LLAMA_BUILD_NUMBER="$APP_VERSION" npm run build

## Build Image
FROM docker.io/ubuntu:${UBUNTU_VERSION} AS build

# Pass proxy args to build stage
ARG http_proxy
ARG https_proxy

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        gnupg \
        wget \
        git \
        cmake \
        ninja-build \
        build-essential \
        libtbb12 \
        libssl-dev \
        ocl-icd-opencl-dev \
        opencl-headers \
        opencl-clhpp-headers \
        intel-opencl-icd && \
    rm -rf /var/lib/apt/lists/*

# OpenVINO toolkit and GPU/NPU drivers are cached via BuildKit cache mounts to avoid re-downloading on rebuilds.
# Install OpenVINO for Ubuntu 24.04.
ARG OPENVINO_VERSION_MAJOR
ARG OPENVINO_VERSION_FULL
RUN --mount=type=cache,target=/var/cache/openvino,sharing=locked \
    mkdir -p /opt/intel && \
    TGZ=/var/cache/openvino/openvino_toolkit_ubuntu24_${OPENVINO_VERSION_FULL}_x86_64.tgz && \
    if [ ! -f "$TGZ" ]; then \
        wget -O "$TGZ" https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION_MAJOR}/linux/openvino_toolkit_ubuntu24_${OPENVINO_VERSION_FULL}_x86_64.tgz; \
    fi && \
    tar -xf "$TGZ" -C /opt/intel/ && \
    mv /opt/intel/openvino_toolkit_ubuntu24_${OPENVINO_VERSION_FULL}_x86_64 /opt/intel/openvino_${OPENVINO_VERSION_MAJOR} && \
    cd /opt/intel/openvino_${OPENVINO_VERSION_MAJOR} && \
    echo "Y" | ./install_dependencies/install_openvino_dependencies.sh && \
    cd - && \
    ln -s /opt/intel/openvino_${OPENVINO_VERSION_MAJOR} /opt/intel/openvino

ENV OpenVINO_DIR=/opt/intel/openvino

WORKDIR /app

COPY . .

COPY --from=web /app/tools/ui/dist tools/ui/dist

# Build Stage
RUN bash -c "source ${OpenVINO_DIR}/setupvars.sh && \
    cmake -B build/ReleaseOV -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=OFF \
        -DGGML_OPENVINO=ON && \
    cmake --build build/ReleaseOV --parallel "

# Copy all necessary libraries (build outputs + OpenVINO runtime libs)
RUN mkdir -p /app/lib && \
    find build/ReleaseOV -name '*.so*' -exec cp -P {} /app/lib \; && \
    find "${OpenVINO_DIR}/runtime/lib/intel64" -name '*.so*' -exec cp -P {} /app/lib \;

# Create runtime directories and copy binaries
RUN mkdir -p /app/full \
    && cp build/ReleaseOV/bin/* /app/full/ \
    && cp *.py /app/full \
    && cp -r conversion /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

## Base Runtime Image
FROM docker.io/ubuntu:${UBUNTU_VERSION} AS base

# Pass proxy args to runtime stage
ARG http_proxy
ARG https_proxy
ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG IMAGE_URL=https://github.com/ggml-org/llama.cpp
ARG IMAGE_SOURCE=https://github.com/ggml-org/llama.cpp
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
      org.opencontainers.image.title="llama.cpp" \
      org.opencontainers.image.description="LLM inference in C/C++" \
      org.opencontainers.image.url=$IMAGE_URL \
      org.opencontainers.image.source=$IMAGE_SOURCE

RUN apt-get update \
    && apt-get install -y libgomp1 libtbb12 curl wget ffmpeg ocl-icd-libopencl1 \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

# Install GPU drivers
ARG IGC_VERSION
ARG IGC_VERSION_FULL
ARG COMPUTE_RUNTIME_VERSION
ARG COMPUTE_RUNTIME_VERSION_FULL
ARG IGDGMM_VERSION
RUN --mount=type=cache,target=/var/cache/intel-gpu,sharing=locked \
    set -eux; \
    cd /var/cache/intel-gpu; \
    for url in \
        https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_VERSION}/intel-igc-core-${IGC_VERSION_FULL}_amd64.deb \
        https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_VERSION}/intel-igc-opencl-${IGC_VERSION_FULL}_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libigdgmm12_${IGDGMM_VERSION}_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb ; do \
        f=$(basename "$url"); \
        [ -f "$f" ] || wget -q -O "$f" "$url"; \
    done; \
    apt-get update; \
    apt-get install -y --no-install-recommends ./*.deb; \
    rm -rf /var/lib/apt/lists/*

# Install NPU drivers
ARG NPU_DRIVER_VERSION
ARG NPU_DRIVER_FULL
ARG LIBZE1_VERSION
RUN --mount=type=cache,target=/var/cache/intel-npu,sharing=locked \
    set -eux; \
    TGZ=/var/cache/intel-npu/linux-npu-driver-${NPU_DRIVER_FULL}-ubuntu2404.tar.gz; \
    if [ ! -f "$TGZ" ]; then \
        wget -q -O "$TGZ" https://github.com/intel/linux-npu-driver/releases/download/${NPU_DRIVER_VERSION}/linux-npu-driver-${NPU_DRIVER_FULL}-ubuntu2404.tar.gz; \
    fi; \
    DEB=/var/cache/intel-npu/libze1_${LIBZE1_VERSION}_amd64.deb; \
    if [ ! -f "$DEB" ]; then \
        wget -q -O "$DEB" https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260324T100000Z/pool/main/l/level-zero-loader/libze1_${LIBZE1_VERSION}_amd64.deb; \
    fi; \
    mkdir /tmp/npu/ && cd /tmp/npu/ && tar -xf "$TGZ" && cp "$DEB" .; \
    apt-get update; \
    apt-get install -y --no-install-recommends ./*.deb; \
    rm -rf /tmp/npu/ /var/lib/apt/lists/*

COPY --from=build /app/lib/ /app/

### Full (all binaries)
FROM base AS full

ARG http_proxy
ARG https_proxy

COPY --from=build /app/full /app/

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-venv \
    python3-pip && \
    python3 -m venv /openvino-venv && \
    /openvino-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /openvino-venv/bin/pip install --no-cache-dir -r requirements.txt && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /tmp/* /var/tmp/* && \
    find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete && \
    find /var/cache -type f -delete

# Activate the venv
ENV VIRTUAL_ENV=/openvino-venv \
    PATH=/openvino-venv/bin:$PATH

ENTRYPOINT ["/app/tools.sh"]


### Light, CLI only
FROM base AS light

COPY --from=build /app/full/llama /app/full/llama-cli /app/full/llama-completion /app/

WORKDIR /app

ENTRYPOINT [ "/app/llama-cli" ]

### Server, Server only
FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/full/llama /app/full/llama-server /app/

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
