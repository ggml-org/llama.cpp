ARG OPENVINO_VERSION_MAJOR=2026.0
ARG OPENVINO_VERSION_FULL=2026.0.0.20965.c6d6a13a886
ARG UBUNTU_VERSION=24.04

# Intel GPU driver versions. https://github.com/intel/compute-runtime/releases
ARG IGC_VERSION=v2.30.1
ARG IGC_VERSION_FULL=2_2.30.1+20950
ARG COMPUTE_RUNTIME_VERSION=26.09.37435.1
ARG COMPUTE_RUNTIME_VERSION_FULL=26.09.37435.1-0
ARG IGDGMM_VERSION=22.9.0

# Intel NPU driver versions. https://github.com/intel/linux-npu-driver/releases
ARG NPU_DRIVER_VERSION=v1.32.0
ARG NPU_DRIVER_FULL=v1.32.0.20260402-23905121947
ARG LIBZE1_VERSION=1.27.0-1~24.04~ppa2

# Optional proxy build arguments
ARG http_proxy=
ARG https_proxy=

## Build Image
FROM ubuntu:${UBUNTU_VERSION} AS build

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

# Install OpenVINO for Ubuntu 24.04
ARG OPENVINO_VERSION_MAJOR
ARG OPENVINO_VERSION_FULL
RUN mkdir -p /opt/intel && \
    wget https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_VERSION_MAJOR}/linux/openvino_toolkit_ubuntu24_${OPENVINO_VERSION_FULL}_x86_64.tgz && \
    tar -xf openvino_toolkit_ubuntu24_${OPENVINO_VERSION_FULL}_x86_64.tgz && \
    mv openvino_toolkit_ubuntu24_${OPENVINO_VERSION_FULL}_x86_64 /opt/intel/openvino_${OPENVINO_VERSION_MAJOR} && \
    cd /opt/intel/openvino_${OPENVINO_VERSION_MAJOR} && \
    echo "Y" | ./install_dependencies/install_openvino_dependencies.sh && \
    cd - && \
    ln -s /opt/intel/openvino_${OPENVINO_VERSION_MAJOR} /opt/intel/openvino

ENV OpenVINO_DIR=/opt/intel/openvino

WORKDIR /app

COPY . .

# Build Stage
RUN bash -c "source ${OpenVINO_DIR}/setupvars.sh && \
    cmake -B build/ReleaseOV -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_OPENVINO=ON && \
    cmake --build build/ReleaseOV -j$(nproc)"

# Copy all necessary libraries
RUN mkdir -p /app/lib && \
    find build/ReleaseOV -name '*.so*' -exec cp {} /app/lib \; && \
    find ${OpenVINO_DIR}/runtime/lib/intel64 -name '*.so*' -exec cp -P {} /app/lib \; 2>/dev/null || \
    find ${OpenVINO_DIR}/lib/intel64 -name '*.so*' -exec cp -P {} /app/lib \;

# Create runtime directories and copy binaries
RUN mkdir -p /app/full \
    && cp build/ReleaseOV/bin/* /app/full/ \
    && cp *.py /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

## Base Runtime Image
FROM ubuntu:${UBUNTU_VERSION} AS base

# Pass proxy args to runtime stage
ARG http_proxy
ARG https_proxy

RUN apt-get update \
    && apt-get install -y libgomp1 libtbb12 curl wget ocl-icd-libopencl1 \
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
RUN mkdir /tmp/neo/ && cd /tmp/neo/ \
    && wget https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_VERSION}/intel-igc-core-${IGC_VERSION_FULL}_amd64.deb \
    && wget https://github.com/intel/intel-graphics-compiler/releases/download/${IGC_VERSION}/intel-igc-opencl-${IGC_VERSION_FULL}_amd64.deb \
    && wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc-dbgsym_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.ddeb \
    && wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-ocloc_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
    && wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-opencl-icd-dbgsym_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.ddeb \
    && wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/intel-opencl-icd_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
    && wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libigdgmm12_${IGDGMM_VERSION}_amd64.deb \
    && wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libze-intel-gpu1-dbgsym_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.ddeb \
    && wget https://github.com/intel/compute-runtime/releases/download/${COMPUTE_RUNTIME_VERSION}/libze-intel-gpu1_${COMPUTE_RUNTIME_VERSION_FULL}_amd64.deb \
    && dpkg --install *.deb \
    && rm -rf /tmp/neo/

# Install NPU drivers
ARG NPU_DRIVER_VERSION
ARG NPU_DRIVER_FULL
ARG LIBZE1_VERSION
RUN mkdir /tmp/npu/ && cd /tmp/npu/ \
    && wget https://github.com/intel/linux-npu-driver/releases/download/${NPU_DRIVER_VERSION}/linux-npu-driver-${NPU_DRIVER_FULL}-ubuntu2404.tar.gz \
    && tar -xf linux-npu-driver-${NPU_DRIVER_FULL}-ubuntu2404.tar.gz \
    && dpkg --install *.deb \
    && rm -rf /tmp/npu/

RUN cd /tmp \
    && wget https://snapshot.ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/20260324T100000Z/pool/main/l/level-zero-loader/libze1_${LIBZE1_VERSION}_amd64.deb \
    && dpkg --install libze1_${LIBZE1_VERSION}_amd64.deb \
    && rm libze1_${LIBZE1_VERSION}_amd64.deb

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
    python3 -m venv /ov-venv && \
    /ov-venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /ov-venv/bin/pip install --no-cache-dir -r requirements.txt && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /tmp/* /var/tmp/* && \
    find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete && \
    find /var/cache -type f -delete

ENTRYPOINT ["/bin/bash", "-c", "source /ov-venv/bin/activate && exec /app/tools.sh \"$@\"", "--"]


### Light, CLI only
FROM base AS light

COPY --from=build /app/full/llama-cli /app/

WORKDIR /app

ENTRYPOINT [ "/app/llama-cli" ]

### Server, Server only
FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/full/llama-server /app/

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
