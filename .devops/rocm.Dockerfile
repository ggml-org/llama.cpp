ARG UBUNTU_VERSION=24.04

# This needs to generally match the container host's environment.
ARG ROCM_VERSION=7.0
ARG AMDGPU_VERSION=7.0

# Target the ROCm build image
ARG BASE_ROCM_DEV_CONTAINER=rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION}-complete

### Build image
FROM ${BASE_ROCM_DEV_CONTAINER} AS build

# Unless otherwise specified, we make a fat build.
# List from https://github.com/ggml-org/llama.cpp/pull/1087#issuecomment-1682807878
# This is mostly tied to rocBLAS supported archs.
# gfx803, gfx900, gfx906, gfx1032, gfx1101, gfx1102,not officialy supported
# check https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.1/reference/system-requirements.html

ARG ROCM_DOCKER_ARCH='gfx803;gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1010;gfx1030;gfx1032;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1151'
#ARG ROCM_DOCKER_ARCH='gfx1151'

# Set ROCm architectures
ENV AMDGPU_TARGETS=${ROCM_DOCKER_ARCH}

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    cmake \
    git \
    libcurl4-openssl-dev \
    curl \
    libgomp1

WORKDIR /app

COPY . .

RUN HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
        -DGGML_HIP=ON \
        -DGGML_HIP_ROCWMMA_FATTN=ON \
        -DAMDGPU_TARGETS="$ROCM_DOCKER_ARCH" \
        -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/lib \
    && find build -name "*.so" -exec cp {} /app/lib \;

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

## Base image
FROM ${BASE_ROCM_DEV_CONTAINER} AS base

RUN apt-get update \
    && apt-get install -y libgomp1 curl\
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

COPY --from=build /app/lib/ /app

### Full
FROM base AS full

COPY --from=build /app/full /app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y \
    git \
    python3-pip \
    python3 \
    python3-wheel\
    && pip install --break-system-packages --upgrade setuptools \
    && pip install --break-system-packages -r requirements.txt \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

ENTRYPOINT ["/app/tools.sh"]

### Light, CLI only
FROM base AS light

COPY --from=build /app/full/llama-cli /app

WORKDIR /app

ENTRYPOINT [ "/app/llama-cli" ]

### Server, Server only
FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/full/llama-server /app

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
