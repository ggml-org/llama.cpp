ARG UBUNTU_VERSION=24.04

FROM ubuntu:$UBUNTU_VERSION AS build

# Ref: https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html

# Install build tools
RUN apt update && apt install -y git build-essential cmake wget xz-utils

# Install Vulkan SDK
ARG VULKAN_VERSION=1.4.321.1
RUN ARCH=$(uname -m) && \
    wget -qO /tmp/vulkan-sdk.tar.xz https://sdk.lunarg.com/sdk/download/${VULKAN_VERSION}/linux/vulkan-sdk-linux-${ARCH}-${VULKAN_VERSION}.tar.xz && \
    mkdir -p /opt/vulkan && \
    tar -xf /tmp/vulkan-sdk.tar.xz -C /tmp --strip-components=1 && \
    mv /tmp/${ARCH}/* /opt/vulkan/ && \
    rm -rf /tmp/*

# Install cURL and Vulkan SDK dependencies
RUN apt install -y libcurl4-openssl-dev curl \
    libxcb-xinput0 libxcb-xinerama0 libxcb-cursor-dev

# Set environment variables
ENV VULKAN_SDK=/opt/vulkan
ENV PATH=$VULKAN_SDK/bin:$PATH
ENV LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH=$VULKAN_SDK:$CMAKE_PREFIX_PATH
ENV PKG_CONFIG_PATH=$VULKAN_SDK/lib/pkgconfig:$PKG_CONFIG_PATH

# Build it
WORKDIR /app

COPY . .

RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_VULKAN=1  -DLLAMA_BUILD_TESTS=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON && \
    cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so" -exec cp {} /app/lib \;

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

## Base image
FROM ubuntu:$UBUNTU_VERSION AS base

RUN apt-get update \
    && apt-get install -y libgomp1 curl libvulkan-dev \
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
    python3 \
    python3-pip \
    python3-wheel \
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
