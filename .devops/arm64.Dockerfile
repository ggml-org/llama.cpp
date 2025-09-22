# Args for Build actions
ARG BUILDPLATFORM_builder=linux/amd64
ARG BUILDPLATFORM_runner=linux/arm64
ARG TARGETARCH

# Stage 1: Builder Docker
FROM --platform=$BUILDPLATFORM_builder debian:bookworm-slim AS builder

# FROM ubuntu:$UBUNTU_VERSION AS builder

# add arm64 deps
RUN dpkg --add-architecture arm64

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    ninja-build \
    ca-certificates \
    libopenblas-dev \
    libgomp1 \
    libcurl4-openssl-dev \
    gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu \
    libc6-dev-arm64-cross \
    libcurl4-openssl-dev:arm64 \
    libssl-dev:arm64 \
    && update-ca-certificates

WORKDIR /workspace

COPY . .

# Set your cross compilers environment variables (adjust if needed)
ENV CC64=aarch64-linux-gnu-gcc
ENV CXX64=aarch64-linux-gnu-g++

# remove 'armv9' since gcc-12 doesn't support it
RUN sed -i '/armv9/d' "ggml/src/CMakeLists.txt"

# Run CMake configure and build

RUN  cmake -S . -B build \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=$CC64 \
    -DCMAKE_CXX_COMPILER=$CXX64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCURL_INCLUDE_DIR=/usr/aarch64-linux-gnu/include \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON 
RUN cmake --build build -j $(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so" -exec cp {} /app/lib \;

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

# Stage 2: Runtime
FROM --platform=$BUILDPLATFORM_runner debian:bookworm-slim AS base

#FROM ubuntu:$UBUNTU_VERSION AS base

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy built binaries from builder
COPY --from=builder /app/full /app

# Full
FROM base AS full

COPY --from=builder /app/full /app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y \
    git \
    python3 \
    python3-pip

RUN pip install --no-cache-dir --upgrade pip setuptools wheel --break-system-packages
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# Clean up unnecessary files to reduce image size
RUN rm -rf /root/.cache/pip && rm -rf /root/.cache/build && rm -rf /tmp/* && rm -rf /var/tmp/*
RUN apt autoremove -y && apt clean -y
RUN find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete
RUN find /var/cache -type f -delete

ENTRYPOINT ["/app/tools.sh"]

# Light
FROM base AS light

COPY --from=builder /app/full/llama-cli /app

WORKDIR /app

ENTRYPOINT [ "/app/llama-cli" ]

# Server
FROM base AS server

COPY --from=builder /app/full/llama-server /app

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
