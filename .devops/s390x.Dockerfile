ARG GCC_VERSION=15.2.0
ARG UBUNTU_VERSION=24.04
ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A

### Build Llama.cpp stage
FROM docker.io/gcc:${GCC_VERSION} AS build

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt update -y && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
        git cmake ccache ninja-build \
        # WARNING: Do not use libopenblas-openmp-dev. libopenblas-dev is faster.
        libopenblas-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=/app/build \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DLLAMA_BUILD_TESTS=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=OpenBLAS && \
    cmake --build build --config Release -j $(nproc) && \
    cmake --install build --prefix /opt/llama.cpp

COPY *.py             /opt/llama.cpp/bin
COPY .devops/tools.sh /opt/llama.cpp/bin
COPY conversion       /opt/llama.cpp/conversion

COPY gguf-py          /opt/llama.cpp/gguf-py
COPY requirements.txt /opt/llama.cpp/gguf-py
COPY requirements     /opt/llama.cpp/gguf-py/requirements


### Collect all llama.cpp binaries, libraries and distro libraries
FROM scratch AS collector

# Copy llama.cpp binaries and libraries
COPY --from=build /opt/llama.cpp/bin        /llama.cpp/bin
COPY --from=build /opt/llama.cpp/lib        /llama.cpp/lib
COPY --from=build /opt/llama.cpp/gguf-py    /llama.cpp/gguf-py
COPY --from=build /opt/llama.cpp/conversion /llama.cpp/conversion


### Base image
FROM docker.io/ubuntu:${UBUNTU_VERSION} AS base

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

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt update -y && \
    apt install -y --no-install-recommends \
        # WARNING: Do not use libopenblas-openmp-dev. libopenblas-dev is faster.
        # See: https://github.com/ggml-org/llama.cpp/pull/15915#issuecomment-3317166506
        curl libgomp1 libopenblas-dev && \
    apt autoremove -y && \
    apt clean -y && \
    rm -rf /tmp/* /var/tmp/* && \
    find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete && \
    find /var/cache -type f -delete

# Copy llama.cpp libraries
COPY --from=collector /llama.cpp/lib /usr/lib/s390x-linux-gnu


### Full
FROM base AS full

ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt update -y && \
    apt install -y \
        git cmake libjpeg-dev \
        python3 python3-pip python3-dev && \
    apt autoremove -y && \
    apt clean -y && \
    rm -rf /tmp/* /var/tmp/* && \
    find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete && \
    find /var/cache -type f -delete

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

COPY --from=collector /llama.cpp/bin /app
COPY --from=collector /llama.cpp/gguf-py /app/gguf-py
COPY --from=collector /llama.cpp/conversion /app/conversion

RUN pip install --no-cache-dir --break-system-packages \
        -r /app/gguf-py/requirements.txt

ENTRYPOINT [ "/app/tools.sh" ]


### CLI Only
FROM base AS light

WORKDIR /llama.cpp/bin

# Copy llama.cpp binaries and libraries
COPY --from=collector /llama.cpp/bin/*.so /llama.cpp/bin
COPY --from=collector /llama.cpp/bin/llama-cli /llama.cpp/bin/llama-completion /llama.cpp/bin

ENTRYPOINT [ "/llama.cpp/bin/llama-cli" ]


### Server
FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

WORKDIR /llama.cpp/bin

# Copy llama.cpp binaries and libraries
COPY --from=collector /llama.cpp/bin/*.so /llama.cpp/bin
COPY --from=collector /llama.cpp/bin/llama-server /llama.cpp/bin

EXPOSE 8080

ENTRYPOINT [ "/llama.cpp/bin/llama-server" ]
