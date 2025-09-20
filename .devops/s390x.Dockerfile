ARG GCC_VERSION=15.2.0
ARG UBUNTU_VERSION=24.04

### Build OpenBLAS stage
FROM --platform=linux/s390x gcc:${GCC_VERSION} AS build-openblas

ARG OPENBLAS_THREADS=8

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt update -y && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
        git pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone https://github.com/taronaeo/OpenBLAS-s390x /build

RUN export NUM_THREADS=${OPENBLAS_THREADS} && \
    export OPENBLAS_NUM_THREADS=$NUM_THREADS && \
    export GOTO_NUM_THREADS=$NUM_THREADS && \
    export OMP_NUM_THREADS=$NUM_THREADS && \
    make -j $(nproc) USE_OPENMP=1 && \
    make -j $(nproc) USE_OPENMP=1 PREFIX=/opt/openblas-libs install


### Build Llama.cpp stage
FROM --platform=linux/s390x gcc:${GCC_VERSION} AS build-llama

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt update -y && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
        git cmake ccache ninja-build libcurl4-openssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
COPY --from=build-openblas /opt/openblas-libs /opt/openblas-libs

RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=cache,target=/app/build \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DLLAMA_BUILD_TESTS=OFF \
        -DGGML_BACKEND_DL=OFF \
        -DGGML_NATIVE=OFF \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=OpenBLAS \
        -DBLAS_LIBRARIES="/opt/openblas-libs/lib/libopenblas.so" \
        -DBLAS_INCLUDE_DIRS="/opt/openblas-libs/include" && \
    cmake --build build --config Release -j $(nproc) && \
    cmake --install build --prefix /opt/llama.cpp

COPY *.py             /opt/llama.cpp/bin
COPY .devops/tools.sh /opt/llama.cpp/bin

COPY gguf-py          /opt/llama.cpp/gguf-py
COPY requirements.txt /opt/llama.cpp/gguf-py
COPY requirements     /opt/llama.cpp/gguf-py/requirements


### Collect all llama.cpp binaries, libraries and distro libraries
FROM --platform=linux/s390x scratch AS collector

# Copy llama.cpp binaries and libraries
COPY --from=build-llama /opt/llama.cpp/bin     /llama.cpp/bin
COPY --from=build-llama /opt/llama.cpp/lib     /llama.cpp/lib
COPY --from=build-llama /opt/llama.cpp/gguf-py /llama.cpp/gguf-py

# Copy patched OpenBLAS libraries
COPY --from=build-openblas /opt/openblas-libs/lib /llama.cpp/lib


### Base image
FROM --platform=linux/s390x ubuntu:${UBUNTU_VERSION} AS base

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
    apt update -y && \
    apt install -y --no-install-recommends \
        curl libgfortran5 && \
    apt autoremove -y && \
    apt clean -y && \
    rm -rf /tmp/* /var/tmp/* && \
    find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete && \
    find /var/cache -type f -delete

# Copy llama.cpp libraries
COPY --from=collector /llama.cpp/lib /usr/lib/s390x-linux-gnu


### Full
FROM --platform=linux/s390x base AS full

ENV PATH="/root/.cargo/bin:${PATH}"
WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists \
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

RUN pip install --no-cache-dir --break-system-packages \
        -r /app/gguf-py/requirements.txt

ENTRYPOINT [ "/app/tools.sh" ]


### CLI Only
FROM --platform=linux/s390x base AS light

WORKDIR /llama.cpp/bin

# Copy llama.cpp binaries and libraries
COPY --from=collector /llama.cpp/bin/llama-cli /llama.cpp/bin

ENTRYPOINT [ "/llama.cpp/bin/llama-cli" ]


### Server
FROM --platform=linux/s390x base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

WORKDIR /llama.cpp/bin

# Copy llama.cpp binaries and libraries
COPY --from=collector /llama.cpp/bin/llama-server /llama.cpp/bin

EXPOSE 8080

ENTRYPOINT [ "/llama.cpp/bin/llama-server" ]
