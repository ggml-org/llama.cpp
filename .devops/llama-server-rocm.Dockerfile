ARG UBUNTU_VERSION=22.04

# This needs to generally match the container host's environment.
ARG ROCM_VERSION=5.6

# Target the CUDA build image
ARG BASE_ROCM_DEV_CONTAINER=rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION}-complete

FROM ${BASE_ROCM_DEV_CONTAINER} AS build

# Unless otherwise specified, we make a fat build.
# List from https://github.com/ggerganov/llama.cpp/pull/1087#issuecomment-1682807878
# This is mostly tied to rocBLAS supported archs.
ARG ROCM_DOCKER_ARCH=\
    gfx803 \
    gfx900 \
    gfx906 \
    gfx908 \
    gfx90a \
    gfx1010 \
    gfx1030 \
    gfx1100 \
    gfx1101 \
    gfx1102

COPY requirements.txt   requirements.txt
COPY requirements       requirements

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

WORKDIR /app

COPY . .

# Set nvcc architecture
ENV GPU_TARGETS=${ROCM_DOCKER_ARCH}
# Enable ROCm
ENV GGML_HIPBLAS=1
ENV CC=/opt/rocm/llvm/bin/clang
ENV CXX=/opt/rocm/llvm/bin/clang++
# Must be set to 0.0.0.0 so it can listen to requests from host machine
ENV LLAMA_ARG_HOST=0.0.0.0

# Enable cURL
ENV LLAMA_CURL=1
RUN apt-get update && \
    apt-get install -y libcurl4-openssl-dev curl

RUN make -j$(nproc) llama-server

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
