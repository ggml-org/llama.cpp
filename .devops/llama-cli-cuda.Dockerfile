ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=12.6.0
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# Target the CUDA runtime image
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} AS build

# CUDA architecture to build for (defaults to all supported archs)
ARG CUDA_DOCKER_ARCH=default

RUN apt-get update && \
    apt-get install -y build-essential git cmake

WORKDIR /app

COPY . .

# Use the default CUDA archs if not specified
RUN if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
        export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build -DGGML_CUDA=ON ${CMAKE_ARGS} -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release --target llama-cli -j$(nproc)

FROM ${BASE_CUDA_RUN_CONTAINER} AS runtime

RUN apt-get update && \
    apt-get install -y libgomp1

COPY --from=build /app/build/ggml/src/libggml.so /libggml.so
COPY --from=build /app/build/src/libllama.so /libllama.so
COPY --from=build /app/build/bin/llama-cli /llama-cli

ENTRYPOINT [ "/llama-cli" ]
