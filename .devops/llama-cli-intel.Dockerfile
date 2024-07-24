ARG ONEAPI_VERSION=2024.1.1-devel-ubuntu22.04

FROM intel/oneapi-basekit:$ONEAPI_VERSION AS build

ARG GGML_SYCL_F16=OFF
RUN apt-get update && \
    apt-get install -y git

WORKDIR /app

COPY . .

RUN if [ "${GGML_SYCL_F16}" = "ON" ]; then \
        echo "GGML_SYCL_F16 is set" && \
        export OPT_SYCL_F16="-DGGML_SYCL_F16=ON"; \
    fi && \
    echo "Building with static libs" && \
    cmake -B build -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
    ${OPT_SYCL_F16} -DBUILD_SHARED_LIBS=OFF && \
    cmake --build build --config Release --target llama-cli

FROM intel/oneapi-basekit:$ONEAPI_VERSION AS runtime

COPY --from=build /app/build/bin/llama-cli /llama-cli

ENV LC_ALL=C.utf8

ENTRYPOINT [ "/llama-cli" ]
