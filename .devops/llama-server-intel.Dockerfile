ARG ONEAPI_VERSION=2024.1.1-devel-ubuntu22.04

FROM intel/oneapi-basekit:$ONEAPI_VERSION as build

ARG LLAMA_SYCL_F16=OFF
RUN apt-get update && \
    apt-get install -y git libcurl4-openssl-dev

WORKDIR /app

COPY . .

RUN if [ "${LLAMA_SYCL_F16}" = "ON" ]; then \
        echo "LLAMA_SYCL_F16 is set" && \
        export OPT_SYCL_F16="-DLLAMA_SYCL_F16=ON"; \
    fi && \
    cmake -B build -DLLAMA_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DLLAMA_CURL=ON ${OPT_SYCL_F16} && \
    cmake --build build --config Release --target llama-server

FROM intel/oneapi-basekit:$ONEAPI_VERSION as runtime

RUN apt-get update && \
    apt-get install -y libcurl4-openssl-dev curl

COPY --from=build /app/build/bin/llama-server /llama-server

ENV LC_ALL=C.utf8

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/llama-server" ]
