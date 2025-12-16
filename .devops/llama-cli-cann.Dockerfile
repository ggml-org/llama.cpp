ARG ASCEND_VERSION=8.1.RC1.alpha001-910b-openeuler22.03-py3.10

FROM ascendai/cann:$ASCEND_VERSION AS build

WORKDIR /app

COPY . .

RUN yum install -y gcc g++ cmake make libcurl-devel
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH}
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${PATH}
ENV ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
ENV TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}

# find libascend_hal.so, because the drive hasn`t been mounted.
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/runtime/lib64/stub:$LD_LIBRARY_PATH

RUN echo "Building with static libs" && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh --force && \
    cmake -B build -DGGML_NATIVE=OFF -DGGML_CANN=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF  && \
    cmake --build build --config Release --target llama-cli && \
    cmake --build build --config Release --target llama-completion

# TODO: use image with NNRT
FROM ascendai/cann:$ASCEND_VERSION AS runtime
COPY --from=build /app/build/bin/llama-cli /app/build/bin/llama-completion /

ENV LC_ALL=C.utf8

ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH}
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${PATH}
ENV ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
ENV TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}

ENTRYPOINT ["/llama-cli" ]
