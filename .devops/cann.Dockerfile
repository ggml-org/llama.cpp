# ==============================================================================
# ARGUMENTS
# ==============================================================================

# Define the CANN base image for easier version updates later
ARG CHIP_TYPE=910b
ARG CANN_BASE_IMAGE=quay.io/ascend/cann:8.3.rc2-${CHIP_TYPE}-openeuler24.03-py3.11

# ==============================================================================
# BUILD STAGE
# Compile all binary files and libraries
# ==============================================================================
FROM ${CANN_BASE_IMAGE} AS build

# -- Install build dependencies --
RUN yum install -y gcc g++ cmake make git openssl-devel python3 python3-pip && \
    yum clean all && \
    rm -rf /var/cache/yum

# -- Set the working directory --
WORKDIR /app

# -- Copy project files --
COPY . .

# -- Set CANN environment variables (required for compilation) --
# Using ENV instead of `source` allows environment variables to persist across the entire image layer
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${PATH}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/runtime/lib64/stub:$LD_LIBRARY_PATH
# ... You can add other environment variables from the original file as needed ...
# For brevity, only core variables are listed here. You can paste the original ENV list here.

# -- Build llama.cpp --
# Use the passed CHIP_TYPE argument and add general build options
ARG CHIP_TYPE
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh --force \
    && \
    cmake -B build \
        -DGGML_CANN=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DSOC_TYPE=ascend${CHIP_TYPE} \
        -DUSE_ACL_GRAPH=ON \
        . && \
    cmake --build build --config Release -j$(nproc)

# -- Organize build artifacts for copying in later stages --
# Create a lib directory to store all .so files
RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

# Create a full directory to store all executables and Python scripts
RUN mkdir -p /app/full && \
    cp build/bin/* /app/full/ && \
    cp *.py /app/full/ && \
    cp -r gguf-py /app/full/ && \
    cp -r requirements /app/full/ && \
    cp requirements.txt /app/full/
    # If you have a tools.sh script, make sure it is copied here
    # cp .devops/tools.sh /app/full/tools.sh

# ==============================================================================
# BASE STAGE
# Create a minimal base image with CANN runtime and common libraries
# ==============================================================================
FROM ${CANN_BASE_IMAGE} AS base

# -- Install runtime dependencies --
RUN yum install -y libgomp curl && \
    yum clean all && \
    rm -rf /var/cache/yum

# -- Set CANN environment variables (required for runtime) --
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=/app:${ASCEND_TOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${PATH}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
# ... You can add other environment variables from the original file as needed ...

WORKDIR /app

# Copy compiled .so files from the build stage
COPY --from=build /app/lib/ /app

# ==============================================================================
# FINAL STAGES (TARGETS)
# ==============================================================================

### Target: full
# Complete image with all tools, Python bindings, and dependencies
# ==============================================================================
FROM base AS full

COPY --from=build /app/full /app

# Install Python dependencies
RUN yum install -y git python3 python3-pip && \
    pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements.txt && \
    yum clean all && \
    rm -rf /var/cache/yum

# You need to provide a tools.sh script as the entrypoint
ENTRYPOINT ["/app/tools.sh"]
# If there is no tools.sh, you can set the default to start the server
# ENTRYPOINT ["/app/llama-server"]

### Target: light
# Lightweight image containing only llama-cli and llama-completion
# ==============================================================================
FROM base AS light

COPY --from=build /app/full/llama-cli /app/full/llama-completion /app

ENTRYPOINT [ "/app/llama-cli" ]

### Target: server
# Dedicated server image containing only llama-server
# ==============================================================================
FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/full/llama-server /app

HEALTHCHECK --interval=5m CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
