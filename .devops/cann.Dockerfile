# ==============================================================================
# ARGUMENTS
# ==============================================================================

# 定义CANN基础镜像，方便后续统一更新版本
ARG CANN_BASE_IMAGE=quay.io/ascend/cann:8.1.rc1-910b-openeuler22.03-py3.10


# ==============================================================================
# BUILD STAGE
# 编译所有二进制文件和库
# ==============================================================================
FROM ${CANN_BASE_IMAGE} AS build

# 定义昇腾芯片型号，用于编译。默认为 Ascend910B3
ARG ASCEND_SOC_TYPE=Ascend910B3

# -- 安装构建依赖 --
RUN yum install -y gcc g++ cmake make git libcurl-devel python3 python3-pip && \
    yum clean all && \
    rm -rf /var/cache/yum

# -- 设置工作目录 --
WORKDIR /app

# -- 拷贝项目文件 --
COPY . .

# -- 设置CANN环境变量 (编译时需要) --
# 相比于 `source`，使用 ENV 可以让环境变量在整个镜像层中持久生效
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${PATH}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/runtime/lib64/stub:$LD_LIBRARY_PATH
# ... 您可以根据需要添加原始文件中其他的环境变量 ...
# 为了简洁，这里只列出核心变量，您可以将原始的ENV列表粘贴于此

# -- 编译 llama.cpp --
# 使用传入的 ASCEND_SOC_TYPE 参数，并增加通用编译选项
RUN source /usr/local/Ascend/ascend-toolkit/set_env.sh --force \
    && \
    cmake -B build \
        -DGGML_CANN=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DSOC_TYPE=${ASCEND_SOC_TYPE} \
        . && \
    cmake --build build --config Release -j$(nproc)

# -- 整理编译产物，方便后续阶段拷贝 --
# 创建一个lib目录存放所有.so文件
RUN mkdir -p /app/lib && \
    find build -name "*.so" -exec cp {} /app/lib \;

# 创建一个full目录存放所有可执行文件和Python脚本
RUN mkdir -p /app/full && \
    cp build/bin/* /app/full/ && \
    cp *.py /app/full/ && \
    cp -r gguf-py /app/full/ && \
    cp -r requirements /app/full/ && \
    cp requirements.txt /app/full/
    # 如果您有 tools.sh 脚本，也请确保它在此处被拷贝
    # cp .devops/tools.sh /app/full/tools.sh


# ==============================================================================
# BASE STAGE
# 创建一个包含CANN运行时和通用库的最小基础镜像
# ==============================================================================
FROM ${CANN_BASE_IMAGE} AS base

# -- 安装运行时依赖 --
RUN yum install -y libgomp curl && \
    yum clean all && \
    rm -rf /var/cache/yum

# -- 设置CANN环境变量 (运行时需要) --
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=/app:${ASCEND_TOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${PATH}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
# ... 您可以根据需要添加原始文件中其他的环境变量 ...

WORKDIR /app

# 从build阶段拷贝编译好的.so文件
COPY --from=build /app/lib/ /app


# ==============================================================================
# FINAL STAGES (TARGETS)
# ==============================================================================

### Target: full
# 包含所有工具、Python绑定和依赖的完整镜像
# ==============================================================================
FROM base AS full

COPY --from=build /app/full /app

# 安装Python依赖
RUN yum install -y git python3 python3-pip && \
    pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir -r requirements.txt && \
    yum clean all && \
    rm -rf /var/cache/yum

# 您需要提供一个 tools.sh 脚本作为入口点
ENTRYPOINT ["/app/tools.sh"]
# 如果没有 tools.sh，可以设置默认启动 server
# ENTRYPOINT ["/app/llama-server"]


### Target: light
# 仅包含 llama-cli 的轻量级镜像
# ==============================================================================
FROM base AS light

COPY --from=build /app/full/llama-cli /app


ENTRYPOINT [ "/app/llama-cli" ]


### Target: server
# 仅包含 llama-server 的专用服务器镜像
# ==============================================================================
FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/full/llama-server /app


HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]