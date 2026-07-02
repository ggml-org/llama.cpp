FROM mixa3607/llama.cpp-gfx906:b9222-rocm-7.2.3

# 容器里已经有 ROCm 环境，安装编译工具（一般镜像已包含，保险起见补上）
RUN apt update && apt install -y build-essential cmake git

# 将整个魔改源码复制到容器内
COPY . /app/llamacpp_source

WORKDIR /app/llamacpp_source

# 用 HIP 后端编译（ROCm）
RUN cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --config Release -j$(nproc)

EXPOSE 1337

# 默认启动命令（你可以根据自己习惯修改参数）
CMD ["/app/llamacpp_source/build/bin/llama-server", \
     "--model", "/models/Qwen3.6-27B-Q6_K-GGUF/qwen3.6-27b-q6_k.gguf", \
     "--n-gpu-layers", "999", \
     "--ctx-size", "98304", \
     "--host", "0.0.0.0", \
     "--port", "1337", \
     "--parallel", "1", \
     "--threads", "12", \
     "--batch-size", "10240", \
     "--ubatch-size", "2048", \
     "--flash-attn", "on", \
     "--cache-type-k", "q8_0", \
     "--cache-type-v", "q8_0", \
     "--temp", "0.9", \
     "--top-p", "0.95", \
     "--top-k", "20", \
     "--checkpoint-every-n-tokens", "4096", \
     "--ctx-checkpoints", "32", \
     "--kv-unified", \
     "--repeat-penalty", "1.0", \
     "--presence-penalty", "0.1", \
     "--timeout", "3600"]
