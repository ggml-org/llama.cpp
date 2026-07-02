#!/bin/bash

# ============================================================
# 重新编译 llama.cpp (ROCm) Docker 镜像脚本
# 用法: ./rebuild_docker_llamacpp.sh
# 说明:
#   1. 停止并删除正在运行的 llama-rocm 容器 (如果存在)
#   2. 使用当前目录下的 Dockerfile 重新构建镜像
#   3. 输出构建结果
#   4. 构建完成后不会自动启动容器，你需要手动运行 start_27b.sh
# ============================================================

set -e  # 遇到错误立即退出

echo "========================================="
echo "1/3 停止并删除现有的 llama-rocm 容器..."
echo "========================================="

# 检查容器是否存在 (运行或已停止)
if docker ps -a --format "table {{.Names}}" | grep -q "llama-rocm"; then
    echo "检测到 llama-rocm 容器，正在停止并删除..."
    docker rm -f llama-rocm 2>/dev/null
    echo "容器已移除。"
else
    echo "未发现 llama-rocm 容器，跳过。"
fi

echo ""
echo "========================================="
echo "2/3 重新构建 Docker 镜像..."
echo "========================================="

# 检查 Dockerfile 是否存在
if [ ! -f "Dockerfile" ]; then
    echo "错误: 当前目录下未找到 Dockerfile，请确保在 llama.cpp 源码目录下运行此脚本。"
    exit 1
fi

# 构建镜像
docker build -t llama.cpp-rocm-custom .

echo ""
echo "========================================="
echo "3/3 构建结果"
echo "========================================="
echo "镜像构建完成。使用以下命令查看镜像:"
echo "  docker images | grep llama.cpp-rocm-custom"
echo ""
echo "注意: 新镜像已就绪，但尚未启动容器。"
echo "请手动运行 start_27b.sh 来启动 llama.cpp 服务。"
