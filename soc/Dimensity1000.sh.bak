#!/bin/bash

echo "Checking environment..."
pkg install ninja

read -p "是否添加 Vulkan 支持? (y/n): " choice
choice=$(echo "$choice" | tr '[:upper:]' '[:lower:]')

opencl_flag=""
if [[ "$choice" == "y" || "$choice" == "yes" ]]; then
    echo "Checking environment..."
    pkg install vulkan-loader-android
    pkg install llama-cpp-backend-vulkan
    echo "将启用 Vulkan 支持。"
    vulkan_flag="-DGGML_VULKAN=ON -DGGML_VULKAN_TARGET_SPIRV_VERSION=1.1 -DGGML_VULKAN_CHECK_RESULTS=0"
else
    echo "将不启用 Vulkan 支持。"
    vulkan_flag="-DGGML_VULKAN=OFF"
fi

# 清理构建目录
mkdir build && cd build

# 修复后的编译器标志 - 移除冲突选项
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -flto -march=armv8.2-a+dotprod -mtune=cortex-a77 -DNDEBUG" \
  -DCMAKE_C_FLAGS="-O3 -flto -march=armv8.2-a+dotprod -mtune=cortex-a77 -DNDEBUG" \
  -DBUILD_SHARED_LIBS=ON \
  ${vulkan_flag} \
  -DGGML_DOTPROD=ON \
  -DGGML_NATIVE=ON

ninja
