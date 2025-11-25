#!/bin/bash

echo "Checking environment..."
pkg install ninja clang

read -p "是否添加 OpenCL 支持? (y/n): " choice
choice=$(echo "$choice" | tr '[:upper:]' '[:lower:]')

opencl_flag=""
if [[ "$choice" == "y" || "$choice" == "yes" ]]; then
    echo "Checking environment..."
    pkg install ocl-icd opencl-headers opencl-clhpp
    echo "将启用 OpenCL 支持。"
    opencl_flag="-DGGML_OPENCL=ON"
else
    echo "将不启用 OpenCL 支持。"
    opencl_flag="-DGGML_OPENCL=OFF"
fi

# 清理构建目录
mkdir build && cd build

# 修复后的编译器标志 - 移除冲突选项
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -march=armv8.2-a+dotprod -mtune=cortex-a77 -DNDEBUG" \
  -DCMAKE_C_FLAGS="-O3 -march=armv8.2-a+dotprod -mtune=cortex-a77 -DNDEBUG" \
  -DBUILD_SHARED_LIBS=ON \
  ${opencl_flag} \
  -DGGML_DOTPROD=ON \
  -DGGML_NATIVE=ON

ninja
