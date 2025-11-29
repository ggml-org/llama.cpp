#!/bin/bash

pkg update
pkg upgrade

echo "Checking environment..."
pkg install ninja clang git

echo "现版本llama.cpp不支持Mali-G77的Vulkan驱动, 回退到CPU"

sleep 1

# 清理构建目录
mkdir build && cd build

# 修复后的编译器标志 - 移除冲突选项
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -flto -march=armv8.2-a+dotprod -mtune=cortex-a77 -DNDEBUG" \
  -DCMAKE_C_FLAGS="-O3 -flto -march=armv8.2-a+dotprod -mtune=cortex-a77 -DNDEBUG" \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_VULKAN=OFF \
  -DGGML_OPENCL=OFF \
  -DGGML_DOTPROD=ON \
  -DGGML_NATIVE=ON

echo "开始编译"
sleep 1

ninja
