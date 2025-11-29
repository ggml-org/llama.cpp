#!/bin/bash

# By https://github.com/Insecta258 Email:2701262643@qq.com

pkg update
pkg upgrade

echo "Checking environment..."
pkg install ninja clang git make

git clone https://github.com/flame/blis.git
chmod +x -R blis/
cd blis/
./configure --enable-cblas --enable-threading=pthreads --prefix=$PREFIX arm64
make -j4 && make install
cd ..

ls $PREFIX/lib/libblis*
ls $PREFIX/include/blis*
sleep 1

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
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=FLAME \
  -DGGML_NATIVE=ON

echo "开始编译"
sleep 1

ninja
