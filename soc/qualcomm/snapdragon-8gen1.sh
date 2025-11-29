#!/bin/bash

# Snapdragon 8+ Gen 1 Llama.cpp 编译优化脚本

# By https://github.com/Insecta258 Email:2701262643@qq.com

pkg update
pkg upgrade

echo "Checking environment..."
pkg install ninja git clang make

git clone https://github.com/flame/blis.git
chmod +x -R blis/
cd blis/
./configure --enable-cblas --enable-threading=pthreads --p>
make -j4 && make install
cd ..

ls $PREFIX/lib/libblis*
ls $PREFIX/include/blis*
sleep 1

read -p "是否添加 Vulkan 支持? (y/n): " choice
choice=$(echo "$choice" | tr '[:upper:]' '[:lower:]')

vulkan_flag=""
if [[ "$choice" == "y" || "$choice" == "yes" ]]; then
    echo "正在为骁龙8+ Gen 1启用 Vulkan 支持..."
    # 确保已安装正确的驱动
    pkg remove mesa-vulkan-icd-swrast
    pkg install mesa-vulkan-icd-freedreno vulkan-loader-generic vulkan-tools
    vulkaninfo | grep -i devicename
    sleep 1
    vulkan_flag="-DGGML_VULKAN=ON"
else
    echo "将不启用 Vulkan 支持。"
    vulkan_flag="-DGGML_VULKAN=OFF"
fi

# 清理并创建构建目录
rm -rf build
mkdir build && cd build

# --- 针对骁龙8+ Gen 1的编译器标志 ---
# 1. CPU调优: Kryo CPU基于Cortex-X2，使用cortex-x2作为tune目标更精确
# 2. 架构: armv9.2-a
# 3. 性能/功耗: 骁龙8+ Gen 1能效提升30%(C8)，可以放心使用-O3
# 4. AI: 虽然硬件支持FP16等，但llama.cpp的GGML后端主要通过Vulkan和CPU指令加速，无需特殊AI标志
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-O3 -flto -march=armv9.2-a+dotprod -mtune=cortex-x2 -DNDEBUG" \
  -DCMAKE_C_FLAGS="-O3 -flto -march=armv9.2-a+dotprod -mtune=cortex-x2 -DNDEBUG" \
  -DBUILD_SHARED_LIBS=ON \
  ${vulkan_flag} \
  -DGGML_DOTPROD=ON \
  -DGGML_BLAS=ON \
  -DGGML_BLAS_VENDOR=FLAME \
  -DGGML_NATIVE=ON

echo "开始编译"
sleep 1

ninja
