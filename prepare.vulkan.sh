cmake -S . \
      -B ../build.vulkan \
      -DGGML_VULKAN=ON \
      -DGGML_NATIVE=OFF \
      -DGGML_METAL=OFF \
      -DCMAKE_BUILD_TYPE=Debug
