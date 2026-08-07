rm -rf build/
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_VULKAN=ON
cmake --build build --config Release -j16
