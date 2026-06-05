rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_VULKAN=ON
cmake --build build -j $(nproc)
