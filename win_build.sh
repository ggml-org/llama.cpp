rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_VULKAN=ON \
  -DVULKAN_SDK="C:/VulkanSDK/1.4.350.0" \
  -DCMAKE_INCLUDE_PATH="C:/VulkanSDK/1.4.350.0/Include" \
  -DCMAKE_LIBRARY_PATH="C:/VulkanSDK/1.4.350.0/Lib" \
  -DVulkan_GLSLC_EXECUTABLE="C:/VulkanSDK/1.4.350.0/Bin/glslc.exe" \
  -DCMAKE_PREFIX_PATH="C:/VulkanSDK/1.4.350.0"
cmake --build build -j $(nproc)
