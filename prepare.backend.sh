cmake -S . -B ../build.remoting-backend \
      -DGGML_REMOTINGBACKEND=ON \
      -DGGML_NATIVE=OFF \
      -DGGML_METAL=ON \
      -DGGML_BACKEND_DL=OFF \
      -DLLAMA_CURL=OFF \
      -DGGML_VULKAN=OFF -DVulkan_INCLUDE_DIR=/opt/homebrew/include/ -DVulkan_LIBRARY=/opt/homebrew/lib/libMoltenVK.dylib \
      "$@"

#      -DCMAKE_BUILD_TYPE=Debug \
#
