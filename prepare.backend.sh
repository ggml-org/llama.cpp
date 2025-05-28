if [[ "${PERF_MODE:-}" ]]; then
    FLAVOR="-prod"
else
    FLAVOR=""
fi

cmake -S . -B ../build.remoting-backend$FLAVOR \
      -DGGML_REMOTINGBACKEND=ON \
      -DGGML_NATIVE=OFF \
      -DGGML_METAL=ON \
      -DGGML_VULKAN=OFF -DVulkan_INCLUDE_DIR=/opt/homebrew/include/ -DVulkan_LIBRARY=/opt/homebrew/lib/libMoltenVK.dylib \
      "$@"

#      -DCMAKE_BUILD_TYPE=Debug \
#
