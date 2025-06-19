cmake -S . -B ../build.remoting-frontend \
      -DGGML_REMOTINGFRONTEND=ON \
      -DGGML_CPU_ARM_ARCH=native \
      -DGGML_NATIVE=OFF \
      -DGGML_OPENMP=OFF \
      -DLLAMA_CURL=OFF \
      -DCMAKE_BUILD_TYPE=Debug \
      "$@"
