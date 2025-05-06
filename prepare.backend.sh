cmake -S . -B ../build.remoting-backend \
      -DGGML_REMOTINGBACKEND=ON \
      -DGGML_NATIVE=OFF \
      -DCMAKE_BUILD_TYPE=Debug \
      "$@"
