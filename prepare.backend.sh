cmake -S . -B ../build.remoting-backend-prod \
      -DGGML_REMOTINGBACKEND=ON \
      -DGGML_NATIVE=OFF \
      "$@"

#      -DCMAKE_BUILD_TYPE=Debug \
