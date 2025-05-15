#!/bin/bash

cp -rpv ../ggml/CMakeLists.txt       ./ggml/CMakeLists.txt
cp -rpv ../ggml/src/CMakeLists.txt   ./ggml/src/CMakeLists.txt
cp -rpv ../ggml/cmake/FindSIMD.cmake ./ggml/cmake/FindSIMD.cmake

cp -rpv ../ggml/src/ggml.c              ./ggml/src/ggml.c
cp -rpv ../ggml/src/ggml-aarch64.c      ./ggml/src/ggml-aarch64.c
cp -rpv ../ggml/src/ggml-aarch64.h      ./ggml/src/ggml-aarch64.h
cp -rpv ../ggml/src/ggml-alloc.c        ./ggml/src/ggml-alloc.c
cp -rpv ../ggml/src/ggml-backend-impl.h ./ggml/src/ggml-backend-impl.h
cp -rpv ../ggml/src/ggml-backend.cpp    ./ggml/src/ggml-backend.cpp
cp -rpv ../ggml/src/ggml-cann/*         ./ggml/src/ggml-cann/
cp -rpv ../ggml/src/ggml-cann.cpp       ./ggml/src/ggml-cann.cpp
cp -rpv ../ggml/src/ggml-common.h       ./ggml/src/ggml-common.h
cp -rpv ../ggml/src/ggml-cuda/*         ./ggml/src/ggml-cuda/
cp -rpv ../ggml/src/ggml-cuda.cu        ./ggml/src/ggml-cuda.cu
cp -rpv ../ggml/src/ggml-impl.h         ./ggml/src/ggml-impl.h
cp -rpv ../ggml/src/ggml-kompute.cpp    ./ggml/src/ggml-kompute.cpp
cp -rpv ../ggml/src/ggml-metal.m        ./ggml/src/ggml-metal.m
cp -rpv ../ggml/src/ggml-metal.metal    ./ggml/src/ggml-metal.metal
cp -rpv ../ggml/src/ggml-quants.c       ./ggml/src/ggml-quants.c
cp -rpv ../ggml/src/ggml-quants.h       ./ggml/src/ggml-quants.h
cp -rpv ../ggml/src/ggml-rpc.cpp        ./ggml/src/ggml-rpc.cpp
cp -rpv ../ggml/src/ggml-sycl/*         ./ggml/src/ggml-sycl/
cp -rpv ../ggml/src/ggml-sycl.cpp       ./ggml/src/ggml-sycl.cpp
cp -rpv ../ggml/src/ggml-vulkan.cpp     ./ggml/src/ggml-vulkan.cpp
cp -rpv ../ggml/src/vulkan-shaders/*    ./ggml/src/vulkan-shaders/

cp -rpv ../ggml/include/ggml.h         ./ggml/include/ggml.h
cp -rpv ../ggml/include/ggml-alloc.h   ./ggml/include/ggml-alloc.h
cp -rpv ../ggml/include/ggml-backend.h ./ggml/include/ggml-backend.h
cp -rpv ../ggml/include/ggml-blas.h    ./ggml/include/ggml-blas.h
cp -rpv ../ggml/include/ggml-cann.h    ./ggml/include/ggml-cann.h
cp -rpv ../ggml/include/ggml-cuda.h    ./ggml/include/ggml-cuda.h
cp -rpv ../ggml/include/ggml-kompute.h ./ggml/include/ggml-kompute.h
cp -rpv ../ggml/include/ggml-metal.h   ./ggml/include/ggml-metal.h
cp -rpv ../ggml/include/ggml-rpc.h     ./ggml/include/ggml-rpc.h
cp -rpv ../ggml/include/ggml-sycl.h    ./ggml/include/ggml-sycl.h
cp -rpv ../ggml/include/ggml-vulkan.h  ./ggml/include/ggml-vulkan.h

cp -rpv ../ggml/tests/test-opt.cpp           ./tests/test-opt.cpp
cp -rpv ../ggml/tests/test-grad0.cpp         ./tests/test-grad0.cpp
cp -rpv ../ggml/tests/test-quantize-fns.cpp  ./tests/test-quantize-fns.cpp
cp -rpv ../ggml/tests/test-quantize-perf.cpp ./tests/test-quantize-perf.cpp
cp -rpv ../ggml/tests/test-backend-ops.cpp   ./tests/test-backend-ops.cpp

cp -rpv ../LICENSE                     ./LICENSE
cp -rpv ../ggml/scripts/gen-authors.sh ./scripts/gen-authors.sh
