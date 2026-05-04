#!/usr/bin/env bash

cp -rpv ../ggml/CMakeLists.txt       ./ggml/CMakeLists.txt
cp -rpv ../ggml/src/CMakeLists.txt   ./ggml/src/CMakeLists.txt

cp -rpv ../ggml/cmake/* ./ggml/cmake/
cp -rpv ../ggml/src/ggml-cpu/cmake/* ./ggml/src/ggml-cpu/cmake/

cp -rpv ../ggml/src/ggml* ./ggml/src/

cp -rpv ../ggml/include/ggml*.h ./ggml/include/
cp -rpv ../ggml/include/gguf*.h ./ggml/include/

cp -rpv ../ggml/tests/test-opt.cpp           ./tests/test-opt.cpp
cp -rpv ../ggml/tests/test-quantize-fns.cpp  ./tests/test-quantize-fns.cpp
cp -rpv ../ggml/tests/test-quantize-perf.cpp ./tests/test-quantize-perf.cpp
cp -rpv ../ggml/tests/test-backend-ops.cpp   ./tests/test-backend-ops.cpp

cp -rpv ../LICENSE                     ./LICENSE
cp -rpv ../ggml/scripts/gen-authors.sh ./scripts/gen-authors.sh
