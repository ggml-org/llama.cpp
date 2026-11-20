# GGUF fuzzers

Build the GGUF parser fuzzer with Clang and static libraries:

```sh
cmake -S . -B build-fuzz \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLAMA_BUILD_FUZZERS=ON \
    -DLLAMA_BUILD_COMMON=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DBUILD_SHARED_LIBS=OFF
cmake --build build-fuzz --target fuzz-gguf
```

Run it with the included minimal GGUF seed:

```sh
./build-fuzz/bin/fuzz-gguf fuzzers/corpus -max_len=65536
```

The seed has one string metadata entry and no tensors. The fuzzer mutates it and parses each input from memory without allocating tensor data. The build instruments `ggml-base`, including `gguf.cpp`, with libFuzzer coverage, AddressSanitizer, and UndefinedBehaviorSanitizer.
