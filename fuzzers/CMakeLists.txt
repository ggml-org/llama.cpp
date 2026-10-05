add_executable(fuzz-gguf fuzz_gguf.cpp)
target_link_libraries(fuzz-gguf PRIVATE ggml-base)

target_compile_options(ggml-base PRIVATE
    -fsanitize=fuzzer-no-link,address,undefined
    -fno-sanitize-recover=undefined
    -fno-omit-frame-pointer)
target_compile_options(fuzz-gguf PRIVATE
    -fsanitize=fuzzer,address,undefined
    -fno-sanitize-recover=undefined
    -fno-omit-frame-pointer)
target_link_options(fuzz-gguf PRIVATE -fsanitize=fuzzer,address,undefined)
