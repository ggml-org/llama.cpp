set(TARGET llama-verifiable-inference)
add_executable(${TARGET} verifiable-inference.cpp)
install(TARGETS ${TARGET} RUNTIME)

# clibs dependencies (reuse the vendored sha256 used by gguf-hash)
include_directories(../gguf-hash/deps/)
add_library(vi_sha256 OBJECT ../gguf-hash/deps/sha256/sha256.c ../gguf-hash/deps/sha256/sha256.h)
target_link_libraries(${TARGET} PRIVATE vi_sha256)

target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
