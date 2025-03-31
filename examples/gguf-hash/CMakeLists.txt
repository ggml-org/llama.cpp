set(TARGET llama-gguf-hash)
add_executable(${TARGET} gguf-hash.cpp)
install(TARGETS ${TARGET} RUNTIME)

# clibs dependencies
include_directories(deps/)

add_library(xxhash OBJECT deps/xxhash/xxhash.c deps/xxhash/xxhash.h)
target_link_libraries(${TARGET} PRIVATE xxhash)

add_library(sha1 OBJECT deps/sha1/sha1.c deps/sha1/sha1.h)
target_link_libraries(${TARGET} PRIVATE sha1)
if (NOT MSVC)
    # disable warnings in 3rd party code
    target_compile_options(sha1 PRIVATE -w)
endif()

add_library(sha256 OBJECT deps/sha256/sha256.c deps/sha256/sha256.h)
target_link_libraries(${TARGET} PRIVATE sha256)

target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
