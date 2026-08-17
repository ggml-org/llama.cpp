set(TARGET vendor-hash)

llama_add_compile_flags()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(${TARGET} STATIC
    xxhash/xxhash.c
    sha1/sha1.c
    sha256/sha256.c
)

# disable warnings in 3rd party code
if (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${TARGET} PRIVATE /w)
else()
    target_compile_options(${TARGET} PRIVATE -w)
endif()

# sha256.c includes "rotate-bits/rotate-bits.h", so consumers get this dir too
target_include_directories(${TARGET} PUBLIC .)
