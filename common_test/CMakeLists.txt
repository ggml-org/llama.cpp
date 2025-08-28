# common_test library for load_into_memory.h and uint8-buff-stream.h

set(TARGET llama-common-test)

add_library(${TARGET} INTERFACE)

target_include_directories(${TARGET} INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(${TARGET} INTERFACE LLAMA_COMMON_TEST_HEADERS)

target_compile_features(${TARGET} INTERFACE cxx_std_17)

target_link_libraries(${TARGET} INTERFACE common)
