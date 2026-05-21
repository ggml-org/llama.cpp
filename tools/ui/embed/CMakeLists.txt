# Standalone host build of the llama-ui-embed helper
cmake_minimum_required(VERSION 3.16)
project(llama-ui-embed CXX)

add_executable(llama-ui-embed embed.cpp)
target_compile_features(llama-ui-embed PRIVATE cxx_std_17)

set_target_properties(llama-ui-embed PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
