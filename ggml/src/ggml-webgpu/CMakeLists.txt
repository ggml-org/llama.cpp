cmake_minimum_required(VERSION 3.13)

find_package(Dawn REQUIRED)

ggml_add_backend_library(ggml-webgpu
                         ggml-webgpu.cpp
                         ../../include/ggml-webgpu.h
                        )

if (GGML_WEBGPU_DEBUG)
    message(STATUS "GGML_WEBGPU_DEBUG is ON")
    target_compile_definitions(ggml-webgpu PRIVATE GGML_WEBGPU_DEBUG=1)
endif()

target_link_libraries(ggml-webgpu PRIVATE dawn::webgpu_dawn)

