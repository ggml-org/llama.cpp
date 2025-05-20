cmake_minimum_required(VERSION 3.13)

find_package(Dawn REQUIRED)

ggml_add_backend_library(ggml-webgpu
                         ggml-webgpu.cpp
                         ../../include/ggml-webgpu.h
                        )

target_link_libraries(ggml-webgpu PRIVATE dawn::webgpu_dawn)