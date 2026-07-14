cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0114 NEW)

message(STATUS "Enable the VirtGPU/Virglrenderer backend library")

ggml_add_backend_library(ggml-virtgpu-backend
                         backend.cpp
                         backend-dispatched.cpp
                         backend-dispatched-backend.cpp
                         backend-dispatched-device.cpp
                         backend-dispatched-buffer.cpp
                         backend-dispatched-buffer-type.cpp
                         shared/api_remoting.h
                         shared/apir_backend.h
                         shared/apir_cs.h
                         apir_cs_ggml-rpc-back.cpp)

target_compile_options(ggml-virtgpu-backend PRIVATE -std=c++20)

# Add include directory for ggml-backend-impl.h and other core headers
target_include_directories(ggml-virtgpu-backend PRIVATE ../..)
