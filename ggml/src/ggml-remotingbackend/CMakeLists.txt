cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0114 NEW)

message(STATUS "Enable API Remoting backend")

ggml_add_backend_library(ggml-remotingbackend
                         backend.cpp
                         backend-dispatched.cpp
                         backend-dispatched-backend.cpp
                         backend-dispatched-device.cpp
                         backend-dispatched-buffer.cpp
                         backend-dispatched-buffer-type.cpp
                         backend-dispatched-metal.cpp
                         backend-utils.cpp
                         shared/api_remoting.h
                         shared/apir_backend.h
                         shared/venus_cs.h
                         venus_cs_ggml-rpc-back.cpp
                        )

target_compile_options(ggml-remotingbackend PRIVATE -std=c++20)
