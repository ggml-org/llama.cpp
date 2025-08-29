cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0114 NEW)

message(STATUS "Enable API Remoting frontend")

ggml_add_backend_library(ggml-remotingfrontend
                         ggml-backend-buffer.cpp
                         ggml-backend.cpp
                         ggml-backend-device.cpp
                         ggml-backend-reg.cpp
                         ggml-backend-buffer-type.cpp
                         ggml-backend-host-buffer-type.cpp
                         ggml-metal-remoting.cpp
                         virtgpu.cpp
                         virtgpu-shm.cpp
                         virtgpu-utils.cpp
                         virtgpu-forward-device.cpp
                         virtgpu-forward-buffer-type.cpp
                         virtgpu-forward-buffer.cpp
                         virtgpu-forward-backend.cpp
                         virtgpu-forward-metal.cpp
                         virtgpu-forward-impl.h
                         ../../include/ggml-remoting-frontend.h
                         venus_cs_ggml-rpc-front.cpp
                        )

# dnf install -y libdrm-devel
target_link_libraries(ggml-remotingfrontend PUBLIC drm)
target_include_directories(ggml-remotingfrontend PUBLIC /usr/include/libdrm/)
target_include_directories(ggml-remotingfrontend PUBLIC ./include)

target_include_directories(ggml-remotingfrontend PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_compile_options(ggml-remotingfrontend PRIVATE -std=c++20)
