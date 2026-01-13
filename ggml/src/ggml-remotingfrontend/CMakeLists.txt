cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0114 NEW)

message(STATUS "Enable API Remoting frontend")

find_package(PkgConfig REQUIRED)
pkg_check_modules(DRM REQUIRED libdrm)

ggml_add_backend_library(ggml-remotingfrontend
                         ggml-backend-buffer.cpp
                         ggml-backend.cpp
                         ggml-backend-device.cpp
                         ggml-backend-reg.cpp
                         ggml-backend-buffer-type.cpp
                         virtgpu-apir.h
                         virtgpu-forward.gen.h
                         virtgpu.cpp
                         virtgpu-shm.cpp
                         virtgpu-utils.cpp
                         virtgpu-forward-device.cpp
                         virtgpu-forward-buffer-type.cpp
                         virtgpu-forward-buffer.cpp
                         virtgpu-forward-backend.cpp
                         virtgpu-forward-impl.h
                         apir_cs_ggml-rpc-front.cpp
                         ../../include/ggml-remoting-frontend.h)

target_include_directories(ggml-remotingfrontend PUBLIC /usr/include/libdrm/)

target_link_libraries(ggml-remotingfrontend PUBLIC ${DRM_LIBRARIES})
target_include_directories(ggml-remotingfrontend PUBLIC ${DRM_INCLUDE_DIRS})
target_compile_options(ggml-remotingfrontend PUBLIC ${DRM_CFLAGS_OTHER})

target_include_directories(ggml-remotingfrontend PUBLIC ./include)
target_include_directories(ggml-remotingfrontend PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_compile_options(ggml-remotingfrontend PRIVATE -std=c++20)
