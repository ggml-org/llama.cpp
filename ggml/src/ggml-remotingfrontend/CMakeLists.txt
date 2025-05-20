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
                         virtgpu.cpp
                         virtgpu-shm.cpp
                         virtgpu-utils.cpp
                         virtgpu-forward-device.cpp
                         virtgpu-forward-buffer-type.cpp
                         virtgpu-forward-buffer.cpp
                         virtgpu-forward-backend.cpp
                         virtgpu-forward-impl.h
                         ../../include/ggml-remoting-frontend.h
                         ../ggml-remotingbackend/shared/venus_cs_ggml-rpc.cpp
                        )

target_link_libraries(ggml-remotingfrontend PUBLIC drm)
target_include_directories(ggml-remotingfrontend PUBLIC /usr/include/libdrm/)

set(REMOTING_PROJECT /Users/kevinpouget/remoting)
set(MESA_PROJECT_HOME ${REMOTING_PROJECT}/mesa)
set(MESA_PROJECT_SRC ${MESA_PROJECT_HOME}/src)

target_include_directories(ggml-remotingfrontend PUBLIC ${MESA_PROJECT_SRC}/virtio/virtio-gpu/)
target_include_directories(ggml-remotingfrontend PUBLIC ${MESA_PROJECT_HOME}/include)
target_include_directories(ggml-remotingfrontend PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_compile_options(ggml-remotingfrontend PRIVATE -std=c++20)

# dnf install -y libdrm-devel
