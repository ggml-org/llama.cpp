cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0114 NEW)

# function(detect_host_compiler)
#     find_program(HOST_C_COMPILER NAMES gcc clang NO_CMAKE_FIND_ROOT_PATH)
#     find_program(HOST_CXX_COMPILER NAMES g++ clang++ NO_CMAKE_FIND_ROOT_PATH)

#     set(HOST_C_COMPILER "${HOST_C_COMPILER}" PARENT_SCOPE)
#     set(HOST_CXX_COMPILER "${HOST_CXX_COMPILER}" PARENT_SCOPE)
# endfunction()

message(STATUS "Enable API Remoting frontend found")

ggml_add_backend_library(ggml-remotingfrontend
                         ggml-backend-buffer.cpp
                         ggml-backend.cpp
                         ggml-backend-device.cpp
                         ggml-backend-reg.cpp
                         ggml-buffer-type.cpp
                         ggml-host-buffer-type.cpp
                         virtgpu.cpp
                         virtgpu-shm.cpp
                         virtgpu-utils.cpp
                         ../../include/ggml-remoting-frontend.h
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
