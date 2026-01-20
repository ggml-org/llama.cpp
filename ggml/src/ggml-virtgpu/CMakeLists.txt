cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0114 NEW)

message(STATUS "Including the VirtGPU/Virglrenderer API Remoting")

if (NOT GGML_VIRTGPU_BACKEND STREQUAL "ONLY")
    message(STATUS "Enable the VirtGPU/Virglrenderer API Remoting frontend libraryf")

    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DRM REQUIRED libdrm)
    if (NOT GGML_BACKEND_DL)
      # cannot simply use USE_VIRTGPU, as in the 'else()' case the
      # frontend isn't compiled
      target_compile_definitions(ggml PUBLIC "GGML_USE_VIRTGPU_FRONTEND")
    endif()

    ggml_add_backend_library(ggml-virtgpu
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
                             ../../include/ggml-virtgpu.h)

    target_include_directories(ggml-virtgpu PUBLIC /usr/include/libdrm/)

    target_link_libraries(ggml-virtgpu PUBLIC ${DRM_LIBRARIES})
    target_include_directories(ggml-virtgpu PUBLIC ${DRM_INCLUDE_DIRS})
    target_compile_options(ggml-virtgpu PUBLIC ${DRM_CFLAGS_OTHER})

    target_include_directories(ggml-virtgpu PUBLIC ./include)
    target_include_directories(ggml-virtgpu PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    target_compile_options(ggml-virtgpu PRIVATE -std=c++20)
else()
    message(STATUS "Not building the VirtGPU/Virglrenderer API Remoting frontend library")
endif()

if (NOT GGML_VIRTGPU_BACKEND STREQUAL "OFF")
    add_subdirectory("backend")
endif()
