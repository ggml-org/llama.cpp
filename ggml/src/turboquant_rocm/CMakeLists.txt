# ─────────────────────────────────────────────────────────────────────────────
# TurboQuant ROCm integration for llama.cpp (Windows Native Optimized)
# ─────────────────────────────────────────────────────────────────────────────

cmake_minimum_required(VERSION 3.21)
project(turboquant_rocm LANGUAGES CXX)

if (LLAMA_TURBOQUANT)
    set(TQ_SOURCES
        turboquant.cpp
        turboquant_hip.hip
        llama_tq_hook.cpp
    )

    add_library(turboquant_rocm STATIC ${TQ_SOURCES})

    # On Windows, treat .hip files as CXX and use -x hip flag.
    # On Linux, use native HIP support if available.
    if (WIN32)
        set_source_files_properties(turboquant_hip.hip PROPERTIES LANGUAGE CXX)
        set_source_files_properties(turboquant_hip.hip PROPERTIES COMPILE_FLAGS "-x hip --offload-arch=gfx1201")
    else()
        set_source_files_properties(turboquant_hip.hip PROPERTIES LANGUAGE HIP)
    endif()

    target_compile_features(turboquant_rocm PUBLIC cxx_std_17)
    target_include_directories(turboquant_rocm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    
    # We must find HIP to link against it
    find_package(hip REQUIRED)
    target_link_libraries(turboquant_rocm PUBLIC hip::device)
    
    target_compile_definitions(turboquant_rocm PRIVATE TQ_BUILD_DLL)
    target_compile_definitions(turboquant_rocm PUBLIC LLAMA_TURBOQUANT)

    message(STATUS "[TQ] Enabled and optimized for gfx1201 (Windows Native)")
endif()
