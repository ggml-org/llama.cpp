cmake_minimum_required(VERSION 3.13)

find_package(Python3 REQUIRED)

# Shader locations
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/wgsl-shaders")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(SHADER_HEADER "${SHADER_OUTPUT_DIR}/ggml-wgsl-shaders.hpp")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

message(STATUS "Shader output dir: ${SHADER_OUTPUT_DIR}")

# Find all WGSL sources
file(GLOB WGSL_SHADER_FILES
    "${SHADER_DIR}/*.wgsl"
    "${SHADER_DIR}/*.tmpl"
)

# Generate the header using a Python script
add_custom_command(
    OUTPUT ${SHADER_HEADER}
    COMMAND ${CMAKE_COMMAND} -E echo "Embedding WGSL shaders to ggml-wgsl-shaders.hpp"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONIOENCODING=utf-8
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/wgsl-shaders/embed_wgsl.py
            --input_dir "${SHADER_DIR}"
            --output_file "${SHADER_HEADER}"
    DEPENDS ${WGSL_SHADER_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/wgsl-shaders/embed_wgsl.py
    VERBATIM
)

add_custom_target(generate_shaders DEPENDS ${SHADER_HEADER})

ggml_add_backend_library(ggml-webgpu
    ggml-webgpu.cpp
    ${SHADER_HEADER}
    ../../include/ggml-webgpu.h
)

add_dependencies(ggml-webgpu generate_shaders)

option(GGML_WEBGPU_HEADER_ONLY
    "Build ggml-webgpu against a caller-supplied webgpu.h + webgpu_cpp.h \
without linking Dawn or emdawnwebgpu. Intended for wasm32-wasip2 targets \
that provide the WebGPU surface at link time (e.g. cognition-webgpu-c-shim). \
The caller MUST set GGML_WEBGPU_HEADER_ONLY_INCLUDE_DIR to a directory \
containing both webgpu/webgpu.h and webgpu/webgpu_cpp.h."
    OFF)

if(GGML_WEBGPU_HEADER_ONLY)
    # wasm32-wasip2 / cognition path: no Dawn, no emdawnwebgpu, no
    # emscripten flags.  ggml-webgpu.cpp compiles against the
    # caller-supplied webgpu.h + webgpu_cpp.h and expects every
    # `wgpu*` symbol to be provided at link time by an external
    # component (the cognition-webgpu-c-shim in this project).
    if(NOT GGML_WEBGPU_HEADER_ONLY_INCLUDE_DIR)
        message(FATAL_ERROR
            "GGML_WEBGPU_HEADER_ONLY=ON requires GGML_WEBGPU_HEADER_ONLY_INCLUDE_DIR \
to point at a directory holding webgpu/webgpu.h and webgpu/webgpu_cpp.h.")
    endif()
    target_include_directories(ggml-webgpu PRIVATE ${GGML_WEBGPU_HEADER_ONLY_INCLUDE_DIR})
    target_compile_definitions(ggml-webgpu PRIVATE GGML_WEBGPU_HEADER_ONLY=1)
    # Sentinel target — nothing to link against.  Downstream users
    # of ggml-webgpu still reference DawnWebGPU_TARGET via
    # target_link_libraries(... PRIVATE ${DawnWebGPU_TARGET}); a
    # skipped find_package() would leave that variable undefined
    # and the link-libraries call would silently drop the symbol.
    # We keep the variable non-empty but pointing at a shim
    # interface library that carries only the header-only include.
    add_library(ggml_webgpu_header_only INTERFACE)
    target_include_directories(ggml_webgpu_header_only
        INTERFACE ${GGML_WEBGPU_HEADER_ONLY_INCLUDE_DIR})
    set(DawnWebGPU_TARGET ggml_webgpu_header_only)
elseif(EMSCRIPTEN)
    set(EMDAWNWEBGPU_DIR "" CACHE PATH "Path to emdawnwebgpu_pkg")

    if(NOT EMDAWNWEBGPU_DIR)
        # default built-in port
        target_compile_options(ggml-webgpu PRIVATE "--use-port=emdawnwebgpu")
        target_link_options(ggml-webgpu INTERFACE "--use-port=emdawnwebgpu")
    else()
        # custom port
        target_compile_options(ggml-webgpu PRIVATE "--use-port=${EMDAWNWEBGPU_DIR}/emdawnwebgpu.port.py")
        target_link_options(ggml-webgpu INTERFACE "--use-port=${EMDAWNWEBGPU_DIR}/emdawnwebgpu.port.py")
    endif()

    if (GGML_WEBGPU_JSPI)
        target_compile_options(ggml-webgpu PRIVATE "-fwasm-exceptions")
        target_link_options(ggml-webgpu INTERFACE "-sJSPI" "-fwasm-exceptions")
    else()
        target_compile_options(ggml-webgpu PRIVATE "-fexceptions")
        target_link_options(ggml-webgpu INTERFACE "-sASYNCIFY" "-exceptions")
    endif()
else()
    find_package(Dawn REQUIRED)
    set(DawnWebGPU_TARGET dawn::webgpu_dawn)
endif()

if (GGML_WEBGPU_DEBUG)
    target_compile_definitions(ggml-webgpu PRIVATE GGML_WEBGPU_DEBUG=1)
    if(EMSCRIPTEN)
        target_link_options(ggml-webgpu INTERFACE "-sASSERTIONS=2")
    endif()
endif()

if (GGML_WEBGPU_CPU_PROFILE)
    target_compile_definitions(ggml-webgpu PRIVATE GGML_WEBGPU_CPU_PROFILE=1)
endif()

if (GGML_WEBGPU_GPU_PROFILE)
    target_compile_definitions(ggml-webgpu PRIVATE GGML_WEBGPU_GPU_PROFILE=1)
endif()

target_include_directories(ggml-webgpu PRIVATE ${SHADER_OUTPUT_DIR})
target_link_libraries(ggml-webgpu PRIVATE ${DawnWebGPU_TARGET})
