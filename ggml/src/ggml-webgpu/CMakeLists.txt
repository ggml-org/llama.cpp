cmake_minimum_required(VERSION 3.13)

find_package(Python3 REQUIRED)

# Shader locations
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/wgsl-shaders")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(SHADER_HEADER "${SHADER_OUTPUT_DIR}/ggml-wgsl-shaders.hpp")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

message(STATUS "Shader output dir: ${SHADER_OUTPUT_DIR}")

# Find all WGSL files
file(GLOB WGSL_SHADER_FILES "${SHADER_DIR}/*.wgsl")

# Generate the header using a Python script
add_custom_command(
    OUTPUT ${SHADER_HEADER}
    COMMAND ${CMAKE_COMMAND} -E echo "Embedding WGSL shaders to ggml-wgsl-shaders.hpp"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONIOENCODING=utf-8
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/wgsl-shaders/embed_wgsl.py
            --input "${SHADER_DIR}"
            --output "${SHADER_HEADER}"
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

if(EMSCRIPTEN)
    set(EMDAWNWEBGPU_DIR "" CACHE PATH "Path to emdawnwebgpu_pkg")

    target_compile_options(ggml-webgpu PRIVATE "--use-port=${EMDAWNWEBGPU_DIR}/emdawnwebgpu.port.py")
    target_link_options(ggml-webgpu PRIVATE "--use-port=${EMDAWNWEBGPU_DIR}/emdawnwebgpu.port.py")
else()
    find_package(Dawn REQUIRED)
    set(DawnWebGPU_TARGET dawn::webgpu_dawn)
endif()

if (GGML_WEBGPU_DEBUG)
    target_compile_definitions(ggml-webgpu PRIVATE GGML_WEBGPU_DEBUG=1)
endif()

target_include_directories(ggml-webgpu PRIVATE ${SHADER_OUTPUT_DIR})
target_link_libraries(ggml-webgpu PRIVATE ${DawnWebGPU_TARGET})
