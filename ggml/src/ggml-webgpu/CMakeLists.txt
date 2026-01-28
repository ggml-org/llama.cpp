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
            --input_dir "${SHADER_DIR}"
            --output_file "${SHADER_HEADER}"
    DEPENDS ${WGSL_SHADER_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/wgsl-shaders/embed_wgsl.py
    VERBATIM
)

add_custom_target(generate_shaders DEPENDS ${SHADER_HEADER})

# Build the binary shader generator tool (uses pre_wgsl to preprocess binary.wgsl)
set(BINARY_SHADER_HEADER "${SHADER_OUTPUT_DIR}/ggml-wgsl-binary-shaders.hpp")

add_executable(gen_binary_shaders ${CMAKE_CURRENT_SOURCE_DIR}/gen_binary_shaders.cpp)
target_include_directories(gen_binary_shaders PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(gen_binary_shaders PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

add_custom_command(
    OUTPUT ${BINARY_SHADER_HEADER}
    COMMAND ${CMAKE_COMMAND} -E echo "Generating binary op shaders via pre_wgsl"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
    COMMAND gen_binary_shaders
        "${SHADER_DIR}/binary.wgsl"
        "${BINARY_SHADER_HEADER}"
    DEPENDS gen_binary_shaders ${SHADER_DIR}/binary.wgsl
    VERBATIM
)

add_custom_target(generate_binary_shaders DEPENDS ${BINARY_SHADER_HEADER})

ggml_add_backend_library(ggml-webgpu
    ggml-webgpu.cpp
    ${SHADER_HEADER}
    ${BINARY_SHADER_HEADER}
    ../../include/ggml-webgpu.h
)

add_dependencies(ggml-webgpu generate_shaders generate_binary_shaders)

if(EMSCRIPTEN)
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
