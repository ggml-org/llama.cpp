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

# Dawn needs C++20 (https://dawn.googlesource.com/dawn/+/refs/heads/main/docs/quickstart-cmake.md#prerequisites)
target_compile_features(ggml-webgpu PRIVATE cxx_std_20)

# Disable C++20 module scanning since emscan-deps fails to find webgpu_cpp.h
set_target_properties(ggml-webgpu PROPERTIES CXX_SCAN_FOR_MODULES OFF)

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
    set(GGML_WEBGPU_WEBKIT_FRAMEWORK "" CACHE PATH "Path to a built WebKit WebGPU.framework")
    if (GGML_WEBGPU_WEBKIT_FRAMEWORK)
        get_filename_component(WEBKIT_PRODUCTS_DIR "${GGML_WEBGPU_WEBKIT_FRAMEWORK}" DIRECTORY)
        add_library(ggml-webgpu-webkit-shim STATIC
            webkit-shim/webgpu-dawn.cpp
            webkit-shim/webgpu-webkit.cpp
        )
        set_source_files_properties(webkit-shim/webgpu-webkit.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-prototypes")
        target_compile_features(ggml-webgpu-webkit-shim PRIVATE cxx_std_23)
        target_include_directories(ggml-webgpu-webkit-shim SYSTEM BEFORE PRIVATE
            "${GGML_WEBGPU_WEBKIT_FRAMEWORK}/PrivateHeaders"
            "${WEBKIT_PRODUCTS_DIR}/usr/local/include"
        )
        target_link_libraries(ggml-webgpu-webkit-shim PRIVATE
            dawn::dawn_public_config
            "${GGML_WEBGPU_WEBKIT_FRAMEWORK}/WebGPU"
            "${WEBKIT_PRODUCTS_DIR}/JavaScriptCore.framework/JavaScriptCore"
        )
        set_property(TARGET ggml-webgpu-webkit-shim PROPERTY NO_SYSTEM_FROM_IMPORTED TRUE)
        set_property(TARGET ggml-webgpu PROPERTY NO_SYSTEM_FROM_IMPORTED TRUE)
        target_compile_definitions(ggml-webgpu PRIVATE GGML_WEBGPU_WEBKIT=1)
        target_link_libraries(ggml-webgpu PRIVATE dawn::dawn_public_config)
        set(DawnWebGPU_TARGET ggml-webgpu-webkit-shim)
    else()
        set(DawnWebGPU_TARGET dawn::webgpu_dawn)
    endif()
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
