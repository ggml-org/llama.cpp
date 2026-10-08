cmake_minimum_required(VERSION 3.16)
get_filename_component(source "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(fixture "${CMAKE_CURRENT_BINARY_DIR}/sycl-build-metadata-fixture")
file(MAKE_DIRECTORY "${fixture}")
file(WRITE "${fixture}/ggml-config.cmake" "add_library(ggml::ggml INTERFACE IMPORTED)\n")
file(WRITE "${fixture}/CMakeLists.txt" "
cmake_minimum_required(VERSION 3.16)
project(metadata-fixture C CXX)
if (NOT LLAMA_USE_SYSTEM_GGML)
    add_library(ggml INTERFACE IMPORTED)
endif()
add_subdirectory(\"${source}\" llama)
")

# External ggml skips ggml/CMakeLists.txt, so normalized arch may be unavailable.
foreach(provider parent system)
    set(use_system OFF)
    if (provider STREQUAL "system")
        set(use_system ON)
    endif()
    foreach(mode JIT AOT normalized-empty)
        set(arch "acm-g10")
        set(expected AOT)
        set(extra -UGGML_SYCL_DEVICE_ARCH_NORMALIZED)
        if (mode STREQUAL "JIT")
            set(arch "")
            set(expected JIT)
        elseif(mode STREQUAL "normalized-empty")
            # An explicitly empty normalized value must override a nonempty raw value.
            set(extra -DGGML_SYCL_DEVICE_ARCH_NORMALIZED=)
            set(expected JIT)
        endif()
        set(build "${fixture}/${provider}-${mode}")
        execute_process(COMMAND "${CMAKE_COMMAND}" -S "${fixture}" -B "${build}"
            "-DLLAMA_USE_SYSTEM_GGML=${use_system}"
            "-Dggml_DIR=${fixture}" "-DGGML_SYCL_DEVICE_ARCH=${arch}" ${extra}
            -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF
            -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF
            RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err TIMEOUT 60)
        if (NOT rc EQUAL 0)
            message(FATAL_ERROR "${provider}/${mode} configure failed: ${out}${err}")
        endif()
        file(READ "${build}/llama/build-metadata.json" metadata)
        if (NOT metadata MATCHES "\"build_mode\": \"${expected}\"")
            message(FATAL_ERROR "${provider}/${mode}: expected ${expected}: ${metadata}")
        endif()
    endforeach()
endforeach()
message(STATUS "External ggml SYCL build metadata passed")
