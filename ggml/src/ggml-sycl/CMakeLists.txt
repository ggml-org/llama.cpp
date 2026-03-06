message(STATUS  "GGML_SYCL_TARGET=${GGML_SYCL_TARGET}")

if (NOT GGML_SYCL_TARGET MATCHES "^(INTEL)$")
    message(FATAL_ERROR "GGML_SYCL_TARGET: Invalid target, the supported options are [INTEL]")
endif()

check_cxx_compiler_flag("-fsycl" SUPPORTS_SYCL)

if (DEFINED ENV{ONEAPI_ROOT})
    message(STATUS "Using oneAPI Release SYCL compiler (icpx).")
elseif(SUPPORTS_SYCL)
    message(WARNING "Using open-source SYCL compiler (clang++). Didn't detect ENV {ONEAPI_ROOT}.
        If you expected the oneAPI Release compiler, please install oneAPI & source it, like:
        source /opt/intel/oneapi/setvars.sh")
else()
    message(FATAL_ERROR "C++ compiler lacks SYCL support.")
endif()
message(STATUS "SYCL found")
#todo: AOT

ggml_add_backend_library(ggml-sycl
                         ggml-sycl.cpp
                         ../../include/ggml-sycl.h
                        )

file(GLOB   GGML_HEADERS_SYCL "*.hpp")
file(GLOB   GGML_SOURCES_SYCL "*.cpp")
target_sources(ggml-sycl PRIVATE ${GGML_HEADERS_SYCL} ${GGML_SOURCES_SYCL})

if (WIN32)
    # To generate a Visual Studio solution, using Intel C++ Compiler for ggml-sycl is mandatory
    if( ${CMAKE_GENERATOR} MATCHES "Visual Studio" AND NOT (${CMAKE_GENERATOR_TOOLSET} MATCHES "Intel C"))
        set_target_properties(ggml-sycl PROPERTIES VS_PLATFORM_TOOLSET "Intel C++ Compiler 2025")
        set(CMAKE_CXX_COMPILER "icx")
        set(CMAKE_CXX_COMPILER_ID "IntelLLVM")
    endif()
endif()

macro(detect_and_find_package package_name)
    set(test_source "
    cmake_minimum_required(VERSION ${CMAKE_VERSION})
    project(check_package LANGUAGES CXX)
    find_package(${package_name} QUIET)
    ")

    set(test_dir "${CMAKE_CURRENT_BINARY_DIR}/check_package_${package_name}")
    file(WRITE "${test_dir}/CMakeLists.txt" "${test_source}")

    set(cmake_args "")
    if(CMAKE_GENERATOR)
        list(APPEND cmake_args "-G" "${CMAKE_GENERATOR}")
    endif()
    if(CMAKE_GENERATOR_PLATFORM)
        list(APPEND cmake_args "-A" "${CMAKE_GENERATOR_PLATFORM}")
    endif()
    if(CMAKE_GENERATOR_TOOLSET)
        list(APPEND cmake_args "-T" "${CMAKE_GENERATOR_TOOLSET}")
    endif()
    if(CMAKE_CXX_COMPILER)
        list(APPEND cmake_args "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} ${cmake_args} .
        WORKING_DIRECTORY "${test_dir}"
        RESULT_VARIABLE result
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(result EQUAL 0)
        find_package(${package_name} ${ARGN})
    else()
        message(WARNING "Detection of ${package_name} failed. The package might be broken or incompatible.")
        set(${package_name}_FOUND FALSE)
    endif()
endmacro()

detect_and_find_package(IntelSYCL)
if (IntelSYCL_FOUND)
    # Use oneAPI CMake when possible
    target_link_libraries(ggml-sycl PRIVATE IntelSYCL::SYCL_CXX)
else()
    # Fallback to the simplest way of enabling SYCL when using intel/llvm nightly for instance
    target_compile_options(ggml-sycl PRIVATE "-fsycl")
    target_link_options(ggml-sycl PRIVATE "-fsycl")
endif()

target_compile_options(ggml-sycl PRIVATE "-Wno-narrowing")

# Link against oneDNN
set(GGML_SYCL_DNNL 0)
if(GGML_SYCL_DNN)
    find_package(DNNL)
    if(DNNL_FOUND)
        if (NOT DEFINED DNNL_GPU_VENDOR)
            # default to intel target
            set(DNNL_GPU_VENDOR "INTEL")
            if(NOT "${GGML_SYCL_TARGET}" STREQUAL "INTEL")
                message(WARNING "oneDNN builds bundled with oneapi release only support INTEL target")
            endif()
        endif()

        # Verify oneDNN was compiled for the same target as llama
        if("${GGML_SYCL_TARGET}" STREQUAL "${DNNL_GPU_VENDOR}")
            target_link_libraries(ggml-sycl PRIVATE DNNL::dnnl)
            set(GGML_SYCL_DNNL 1)
            get_target_property(CONFIGS DNNL::dnnl IMPORTED_CONFIGURATIONS)
            foreach(CONFIG ${CONFIGS})
                get_target_property(DNNL_LIB DNNL::dnnl IMPORTED_LOCATION_${CONFIG})
                message(STATUS "Found oneDNN: ${DNNL_LIB}")
            endforeach()
        else()
            message(WARNING
                "oneDNN must be compiled for the same target as llama.cpp.
                 llama.cpp: ${GGML_SYCL_TARGET}, oneDNN: ${DNNL_GPU_VENDOR}.
                 Disabling oneDNN support.")
        endif()
    else()
        message(STATUS "oneDNN not found, disabling oneDNN support")
    endif()
else()
    message(STATUS "oneDNN support disabled by the user")
endif()
target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_DNNL=${GGML_SYCL_DNNL})

if (GGML_SYCL_F16)
    add_compile_definitions(GGML_SYCL_F16)
endif()

if (GGML_SYCL_TARGET STREQUAL "INTEL")
    add_compile_definitions(GGML_SYCL_WARP_SIZE=16)
    target_link_options(ggml-sycl PRIVATE  -Xs   -ze-intel-greater-than-4GB-buffer-required)

    # Link against Intel oneMKL
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(SYCL_COMPILER ON)
    endif()
    find_package(MKL REQUIRED)
    target_link_libraries(ggml-sycl PRIVATE MKL::MKL_SYCL::BLAS)
else()
    # default for other target
    message(FATAL_ERROR "GGML_SYCL_TARGET is not supported")
    add_compile_definitions(GGML_SYCL_WARP_SIZE=32)
endif()

if (GGML_SYCL_GRAPH)
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_GRAPH)
endif()

if (GGML_SYCL_DEVICE_ARCH)
    target_compile_options(ggml-sycl PRIVATE -Xsycl-target-backend --offload-arch=${GGML_SYCL_DEVICE_ARCH})
    target_link_options(ggml-sycl PRIVATE -Xsycl-target-backend --offload-arch=${GGML_SYCL_DEVICE_ARCH})
endif()

