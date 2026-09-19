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
file(GLOB   SRCS "template-instances/fattn-tile*.cpp")
list(APPEND GGML_SOURCES_SYCL ${SRCS})
file(GLOB   SRCS "template-instances/fattn-vec*.cpp")
list(APPEND GGML_SOURCES_SYCL ${SRCS})

target_sources(ggml-sycl PRIVATE ${GGML_HEADERS_SYCL} ${GGML_SOURCES_SYCL})

if (WIN32)
    # To generate a Visual Studio solution, using Intel C++ Compiler for ggml-sycl is mandatory
    if( ${CMAKE_GENERATOR} MATCHES "Visual Studio" AND NOT (${CMAKE_GENERATOR_TOOLSET} MATCHES "Intel C"))
        set_target_properties(ggml-sycl PROPERTIES VS_PLATFORM_TOOLSET "Intel C++ Compiler 2025")
        set(CMAKE_CXX_COMPILER "icx")
        set(CMAKE_CXX_COMPILER_ID "IntelLLVM")
    endif()
    # Level Zero SDK path for Windows (only when GGML_SYCL_SUPPORT_LEVEL_ZERO_API is enabled)
    if(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        if(DEFINED ENV{LEVEL_ZERO_V1_SDK_PATH})
            set(LEVEL_ZERO_V1_SDK_PATH $ENV{LEVEL_ZERO_V1_SDK_PATH})
            if(EXISTS "${LEVEL_ZERO_V1_SDK_PATH}")
                target_include_directories(ggml-sycl PRIVATE "${LEVEL_ZERO_V1_SDK_PATH}/include")
                set(LEVEL_ZERO_V1_SDK_LIB_PATH "${LEVEL_ZERO_V1_SDK_PATH}/lib")
            else()
                message(WARNING "LEVEL_ZERO_V1_SDK_PATH set but folder not found: ${LEVEL_ZERO_V1_SDK_PATH}")
            endif()
        endif()
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

message(STATUS "GGML_SYCL_SUPPORT_LEVEL_ZERO_API ${GGML_SYCL_SUPPORT_LEVEL_ZERO_API}")
if (GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
    # Link against Level Zero loader for direct device memory allocation.
    # Avoids sycl::malloc_device triggering DMA-buf/TTM system RAM staging
    # in the xe kernel driver during multi-GPU inference.
    find_path(LEVEL_ZERO_DEV_INCLUDE_DIR level_zero/ze_api.h HINTS ${ONEAPI_ROOT}/include ${LEVEL_ZERO_V1_SDK_PATH}/include)
    find_library(ZE_LOADER_LIB ze_loader HINTS ${ONEAPI_ROOT}/lib ${LEVEL_ZERO_V1_SDK_LIB_PATH} ENV LD_LIBRARY_PATH)
    if(ZE_LOADER_LIB AND LEVEL_ZERO_DEV_INCLUDE_DIR)
        target_link_libraries(ggml-sycl PRIVATE ${ZE_LOADER_LIB})
        target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        message(STATUS "Level Zero loader found: ${ZE_LOADER_LIB}")
        message(STATUS "Level Zero development headers found: ${LEVEL_ZERO_DEV_INCLUDE_DIR}")
    else()
        message(WARNING "Level Zero loader or development headers not found, "
                        "Level Zero API support disabled. "
                        "Please install the Level Zero SDK/development package "
                        "to support Level Zero API features. "
                        "Level Zero API is not mandatory for SYCL backend, "
                        "but it is required by the special features for better "
                        "function & performance on Intel GPUs.")
    endif()
endif()

# Link against oneDNN
set(GGML_SYCL_DNNL 0)
if(GGML_SYCL_DNN)
    # oneAPI installs dnnl-config.cmake under a versioned path CMake does not
    # search by default. When the user did not point CMake at it explicitly
    # (DNNL_DIR), auto-detect the oneAPI-bundled GPU build via the DNNLROOT
    # environment variable or the standard oneAPI install prefixes. A shell
    # with setvars.sh sourced already exports CMAKE_PREFIX_PATH and finds it
    # without any of this.
    if(NOT DEFINED DNNL_DIR)
        if(DEFINED ENV{DNNLROOT} AND EXISTS "$ENV{DNNLROOT}/lib/cmake/dnnl/dnnl-config.cmake")
            set(DNNL_DIR "$ENV{DNNLROOT}/lib/cmake/dnnl")
        elseif(EXISTS "/opt/intel/oneapi/dnnl/latest/lib/cmake/dnnl/dnnl-config.cmake")
            set(DNNL_DIR "/opt/intel/oneapi/dnnl/latest/lib/cmake/dnnl")
        elseif(EXISTS "/usr/intel/oneapi/dnnl/latest/lib/cmake/dnnl/dnnl-config.cmake")
            set(DNNL_DIR "/usr/intel/oneapi/dnnl/latest/lib/cmake/dnnl")
        endif()
        if(DEFINED DNNL_DIR)
            message(STATUS "oneDNN: auto-detected config at ${DNNL_DIR}")
        endif()
    endif()
    find_package(DNNL)
    if(NOT DNNL_FOUND)
        # GGML_SYCL_DNN defaults to ON, so a plain build used to continue
        # silently without oneDNN: every flash-attention then runs the much
        # slower MKL-FA path (20-40% prefill loss on long prompts). Fail
        # loudly instead of producing a degraded binary.
        message(FATAL_ERROR
            "oneDNN (DNNL) was requested (GGML_SYCL_DNN=ON) but not found. "
            "The SYCL backend needs the oneAPI-bundled GPU build: install "
            "intel-oneapi-dnnl and intel-oneapi-dnnl-devel from the oneAPI "
            "repo, source setvars.sh, or pass "
            "-DDNNL_DIR=<dnnl-prefix>/lib/cmake/dnnl (or export DNNLROOT). "
            "To build without oneDNN on purpose, pass -DGGML_SYCL_DNN=OFF.")
    endif()
    if(NOT DEFINED DNNL_GPU_VENDOR)
        # A DNNL package that does not declare DNNL_GPU_VENDOR has no GPU/SYCL
        # support compiled in (e.g. Debian/Ubuntu's CPU-only libdnnl-dev).
        # The old code defaulted the vendor to INTEL here, which fake-passed
        # the target check below and silently linked a CPU-only library into
        # the SYCL backend.
        message(FATAL_ERROR
            "Found oneDNN at ${DNNL_DIR} but it does not declare "
            "DNNL_GPU_VENDOR - it is a CPU-only build (Debian/Ubuntu "
            "libdnnl-dev?). Install the oneAPI-bundled GPU build "
            "(intel-oneapi-dnnl-devel) instead.")
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
        message(FATAL_ERROR
            "oneDNN must be compiled for the same target as llama.cpp: "
            "llama.cpp: ${GGML_SYCL_TARGET}, oneDNN: ${DNNL_GPU_VENDOR}. "
            "Pass -DGGML_SYCL_DNN=OFF to build without oneDNN.")
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
    if (NOT GGML_SYCL_DEVICE_ARCH)
        target_link_options(ggml-sycl PRIVATE -Xs -ze-intel-greater-than-4GB-buffer-required)
    else()
        message(STATUS "Skipping -ze-intel-greater-than-4GB-buffer-required for spir64_gen AOT")
    endif()

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
    message(STATUS "find GGML_SYCL_GRAPH")
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_GRAPH)
endif()

if (GGML_SYCL_HOST_MEM_FALLBACK)
    message(STATUS "find GGML_SYCL_HOST_MEM_FALLBACK")
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_HOST_MEM_FALLBACK)
endif()

if (GGML_SYCL_DEVICE_ARCH)
    message(STATUS "GGML_SYCL_DEVICE_ARCH=${GGML_SYCL_DEVICE_ARCH} (AOT via spir64_gen)")
    target_compile_options(
        ggml-sycl PRIVATE
        -fsycl-targets=spir64_gen
        "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${GGML_SYCL_DEVICE_ARCH}\""
    )

    # Pass through parallel job (process) count for parallelising the
    # `llvm-foreach -- ocloc` invocation for compiling AOT device images.
    include(ProcessorCount)
    ProcessorCount(_ggml_sycl_nproc)
    if (_ggml_sycl_nproc LESS 1)
        set(_ggml_sycl_nproc 1)
    endif()
    set(GGML_SYCL_MAX_PARALLEL_LINK_JOBS ${_ggml_sycl_nproc} CACHE STRING
        "Parallel ocloc jobs for spir64_gen AOT device-image lowering")
    target_link_options(
        ggml-sycl PRIVATE
        -fsycl-targets=spir64_gen
        "SHELL:-Xsycl-target-backend=spir64_gen \"-device ${GGML_SYCL_DEVICE_ARCH}\""
        -fsycl-max-parallel-link-jobs=${GGML_SYCL_MAX_PARALLEL_LINK_JOBS}
    )
endif()
