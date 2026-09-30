message(STATUS  "GGML_SYCL_TARGET=${GGML_SYCL_TARGET}")

if (NOT GGML_SYCL_TARGET MATCHES "^(INTEL)$")
    message(FATAL_ERROR "GGML_SYCL_TARGET: Invalid target, the supported options are [INTEL]")
endif()

# CMake build tree wants and can work with only one C++ compiler, but
# - ggml-sycl wants -fsycl capable compiler
# - ggml-hip wants hipcc
# thanks ggml backends talking only through C API, we can freely build one backend with it's own compiler
# so this branch does exactly that: we build this backend as externalproject, an existing pattern from hexagon backend
if (GGML_SYCL_SEPARATE_BUILD)
    if (CMAKE_GENERATOR MATCHES "^Visual Studio")
        # the nested project inherits the Visual Studio generator and its MSVC toolset, and
        # CMAKE_CXX_COMPILER does not select a toolset there; use Ninja (docs/backend/SYCL.md),
        # or the non-separate build, which already switches ggml-sycl to the Intel toolset
        message(FATAL_ERROR "GGML_SYCL_SEPARATE_BUILD does not support Visual Studio generators, use -G Ninja")
    endif()
    if (NOT GGML_BACKEND_DL)
        # only tested with shared libraries
        message(FATAL_ERROR "GGML_SYCL_SEPARATE_BUILD requires GGML_BACKEND_DL")
    endif()
    unset(GGML_SYCL_SEPARATE_BUILD_CXX_PATH CACHE)
    find_program(GGML_SYCL_SEPARATE_BUILD_CXX_PATH ${GGML_SYCL_SEPARATE_BUILD_CXX})
    if (NOT GGML_SYCL_SEPARATE_BUILD_CXX_PATH)
        message(FATAL_ERROR "GGML_SYCL_SEPARATE_BUILD_CXX: ${GGML_SYCL_SEPARATE_BUILD_CXX} not found")
    endif()

    # select the minimal possible build because we only need one backend
    set(sycl_args
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${GGML_SYCL_SEPARATE_BUILD_CXX_PATH}
        -DCMAKE_SKIP_BUILD_RPATH=ON
        -DBUILD_SHARED_LIBS=ON
        -DGGML_BACKEND_DL=ON
        -DGGML_CPU=OFF
        -DGGML_BUILD_TESTS=OFF
        -DGGML_BUILD_EXAMPLES=OFF
        -DGGML_CCACHE=OFF)
    # no compiler cache in front of the nested icpx build: an sccache launcher has replaced icpx with
    # the host c++ on this fork (AGENTS.md, build gotchas). Empty launchers alone are not enough,
    # because GGML_CCACHE then installs ccache or sccache as a global rule launcher, so it is OFF too.
    list(APPEND sycl_args -DCMAKE_C_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER=)

    # append everything starting with GGML_SYCL to the flags
    get_cmake_property(cache_vars CACHE_VARIABLES)
    foreach (var ${cache_vars})
        if (var MATCHES "^GGML_SYCL" AND NOT var MATCHES "^GGML_SYCL_SEPARATE_BUILD")
            list(APPEND sycl_args -D${var}=${${var}})
        endif()
    endforeach()

    # package discovery and install RPATH settings do not reach the nested configure on their own;
    # this fork's documented recipe supplies them as cache variables (packaging/arch/PKGBUILD).
    # list values are joined with | and split again by LIST_SEPARATOR below
    set(sycl_forward_vars MKL_DIR TBB_DIR IntelSYCL_DIR DNNL_DIR DNNL_GPU_VENDOR CMAKE_PREFIX_PATH
                          CMAKE_INSTALL_RPATH CMAKE_BUILD_WITH_INSTALL_RPATH CMAKE_MAKE_PROGRAM)
    get_property(sycl_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if (sycl_multi_config)
        # custom configurations (e.g. Profile) must exist in the nested project too
        list(APPEND sycl_forward_vars CMAKE_CONFIGURATION_TYPES)
    endif()
    foreach (var ${sycl_forward_vars})
        if (DEFINED ${var})
            string(REPLACE ";" "|" sycl_arg_value "${${var}}")
            list(APPEND sycl_args "-D${var}=${sycl_arg_value}")
        endif()
    endforeach()
    # the nested module is copied into place, never relinked by an install step, so it has to be
    # linked with the install RPATH itself
    if (DEFINED CMAKE_INSTALL_RPATH)
        list(APPEND sycl_args -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON)
    endif()
    # anything else the nested configure needs (list values use | for ;)
    list(APPEND sycl_args ${GGML_SYCL_SEPARATE_BUILD_ARGS})

    # multi-config generators put the module under a per-configuration directory and need
    # --config on the nested build; single-config generators leave both empty
    if (sycl_multi_config)
        set(sycl_cfg_arg --config $<CONFIG>)
        set(sycl_cfg_dir $<CONFIG>/)
    endif()
    set(sycl_lib ${CMAKE_SHARED_MODULE_PREFIX}ggml-sycl${CMAKE_SHARED_MODULE_SUFFIX})
    set(sycl_out ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${sycl_cfg_dir}${sycl_lib})
    # generator expressions in BYPRODUCTS need CMake 3.20; nothing consumes the module as a build
    # input, so older CMake with a multi-config generator only loses it from the clean target
    set(sycl_byproducts ${sycl_out})
    if (sycl_multi_config AND CMAKE_VERSION VERSION_LESS 3.20)
        set(sycl_byproducts "")
    endif()

    # ggml is not built standalone here, so use a wrapper project
    set(sycl_src ${CMAKE_CURRENT_BINARY_DIR}/nested-src)
    file(WRITE ${sycl_src}/CMakeLists.txt
        "cmake_minimum_required(VERSION ${CMAKE_VERSION})\n"
        "project(ggml-sycl-nested C CXX)\n"
        "add_subdirectory(\"${CMAKE_CURRENT_SOURCE_DIR}/../..\" ggml)\n")

    include(ExternalProject)
    ExternalProject_Add(ggml-sycl
        SOURCE_DIR ${sycl_src}
        BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/nested
        LIST_SEPARATOR |
        CMAKE_ARGS ${sycl_args}
        BUILD_COMMAND ${CMAKE_COMMAND} --build . ${sycl_cfg_arg} --target ggml-sycl
        BUILD_ALWAYS ON
        BUILD_BYPRODUCTS ${sycl_byproducts}
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_if_different <BINARY_DIR>/bin/${sycl_cfg_dir}${sycl_lib} ${sycl_out})
    add_dependencies(ggml ggml-sycl)

    if (GGML_BACKEND_DIR)
        install(PROGRAMS ${sycl_out} DESTINATION ${GGML_BACKEND_DIR})
    else()
        install(PROGRAMS ${sycl_out} DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
    return()
endif()

# icpx links temporary objects that are deleted afterwards, so the linker depfile makes ggml-sycl always out of date
set(CMAKE_LINK_DEPENDS_USE_LINKER OFF)

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
list(REMOVE_ITEM GGML_SOURCES_SYCL "${CMAKE_CURRENT_SOURCE_DIR}/sycl-mutable-command-list-probe.cpp")
file(GLOB   SRCS "template-instances/fattn-tile*.cpp")
list(APPEND GGML_SOURCES_SYCL ${SRCS})
file(GLOB   SRCS "template-instances/fattn-vec*.cpp")
list(APPEND GGML_SOURCES_SYCL ${SRCS})

target_sources(ggml-sycl PRIVATE ${GGML_HEADERS_SYCL} ${GGML_SOURCES_SYCL})

# LLAMA_BUILD_TESTS covers this fork's canonical build (a llama.cpp superproject
# build, where GGML_BUILD_TESTS is off by default per docs/); GGML_BUILD_TESTS
# covers a ggml-standalone test build, where LLAMA_BUILD_TESTS doesn't apply.
if (LLAMA_BUILD_TESTS OR GGML_BUILD_TESTS)
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_TESTING)
endif()

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

if (GGML_SYCL_DEVICE_CODE_SPLIT)
    target_compile_options(ggml-sycl PRIVATE "-fsycl-device-code-split=per_kernel")
    target_link_options(ggml-sycl PRIVATE "-fsycl-device-code-split=per_kernel")
endif()

message(STATUS "GGML_SYCL_DEVICE_CODE_SPLIT ${GGML_SYCL_DEVICE_CODE_SPLIT}")

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

if (GGML_SYCL_SUPPORT_LEVEL_ZERO_API AND ZE_LOADER_LIB AND LEVEL_ZERO_INCLUDE_DIR)
    get_filename_component(GGML_SYCL_COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    find_program(GGML_SYCL_OPENCL_CLANG
                 NAMES clang
                 HINTS "${GGML_SYCL_COMPILER_BIN_DIR}/compiler"
                 NO_DEFAULT_PATH)
    find_program(GGML_SYCL_LLVM_SPIRV NAMES llvm-spirv)
    if (GGML_SYCL_OPENCL_CLANG AND GGML_SYCL_LLVM_SPIRV)
        set(MUTABLE_PROBE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/sycl-mutable-command-list-probe.cl")
        set(MUTABLE_PROBE_BITCODE "${CMAKE_CURRENT_BINARY_DIR}/sycl-mutable-command-list-probe.bc")
        set(MUTABLE_PROBE_SPIRV "${CMAKE_CURRENT_BINARY_DIR}/sycl-mutable-command-list-probe.spv")
        add_custom_command(
            OUTPUT "${MUTABLE_PROBE_SPIRV}"
            BYPRODUCTS "${MUTABLE_PROBE_BITCODE}"
            COMMAND "${GGML_SYCL_OPENCL_CLANG}" -cc1 -triple spir64 -cl-std=CL2.0
                    -finclude-default-header -emit-llvm-bc "${MUTABLE_PROBE_SOURCE}"
                    -o "${MUTABLE_PROBE_BITCODE}"
            COMMAND "${GGML_SYCL_LLVM_SPIRV}" "${MUTABLE_PROBE_BITCODE}" -o "${MUTABLE_PROBE_SPIRV}"
            DEPENDS "${MUTABLE_PROBE_SOURCE}"
            VERBATIM)
        add_executable(sycl-mutable-command-list-probe EXCLUDE_FROM_ALL
                       sycl-mutable-command-list-probe.cpp)
        target_include_directories(sycl-mutable-command-list-probe PRIVATE "${LEVEL_ZERO_INCLUDE_DIR}")
        target_link_libraries(sycl-mutable-command-list-probe PRIVATE "${ZE_LOADER_LIB}")
        target_compile_definitions(
            sycl-mutable-command-list-probe
            PRIVATE MUTABLE_PROBE_SPIRV_PATH="${MUTABLE_PROBE_SPIRV}")
        add_custom_target(sycl-mutable-command-list-probe-spirv DEPENDS "${MUTABLE_PROBE_SPIRV}")
        add_dependencies(sycl-mutable-command-list-probe sycl-mutable-command-list-probe-spirv)
    else()
        message(STATUS "Mutable command-list probe disabled: OpenCL clang or llvm-spirv not found")
    endif()
endif()

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

# 256-GRF variants of the flash-attention kernels (option in ggml/CMakeLists.txt, runtime
# knob GGML_SYCL_FA_LARGE_GRF). Off by default: every FA tile kernel would carry a second
# device image, and an AOT build compiles each twice.
if (GGML_SYCL_FA_LARGE_GRF)
    if (GGML_SYCL_DEVICE_ARCH)
        message(WARNING "GGML_SYCL_FA_LARGE_GRF=ON with GGML_SYCL_DEVICE_ARCH=${GGML_SYCL_DEVICE_ARCH}: every flash-attention tile kernel is compiled twice ahead of time; this combination has not been built by the fork")
    endif()
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_FA_LARGE_GRF_VARIANTS=1)
else()
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_FA_LARGE_GRF_VARIANTS=0)
endif()
message(STATUS "GGML_SYCL_FA_LARGE_GRF ${GGML_SYCL_FA_LARGE_GRF}")


if (GGML_SYCL_F16)
    add_compile_definitions(GGML_SYCL_F16)
endif()

option(GGML_SYCL_SCALAR_MMQ "Force scalar MMQ tile sizes (for B70 without XMX kernels)" OFF)
if (GGML_SYCL_SCALAR_MMQ)
    add_compile_definitions(GGML_SYCL_SCALAR_MMQ)
endif()

option(GGML_SYCL_TURBO_QUANT "Enable dispatch to TurboQuant KV cache FA kernels" ON)
if (GGML_SYCL_TURBO_QUANT)
    add_compile_definitions(GGML_SYCL_TURBO_QUANT)
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
