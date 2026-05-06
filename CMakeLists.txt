cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("llama.cpp" C CXX)
include(CheckIncludeFileCXX)

#set(CMAKE_WARN_DEPRECATED YES)
set(CMAKE_WARN_UNUSED_CLI YES)

if (GGML_TSAVORITE)

    if  (NOT DEFINED GGML_TSAVORITE_TARGET)
        set(GGML_TSAVORITE_TARGET "posix")
    endif()
    if (NOT ${GGML_TSAVORITE_TARGET} STREQUAL fpga)
        set(GGML_TSAVORITE_TARGET "posix")
    endif()

    # ---------------------------------------------------------------------------
    # MLIR_SDK_ARCH — host architecture, used only for locating SDK paths.
    # The cross-compilation target is always aarch64 and is handled separately
    # by arm.cmake via CMAKE_TOOLCHAIN_FILE. These are two distinct concerns:
    #   MLIR_SDK_ARCH  = which SDK binaries to use (must match the build host)
    #   cross target   = what the output binaries run on (always aarch64/FPGA)
    # ---------------------------------------------------------------------------
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE HOST_M
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (HOST_M STREQUAL "x86_64" OR HOST_M STREQUAL "amd64")
        set(MLIR_SDK_ARCH "x86_64")
    elseif (HOST_M STREQUAL "aarch64" OR HOST_M STREQUAL "arm64")
        set(MLIR_SDK_ARCH "aarch64")
    else()
        message(FATAL_ERROR "Unsupported host arch from uname -m: ${HOST_M}")
    endif()
    message(STATUS "Host arch: ${HOST_M} => MLIR_SDK_ARCH=${MLIR_SDK_ARCH}")

    # SDK_VERSION is required. Passed via -DSDK_VERSION=<ver> or env var SDK_VERSION.
    if (NOT DEFINED SDK_VERSION AND DEFINED ENV{SDK_VERSION})
        set(SDK_VERSION "$ENV{SDK_VERSION}")
    endif()
    if (NOT DEFINED SDK_VERSION)
        message(FATAL_ERROR "SDK_VERSION must be provided (e.g. SDK_VERSION=0.4.2 source tsi-pkg-build.sh)")
    endif()

    # ---------------------------------------------------------------------------
    # MLIR_COMPILER_DIR
    # Resolve order:
    #   1) -DMLIR_COMPILER_DIR=<path>  (explicit cmake flag, highest priority)
    #   2) ENV{MLIR_COMPILER_DIR}      (exported from tsi-pkg-build.sh)
    #   3) Derived from SDK_VERSION    (fallback default)
    # ---------------------------------------------------------------------------
    if (NOT DEFINED MLIR_COMPILER_DIR)
        if (DEFINED ENV{MLIR_COMPILER_DIR})
            set(MLIR_COMPILER_DIR "$ENV{MLIR_COMPILER_DIR}")
            message(STATUS "MLIR_COMPILER_DIR from environment: ${MLIR_COMPILER_DIR}")
        else()
            set(MLIR_COMPILER_DIR /proj/rel/sw/tsi-sw/staging/sdk/sdk-r.${SDK_VERSION}/${MLIR_SDK_ARCH}/compiler)
            message(STATUS "MLIR_COMPILER_DIR defaulted to: ${MLIR_COMPILER_DIR}")
        endif()
    endif()

    # ---------------------------------------------------------------------------
    # RUNTIME_DIR
    # Single source of truth for runtime + shim headers and libraries.
    # For FPGA this is the installed runtime directory (install-fpga layout).
    #
    # Resolve order:
    #   1) -DRUNTIME_DIR=<path>        (explicit cmake flag, highest priority)
    #      Use this to point at a local runtime build, e.g.:
    #        -DRUNTIME_DIR=/proj/work/<user>/runtime/install-fpga
    #   2) ENV{RUNTIME_DIR}            (exported from tsi-pkg-build.sh)
    #   3) Derived from SDK_VERSION    (fallback default, SDK-provided runtime)
    #
    # NOTE: When overriding RUNTIME_DIR to a local build, the local install must
    # follow the standard install-fpga layout:
    #   <RUNTIME_DIR>/include/   -- all runtime + shim headers (tsi-rt/shim/ etc.)
    #   <RUNTIME_DIR>/lib/       -- all runtime + shim .so files
    #   <RUNTIME_DIR>/lib/cmake/ -- CMake package files (TSIRuntime, TSICommonRuntime)
    # ---------------------------------------------------------------------------
    if (NOT DEFINED RUNTIME_DIR)
        if (DEFINED ENV{RUNTIME_DIR})
            set(RUNTIME_DIR "$ENV{RUNTIME_DIR}")
            message(STATUS "RUNTIME_DIR from environment: ${RUNTIME_DIR}")
        else()
            if (${GGML_TSAVORITE_TARGET} STREQUAL fpga)
                set(RUNTIME_DIR /proj/rel/sw/tsi-sw/staging/sdk/sdk-r.${SDK_VERSION}/${MLIR_SDK_ARCH}/fpga/runtime)
            else()
                set(RUNTIME_DIR /proj/rel/sw/tsi-sw/staging/sdk/sdk-r.${SDK_VERSION}/${MLIR_SDK_ARCH}/posix/runtime)
            endif()
            message(STATUS "RUNTIME_DIR defaulted to: ${RUNTIME_DIR}")
        endif()
    endif()

    # ---------------------------------------------------------------------------
    # TOOLBOX_DIR
    # Provides the ARM cross-compilation toolchain file (arm.cmake).
    # Always sourced from the SDK — not overridden by the local runtime build.
    #
    # Resolve order:
    #   1) -DTOOLBOX_DIR=<path>        (explicit cmake flag)
    #   2) ENV{TOOLBOX_DIR}            (exported from tsi-pkg-build.sh)
    #   3) Derived from SDK_VERSION    (fallback default)
    # ---------------------------------------------------------------------------
    if (NOT DEFINED TOOLBOX_DIR)
        if (DEFINED ENV{TOOLBOX_DIR})
            set(TOOLBOX_DIR "$ENV{TOOLBOX_DIR}")
            message(STATUS "TOOLBOX_DIR from environment: ${TOOLBOX_DIR}")
        else()
            set(TOOLBOX_DIR /proj/rel/sw/tsi-sw/staging/sdk/sdk-r.${SDK_VERSION}/${MLIR_SDK_ARCH}/toolbox/build/install-fpga)
            message(STATUS "TOOLBOX_DIR defaulted to: ${TOOLBOX_DIR}")
        endif()
    endif()

    if (NOT DEFINED GGML_TSI_KERNEL_DIR)
        set(GGML_TSI_KERNEL_DIR ${CMAKE_SOURCE_DIR}/ggml-tsi-kernel/${GGML_TSAVORITE_TARGET})
    endif()

    # ---------------------------------------------------------------------------
    # TLIBS — libraries linked into the ggml-tsavorite backend
    #
    # FPGA:
    #   - Runtime + shim .so files come exclusively from RUNTIME_DIR/lib/
    #     (whether that is the SDK fpga/runtime or a local install-fpga override)
    #   - Kernel host objects come from the ggml-tsi-kernel build
    #   - MLIR compiler .so files (libFFMDeviceShim etc.) are NOT included here;
    #     they are not needed for the FPGA cross-compiled target
    #
    # POSIX:
    #   - Runtime + shim .so files come from RUNTIME_DIR/lib/
    #   - MLIR compiler .so files come from MLIR_COMPILER_DIR/lib/
    #   - Kernel host objects come from the ggml-tsi-kernel build
    # ---------------------------------------------------------------------------
    if (${GGML_TSAVORITE_TARGET} STREQUAL fpga)
        set(CMAKE_CROSSCOMPILING ON)
        set(ARCH_FLAGS -march=armv8-a)
        file(GLOB TLIBS
            "${RUNTIME_DIR}/lib/*.so"
            "${GGML_TSI_KERNEL_DIR}/host/*.o"
        )
        message(STATUS "Setting target as fpga")
    elseif (${GGML_TSAVORITE_TARGET} STREQUAL "posix")
        file(GLOB TLIBS
            "${RUNTIME_DIR}/lib/*.so"
            "${GGML_TSI_KERNEL_DIR}/host/*.o"
            "${MLIR_COMPILER_DIR}/lib/libFFMDeviceShim.so"
            "${MLIR_COMPILER_DIR}/lib/libTsavRTPosixShimCAPI.so"
            "${MLIR_COMPILER_DIR}/lib/libtxe-ffm-c-native.so"
            "${MLIR_COMPILER_DIR}/lib/libtxe-ffm-cpp.so"
            "${MLIR_COMPILER_DIR}/lib/libtvu_fau_full.so"
        )
        message(STATUS "Setting target as posix for tsavorite")
    endif()

    set(GGML_TSAVORITE_TARGET "${GGML_TSAVORITE_TARGET}" CACHE STRING "Target for tsavorite")
    set(TSAVORITE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/ggml/src/ggml-tsavorite/include)

    # ---------------------------------------------------------------------------
    # Include directories
    #
    # RUNTIME_DIR/include  — runtime + shim headers (tsi-rt/shim/, tsi-rt/host/,
    #                         tsi-rt/queues/, tsi-rt/memory/, tsi-rt/utils/, etc.)
    #                         This is the single authoritative source for all
    #                         tsi-rt headers. When RUNTIME_DIR points to a local
    #                         build, its shim headers supersede the SDK's — there
    #                         is no fallback to MLIR_COMPILER_DIR/include/tsi-rt/.
    #
    # MLIR_COMPILER_DIR/include — MLIR compiler headers (tsi-mlir/, CAPI/, txe/,
    #                              HAL/, runtime/CAPI/, device/, etc.)
    #                              Always sourced from the SDK.
    #
    # TOOLBOX_DIR/../../.. — toolbox headers (derived from SDK toolbox path)
    # ---------------------------------------------------------------------------
    include_directories($ENV{TSAVORITE_SYSROOT_INCLUDE_DIR})
    include_directories(${TSAVORITE_INCLUDE_DIR})
    include_directories(${RUNTIME_DIR}/include)
    include_directories(${RUNTIME_DIR}/include/tsi-rt/shim)
    include_directories(${MLIR_COMPILER_DIR}/include)
    include_directories(${TOOLBOX_DIR}/include)

    message(STATUS "tsavorite backend is enabled")
    message(STATUS "  MLIR_COMPILER_DIR : ${MLIR_COMPILER_DIR}")
    message(STATUS "  RUNTIME_DIR       : ${RUNTIME_DIR}")
    message(STATUS "  TOOLBOX_DIR       : ${TOOLBOX_DIR}")
    message(STATUS "  GGML_TSI_KERNEL_DIR: ${GGML_TSI_KERNEL_DIR}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    #set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

# Code coverage flags
if (ENABLE_COVERAGE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    message(STATUS "Code coverage (gcov) is enabled")
endif()

# Add path to modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LLAMA_STANDALONE ON)

    include(git-vars)

    # configure project version
    # TODO
else()
    set(LLAMA_STANDALONE OFF)
endif()

option(LLAMA_USE_SYSTEM_GGML "Use system libggml" OFF)

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(LLAMA_WASM_SINGLE_FILE "llama: embed WASM inside the generated llama.js" ON)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

option(BUILD_SHARED_LIBS "build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
    add_compile_options("$<$<COMPILE_LANGUAGE:C>:/bigobj>")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/bigobj>")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(LLAMA_TOOLS_INSTALL_DEFAULT OFF)
else()
    set(LLAMA_TOOLS_INSTALL_DEFAULT ${LLAMA_STANDALONE})
endif()

#
# option list
#

# debug
option(LLAMA_ALL_WARNINGS           "llama: enable all compiler warnings"                   ON)
option(LLAMA_ALL_WARNINGS_3RD_PARTY "llama: enable all compiler warnings in 3rd party libs" OFF)

# build
option(LLAMA_FATAL_WARNINGS "llama: enable -Werror flag" OFF)

# sanitizers
option(LLAMA_SANITIZE_THREAD    "llama: enable thread sanitizer"    OFF)
option(LLAMA_SANITIZE_ADDRESS   "llama: enable address sanitizer"   OFF)
option(LLAMA_SANITIZE_UNDEFINED "llama: enable undefined sanitizer" OFF)

# utils
option(LLAMA_BUILD_COMMON "llama: build common utils library" ${LLAMA_STANDALONE})

# extra artifacts
option(LLAMA_BUILD_TESTS    "llama: build tests"          ${LLAMA_STANDALONE})
option(LLAMA_BUILD_TOOLS    "llama: build tools"          ${LLAMA_STANDALONE})
option(LLAMA_BUILD_EXAMPLES "llama: build examples"       ${LLAMA_STANDALONE})
option(LLAMA_BUILD_SERVER   "llama: build server example" ${LLAMA_STANDALONE})
option(LLAMA_TOOLS_INSTALL  "llama: install tools"        ${LLAMA_TOOLS_INSTALL_DEFAULT})

# 3rd party libs
option(LLAMA_CURL       "llama: use libcurl to download model from an URL" ON)
option(LLAMA_OPENSSL    "llama: use openssl to support HTTPS" OFF)
option(LLAMA_LLGUIDANCE "llama-common: include LLGuidance library for structured output in common utils" OFF)

if (GGML_TSAVORITE)
    if (${GGML_TSAVORITE_TARGET} STREQUAL fpga)
        option(LLAMA_CURL       "llama: use libcurl to download model from an URL" OFF)
    else()
        option(LLAMA_CURL       "llama: use libcurl to download model from an URL" ON)
    endif()
else()
    option(LLAMA_CURL       "llama: use libcurl to download model from an URL" ON)
endif()

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/common.cmake)

if (NOT DEFINED LLAMA_BUILD_NUMBER)
    set(LLAMA_BUILD_NUMBER        ${BUILD_NUMBER})
endif()
if (NOT DEFINED LLAMA_BUILD_COMMIT)
    set(LLAMA_BUILD_COMMIT        ${BUILD_COMMIT})
endif()
set(LLAMA_INSTALL_VERSION 0.0.${LLAMA_BUILD_NUMBER})

# override ggml options
set(GGML_ALL_WARNINGS   ${LLAMA_ALL_WARNINGS})
set(GGML_FATAL_WARNINGS ${LLAMA_FATAL_WARNINGS})

# change the default for these ggml options
if (NOT DEFINED GGML_LLAMAFILE)
    set(GGML_LLAMAFILE_DEFAULT ON)
endif()

if (NOT DEFINED GGML_CUDA_GRAPHS)
    set(GGML_CUDA_GRAPHS_DEFAULT ON)
endif()

# transition helpers
function (llama_option_depr TYPE OLD NEW)
    if (${OLD})
        message(${TYPE} "${OLD} is deprecated and will be removed in the future.\nUse ${NEW} instead\n")
        set(${NEW} ON PARENT_SCOPE)
    endif()
endfunction()

llama_option_depr(FATAL_ERROR LLAMA_CUBLAS              GGML_CUDA)
llama_option_depr(WARNING     LLAMA_CUDA                GGML_CUDA)
llama_option_depr(WARNING     LLAMA_METAL               GGML_METAL)
llama_option_depr(WARNING     LLAMA_METAL_EMBED_LIBRARY GGML_METAL_EMBED_LIBRARY)
llama_option_depr(WARNING     LLAMA_NATIVE              GGML_NATIVE)
llama_option_depr(WARNING     LLAMA_RPC                 GGML_RPC)
llama_option_depr(WARNING     LLAMA_SYCL                GGML_SYCL)
llama_option_depr(WARNING     LLAMA_SYCL_F16            GGML_SYCL_F16)
llama_option_depr(WARNING     LLAMA_CANN                GGML_CANN)

if (NOT MSVC)
    if (LLAMA_SANITIZE_THREAD)
        message(STATUS "Using -fsanitize=thread")

        add_compile_options(-fsanitize=thread)
        link_libraries     (-fsanitize=thread)
    endif()

    if (LLAMA_SANITIZE_ADDRESS)
        message(STATUS "Using -fsanitize=address")

        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries     (-fsanitize=address)
    endif()

    if (LLAMA_SANITIZE_UNDEFINED)
        message(STATUS "Using -fsanitize=undefined")

        add_compile_options(-fsanitize=undefined)
        link_libraries     (-fsanitize=undefined)
    endif()
endif()

#
# 3rd-party
#

if (LLAMA_USE_SYSTEM_GGML)
    message(STATUS "Using system-provided libggml, skipping ggml build")
    find_package(ggml REQUIRED)
    add_library(ggml ALIAS ggml::ggml)
endif()

if (NOT TARGET ggml AND NOT LLAMA_USE_SYSTEM_GGML)
    set(GGML_BUILD_NUMBER ${LLAMA_BUILD_NUMBER})
    set(GGML_BUILD_COMMIT ${LLAMA_BUILD_COMMIT})
    add_subdirectory(ggml)
    # ... otherwise assume ggml is added by a parent CMakeLists.txt
endif()

if (MINGW)
    # Target Windows 8 for PrefetchVirtualMemory
    add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})
endif()

#
# build the library
#

add_subdirectory(src)

#
# utils, programs, examples and tests
#

if (NOT LLAMA_BUILD_COMMON)
    message(STATUS "LLAMA_BUILD_COMMON is OFF, disabling LLAMA_CURL")
    set(LLAMA_CURL OFF)
endif()

if (LLAMA_BUILD_COMMON)
    add_subdirectory(common)
endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    include(CTest)
    add_subdirectory(tests)
endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_EXAMPLES)
    add_subdirectory(examples)
    add_subdirectory(pocs)
endif()

if (LLAMA_BUILD_COMMON AND LLAMA_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(LLAMA_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(LLAMA_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(LLAMA_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

set(LLAMA_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llama.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/llama-cpp.h)

set_target_properties(llama
    PROPERTIES
        PUBLIC_HEADER "${LLAMA_PUBLIC_HEADERS}")

install(TARGETS llama LIBRARY PUBLIC_HEADER)

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/llama-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llama
    PATH_VARS LLAMA_INCLUDE_INSTALL_DIR
              LLAMA_LIB_INSTALL_DIR
              LLAMA_BIN_INSTALL_DIR )

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/llama-version.cmake
    VERSION ${LLAMA_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/llama-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/llama-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/llama)

install(
    FILES convert_hf_to_gguf.py
    PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
    DESTINATION ${CMAKE_INSTALL_BINDIR})

configure_file(cmake/llama.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/llama.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/llama.pc"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)