cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("ggml" C CXX ASM)

### GGML Version
set(GGML_VERSION_MAJOR 0)
set(GGML_VERSION_MINOR 9)
set(GGML_VERSION_PATCH 4)
set(GGML_VERSION_BASE "${GGML_VERSION_MAJOR}.${GGML_VERSION_MINOR}.${GGML_VERSION_PATCH}")

find_program(GIT_EXE NAMES git git.exe NO_CMAKE_FIND_ROOT_PATH)
if(GIT_EXE)
    # Get current git commit hash
    execute_process(COMMAND ${GIT_EXE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GGML_BUILD_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    # Check if the working directory is dirty (i.e., has uncommitted changes)
    execute_process(COMMAND ${GIT_EXE} diff-index --quiet HEAD -- .
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GGML_GIT_DIRTY
        ERROR_QUIET
    )
endif()

set(GGML_VERSION "${GGML_VERSION_BASE}")

if(NOT GGML_BUILD_COMMIT)
    set(GGML_BUILD_COMMIT "unknown")
endif()

# Build the commit string with optional dirty flag
if(DEFINED GGML_GIT_DIRTY AND GGML_GIT_DIRTY EQUAL 1)
    set(GGML_BUILD_COMMIT "${GGML_BUILD_COMMIT}-dirty")
endif()

include(CheckIncludeFileCXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(GGML_STANDALONE ON)

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    # configure project version
    # TODO
else()
    set(GGML_STANDALONE OFF)

    if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    endif()
endif()

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(GGML_WASM_SINGLE_FILE "ggml: embed WASM inside the generated ggml.js" ON)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()

# remove the lib prefix on win32 mingw
if (WIN32)
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
    set(CMAKE_SHARED_MODULE_PREFIX  "")
endif()

option(BUILD_SHARED_LIBS           "ggml: build shared libraries" ${BUILD_SHARED_LIBS_DEFAULT})
option(GGML_BACKEND_DL             "ggml: build backends as dynamic libraries (requires BUILD_SHARED_LIBS)" OFF)
set(GGML_BACKEND_DIR "" CACHE PATH "ggml: directory to load dynamic backends from (requires GGML_BACKEND_DL")

#
# option list
#

# TODO: mark all options as advanced when not GGML_STANDALONE

if (APPLE)
    set(GGML_METAL_DEFAULT ON)
    set(GGML_BLAS_DEFAULT ON)
    set(GGML_BLAS_VENDOR_DEFAULT "Apple")
else()
    set(GGML_METAL_DEFAULT OFF)
    set(GGML_BLAS_DEFAULT OFF)
    set(GGML_BLAS_VENDOR_DEFAULT "Generic")
endif()

if (CMAKE_CROSSCOMPILING OR DEFINED ENV{SOURCE_DATE_EPOCH})
    message(STATUS "Setting GGML_NATIVE_DEFAULT to OFF")
    set(GGML_NATIVE_DEFAULT OFF)
else()
    set(GGML_NATIVE_DEFAULT ON)
endif()

# defaults
if (NOT GGML_LLAMAFILE_DEFAULT)
    set(GGML_LLAMAFILE_DEFAULT OFF)
endif()

if (NOT GGML_CUDA_GRAPHS_DEFAULT)
    set(GGML_CUDA_GRAPHS_DEFAULT OFF)
endif()

# general
option(GGML_STATIC "ggml: static link libraries"                     OFF)
option(GGML_NATIVE "ggml: optimize the build for the current system" ${GGML_NATIVE_DEFAULT})
option(GGML_LTO    "ggml: enable link time optimization"             OFF)
option(GGML_CCACHE "ggml: use ccache if available"                   ON)

# debug
option(GGML_ALL_WARNINGS           "ggml: enable all compiler warnings"                   ON)
option(GGML_ALL_WARNINGS_3RD_PARTY "ggml: enable all compiler warnings in 3rd party libs" OFF)
option(GGML_GPROF                  "ggml: enable gprof"                                   OFF)

# build
option(GGML_FATAL_WARNINGS    "ggml: enable -Werror flag"    OFF)

# sanitizers
option(GGML_SANITIZE_THREAD    "ggml: enable thread sanitizer"    OFF)
option(GGML_SANITIZE_ADDRESS   "ggml: enable address sanitizer"   OFF)
option(GGML_SANITIZE_UNDEFINED "ggml: enable undefined sanitizer" OFF)

# instruction set specific
if (GGML_NATIVE OR NOT GGML_NATIVE_DEFAULT)
    set(INS_ENB OFF)
else()
    set(INS_ENB ON)
endif()

message(DEBUG "GGML_NATIVE         : ${GGML_NATIVE}")
message(DEBUG "GGML_NATIVE_DEFAULT : ${GGML_NATIVE_DEFAULT}")
message(DEBUG "INS_ENB             : ${INS_ENB}")

option(GGML_CPU_HBM          "ggml: use memkind for CPU HBM" OFF)
option(GGML_CPU_REPACK       "ggml: use runtime weight conversion of Q4_0 to Q4_X_X" ON)
option(GGML_CPU_KLEIDIAI     "ggml: use KleidiAI optimized kernels if applicable" OFF)
option(GGML_SSE42            "ggml: enable SSE 4.2"          ${INS_ENB})
option(GGML_AVX              "ggml: enable AVX"              ${INS_ENB})
option(GGML_AVX_VNNI         "ggml: enable AVX-VNNI"         OFF)
option(GGML_AVX2             "ggml: enable AVX2"             ${INS_ENB})
option(GGML_BMI2             "ggml: enable BMI2"             ${INS_ENB})
option(GGML_AVX512           "ggml: enable AVX512F"          OFF)
option(GGML_AVX512_VBMI      "ggml: enable AVX512-VBMI"      OFF)
option(GGML_AVX512_VNNI      "ggml: enable AVX512-VNNI"      OFF)
option(GGML_AVX512_BF16      "ggml: enable AVX512-BF16"      OFF)
if (NOT MSVC)
    # in MSVC F16C and FMA is implied with AVX2/AVX512
    option(GGML_FMA          "ggml: enable FMA"              ${INS_ENB})
    option(GGML_F16C         "ggml: enable F16C"             ${INS_ENB})
    # MSVC does not seem to support AMX
    option(GGML_AMX_TILE     "ggml: enable AMX-TILE"         OFF)
    option(GGML_AMX_INT8     "ggml: enable AMX-INT8"         OFF)
    option(GGML_AMX_BF16     "ggml: enable AMX-BF16"         OFF)
endif()
option(GGML_LASX             "ggml: enable lasx"             ON)
option(GGML_LSX              "ggml: enable lsx"              ON)
option(GGML_RVV              "ggml: enable rvv"              ON)
option(GGML_RV_ZFH           "ggml: enable riscv zfh"        ON)
option(GGML_RV_ZVFH          "ggml: enable riscv zvfh"       ON)
option(GGML_RV_ZICBOP        "ggml: enable riscv zicbop"     ON)
option(GGML_RV_ZIHINTPAUSE   "ggml: enable riscv zihintpause "  ON)
option(GGML_XTHEADVECTOR     "ggml: enable xtheadvector"     OFF)
option(GGML_VXE              "ggml: enable vxe"              ${GGML_NATIVE})

option(GGML_CPU_ALL_VARIANTS "ggml: build all variants of the CPU backend (requires GGML_BACKEND_DL)" OFF)
set(GGML_CPU_ARM_ARCH        "" CACHE STRING "ggml: CPU architecture for ARM")
set(GGML_CPU_POWERPC_CPUTYPE "" CACHE STRING "ggml: CPU type for PowerPC")

# ggml core
set(GGML_SCHED_MAX_COPIES  "4" CACHE STRING "ggml: max input copies for pipeline parallelism")
option(GGML_CPU                             "ggml: enable CPU backend"                        ON)
option(GGML_SCHED_NO_REALLOC                "ggml: disallow reallocations in ggml-alloc (for debugging)" OFF)

# 3rd party libs / backends
option(GGML_ACCELERATE                      "ggml: enable Accelerate framework"               ON)
option(GGML_BLAS                            "ggml: use BLAS"                                  ${GGML_BLAS_DEFAULT})
set(GGML_BLAS_VENDOR ${GGML_BLAS_VENDOR_DEFAULT} CACHE STRING
                                            "ggml: BLAS library vendor")
option(GGML_LLAMAFILE                       "ggml: use LLAMAFILE"                             ${GGML_LLAMAFILE_DEFAULT})

option(GGML_CUDA                            "ggml: use CUDA"                                  OFF)
option(GGML_MUSA                            "ggml: use MUSA"                                  OFF)
option(GGML_CUDA_FORCE_MMQ                  "ggml: use mmq kernels instead of cuBLAS"         OFF)
option(GGML_CUDA_FORCE_CUBLAS               "ggml: always use cuBLAS instead of mmq kernels"  OFF)
set   (GGML_CUDA_PEER_MAX_BATCH_SIZE "128" CACHE STRING
                                            "ggml: max. batch size for using peer access")
option(GGML_CUDA_NO_PEER_COPY               "ggml: do not use peer to peer copies"            OFF)
option(GGML_CUDA_NO_VMM                     "ggml: do not try to use CUDA VMM"                OFF)
option(GGML_CUDA_FA                         "ggml: compile ggml FlashAttention CUDA kernels"  ON)
option(GGML_CUDA_FA_ALL_QUANTS              "ggml: compile all quants for FlashAttention"     OFF)
option(GGML_CUDA_GRAPHS                     "ggml: use CUDA graphs (llama.cpp only)"          ${GGML_CUDA_GRAPHS_DEFAULT})
set   (GGML_CUDA_COMPRESSION_MODE "size" CACHE STRING
                                            "ggml: cuda link binary compression mode; requires cuda 12.8+")
set_property(CACHE GGML_CUDA_COMPRESSION_MODE PROPERTY STRINGS "none;speed;balance;size")

option(GGML_HIP                             "ggml: use HIP"                                   OFF)
option(GGML_HIP_GRAPHS                      "ggml: use HIP graph, experimental, slow"         OFF)
option(GGML_HIP_NO_VMM                      "ggml: do not try to use HIP VMM"                 ON)
option(GGML_HIP_ROCWMMA_FATTN               "ggml: enable rocWMMA for FlashAttention"         OFF)
option(GGML_HIP_MMQ_MFMA                    "ggml: enable MFMA MMA for CDNA in MMQ"           ON)
option(GGML_HIP_EXPORT_METRICS              "ggml: enable kernel perf metrics output"         OFF)
option(GGML_MUSA_GRAPHS                     "ggml: use MUSA graph, experimental, unstable"    OFF)
option(GGML_MUSA_MUDNN_COPY                 "ggml: enable muDNN for accelerated copy"         OFF)
option(GGML_VULKAN                          "ggml: use Vulkan"                                OFF)
option(GGML_VULKAN_CHECK_RESULTS            "ggml: run Vulkan op checks"                      OFF)
option(GGML_VULKAN_DEBUG                    "ggml: enable Vulkan debug output"                OFF)
option(GGML_VULKAN_MEMORY_DEBUG             "ggml: enable Vulkan memory debug output"         OFF)
option(GGML_VULKAN_SHADER_DEBUG_INFO        "ggml: enable Vulkan shader debug info"           OFF)
option(GGML_VULKAN_VALIDATE                 "ggml: enable Vulkan validation"                  OFF)
option(GGML_VULKAN_RUN_TESTS                "ggml: run Vulkan tests"                          OFF)
option(GGML_WEBGPU                          "ggml: use WebGPU"                                OFF)
option(GGML_WEBGPU_DEBUG                    "ggml: enable WebGPU debug output"                OFF)
option(GGML_WEBGPU_CPU_PROFILE              "ggml: enable WebGPU profiling (CPU)"             OFF)
option(GGML_WEBGPU_GPU_PROFILE              "ggml: enable WebGPU profiling (GPU)"             OFF)
option(GGML_WEBGPU_JSPI                     "ggml: use JSPI for WebGPU"                       ON)
option(GGML_ZDNN                            "ggml: use zDNN"                                  OFF)
option(GGML_METAL                           "ggml: use Metal"                                 ${GGML_METAL_DEFAULT})
option(GGML_METAL_NDEBUG                    "ggml: disable Metal debugging"                   OFF)
option(GGML_METAL_SHADER_DEBUG              "ggml: compile Metal with -fno-fast-math"         OFF)
option(GGML_METAL_EMBED_LIBRARY             "ggml: embed Metal library"                       ${GGML_METAL})
set   (GGML_METAL_MACOSX_VERSION_MIN "" CACHE STRING
                                            "ggml: metal minimum macOS version")
set   (GGML_METAL_STD "" CACHE STRING       "ggml: metal standard version (-std flag)")
option(GGML_OPENMP                          "ggml: use OpenMP"                                ON)
option(GGML_RPC                             "ggml: use RPC"                                   OFF)
option(GGML_SYCL                            "ggml: use SYCL"                                  OFF)
option(GGML_SYCL_F16                        "ggml: use 16 bit floats for sycl calculations"   OFF)
option(GGML_SYCL_GRAPH                      "ggml: enable graphs in the SYCL backend"         ON)
option(GGML_SYCL_DNN                        "ggml: enable oneDNN in the SYCL backend"         ON)
set   (GGML_SYCL_TARGET "INTEL" CACHE STRING
                                            "ggml: sycl target device")
set   (GGML_SYCL_DEVICE_ARCH "" CACHE STRING
                                            "ggml: sycl device architecture")

option(GGML_OPENCL                          "ggml: use OpenCL"                                OFF)
option(GGML_OPENCL_PROFILING                "ggml: use OpenCL profiling (increases overhead)" OFF)
option(GGML_OPENCL_EMBED_KERNELS            "ggml: embed kernels"                             ON)
option(GGML_OPENCL_USE_ADRENO_KERNELS       "ggml: use optimized kernels for Adreno"          ON)
set   (GGML_OPENCL_TARGET_VERSION "300" CACHE STRING
                                            "gmml: OpenCL API version to target")

option(GGML_HEXAGON                         "ggml: enable Hexagon backend"                    OFF)
set(GGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE 128 CACHE STRING "ggml: quantize group size (32, 64, or 128)")

# toolchain for vulkan-shaders-gen
set   (GGML_VULKAN_SHADERS_GEN_TOOLCHAIN "" CACHE FILEPATH "ggml: toolchain file for vulkan-shaders-gen")

option(GGML_ZENDNN                          "ggml: use ZenDNN"                                OFF)
option(ZENDNN_ROOT                          "ggml: path to ZenDNN installation"               "")

# extra artifacts
option(GGML_BUILD_TESTS    "ggml: build tests"    ${GGML_STANDALONE})
option(GGML_BUILD_EXAMPLES "ggml: build examples" ${GGML_STANDALONE})

#
# dependencies
#

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)

include(GNUInstallDirs)

#
# build the library
#

add_subdirectory(src)

#
# tests and examples
#

if (GGML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

if (GGML_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

#
# install
#

include(CMakePackageConfigHelpers)

# all public headers
set(GGML_PUBLIC_HEADERS
    include/ggml.h
    include/ggml-cpu.h
    include/ggml-alloc.h
    include/ggml-backend.h
    include/ggml-blas.h
    include/ggml-cann.h
    include/ggml-cpp.h
    include/ggml-cuda.h
    include/ggml-opt.h
    include/ggml-metal.h
    include/ggml-rpc.h
    include/ggml-sycl.h
    include/ggml-vulkan.h
    include/ggml-webgpu.h
    include/ggml-zendnn.h
    include/gguf.h)

set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${GGML_PUBLIC_HEADERS}")
#if (GGML_METAL)
#    set_target_properties(ggml PROPERTIES RESOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/ggml-metal.metal")
#endif()
install(TARGETS ggml LIBRARY PUBLIC_HEADER)
install(TARGETS ggml-base LIBRARY)

if (GGML_STANDALONE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/ggml.pc.in
        ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
        @ONLY)

    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ggml.pc
        DESTINATION share/pkgconfig)
endif()

#
# Create CMake package
#



# Capture variables prefixed with GGML_.

set(variable_set_statements
"
####### Expanded from @GGML_VARIABLES_EXPANED@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run        #######

")

set(GGML_SHARED_LIB ${BUILD_SHARED_LIBS})

get_cmake_property(all_variables VARIABLES)
foreach(variable_name IN LISTS all_variables)
    if(variable_name MATCHES "^GGML_")
        string(REPLACE ";" "\\;"
               variable_value "${${variable_name}}")

        set(variable_set_statements
            "${variable_set_statements}set(${variable_name} \"${variable_value}\")\n")
    endif()
endforeach()

set(GGML_VARIABLES_EXPANDED ${variable_set_statements})

# Create the CMake package and set install location.

set(GGML_INSTALL_VERSION ${GGML_VERSION})
set(GGML_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Location of header  files")
set(GGML_LIB_INSTALL_DIR     ${CMAKE_INSTALL_LIBDIR}     CACHE PATH "Location of library files")
set(GGML_BIN_INSTALL_DIR     ${CMAKE_INSTALL_BINDIR}     CACHE PATH "Location of binary  files")

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ggml-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/ggml-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ggml
    PATH_VARS GGML_INCLUDE_INSTALL_DIR
              GGML_LIB_INSTALL_DIR
              GGML_BIN_INSTALL_DIR)

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/ggml-version.cmake
    VERSION ${GGML_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

target_compile_definitions(ggml-base PRIVATE
    GGML_VERSION="${GGML_INSTALL_VERSION}"
    GGML_COMMIT="${GGML_BUILD_COMMIT}"
)
message(STATUS "ggml version: ${GGML_INSTALL_VERSION}")
message(STATUS "ggml commit:  ${GGML_BUILD_COMMIT}")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ggml-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/ggml-version.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ggml)

if (MSVC)
    set(MSVC_WARNING_FLAGS
        /wd4005  # Macro redefinition
        /wd4244  # Conversion from one type to another type, possible loss of data
        /wd4267  # Conversion from 'size_t' to a smaller type, possible loss of data
        /wd4305  # Conversion from 'type1' to 'type2', possible loss of data
        /wd4566  # Conversion from 'char' to 'wchar_t', possible loss of data
        /wd4996  # Disable POSIX deprecation warnings
        /wd4702  # Unreachable code warnings
    )
    set(MSVC_COMPILE_OPTIONS
        "$<$<COMPILE_LANGUAGE:C>:/utf-8>"
        "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
    )
    function(configure_msvc_target target_name)
        if(TARGET ${target_name})
            target_compile_options(${target_name} PRIVATE ${MSVC_WARNING_FLAGS})
            target_compile_options(${target_name} PRIVATE ${MSVC_COMPILE_OPTIONS})
        endif()
    endfunction()

    configure_msvc_target(ggml-base)
    configure_msvc_target(ggml)
    configure_msvc_target(ggml-cpu)
    configure_msvc_target(ggml-cpu-x64)
    configure_msvc_target(ggml-cpu-sse42)
    configure_msvc_target(ggml-cpu-sandybridge)
    configure_msvc_target(ggml-cpu-haswell)
    configure_msvc_target(ggml-cpu-skylakex)
    configure_msvc_target(ggml-cpu-icelake)
    configure_msvc_target(ggml-cpu-alderlake)

    if (GGML_BUILD_EXAMPLES)
        configure_msvc_target(common-ggml)
        configure_msvc_target(common)

        configure_msvc_target(mnist-common)
        configure_msvc_target(mnist-eval)
        configure_msvc_target(mnist-train)

        configure_msvc_target(gpt-2-ctx)
        configure_msvc_target(gpt-2-alloc)
        configure_msvc_target(gpt-2-backend)
        configure_msvc_target(gpt-2-sched)
        configure_msvc_target(gpt-2-quantize)
        configure_msvc_target(gpt-2-batched)

        configure_msvc_target(gpt-j)
        configure_msvc_target(gpt-j-quantize)

        configure_msvc_target(magika)
        configure_msvc_target(yolov3-tiny)
        configure_msvc_target(sam)

        configure_msvc_target(simple-ctx)
        configure_msvc_target(simple-backend)
    endif()

    if (GGML_BUILD_TESTS)
        configure_msvc_target(test-mul-mat)
        configure_msvc_target(test-arange)
        configure_msvc_target(test-backend-ops)
        configure_msvc_target(test-cont)
        configure_msvc_target(test-conv-transpose)
        configure_msvc_target(test-conv-transpose-1d)
        configure_msvc_target(test-conv1d)
        configure_msvc_target(test-conv2d)
        configure_msvc_target(test-conv2d-dw)
        configure_msvc_target(test-customop)
        configure_msvc_target(test-dup)
        configure_msvc_target(test-opt)
        configure_msvc_target(test-pool)
    endif ()
endif()
