cmake_minimum_required(VERSION 3.14) # for add_link_options and implicit target directories.
project("llama.cpp" C CXX)
include(CheckIncludeFileCXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LLAMA_STANDALONE ON)

    # configure project version
    # TODO
else()
    set(LLAMA_STANDALONE OFF)
endif()

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


#
# Option list
#

if (APPLE)
    set(LLAMA_METAL_DEFAULT ON)
    set(LLAMA_BLAS_DEFAULT ON)
    set(LLAMA_BLAS_VENDOR_DEFAULT "Apple")
else()
    set(LLAMA_METAL_DEFAULT OFF)
    set(LLAMA_BLAS_DEFAULT OFF)
    set(LLAMA_BLAS_VENDOR_DEFAULT "Generic")
endif()

set(LLAMA_LLAMAFILE_DEFAULT ON)

# general
option(BUILD_SHARED_LIBS                "build shared libraries"                                OFF)
option(LLAMA_STATIC                     "llama: static link libraries"                          OFF)
option(LLAMA_NATIVE                     "llama: enable -march=native flag"                      ON)
option(LLAMA_LTO                        "llama: enable link time optimization"                  OFF)
option(LLAMA_CCACHE                     "llama: use ccache if available"                        ON)

# debug
option(LLAMA_ALL_WARNINGS               "llama: enable all compiler warnings"                   ON)
option(LLAMA_ALL_WARNINGS_3RD_PARTY     "llama: enable all compiler warnings in 3rd party libs" OFF)
option(LLAMA_GPROF                      "llama: enable gprof"                                   OFF)

# build
option(LLAMA_FATAL_WARNINGS             "llama: enable -Werror flag"                            OFF)

# sanitizers
option(LLAMA_SANITIZE_THREAD            "llama: enable thread sanitizer"                        OFF)
option(LLAMA_SANITIZE_ADDRESS           "llama: enable address sanitizer"                       OFF)
option(LLAMA_SANITIZE_UNDEFINED         "llama: enable undefined sanitizer"                     OFF)

# instruction set specific
if (LLAMA_NATIVE)
    set(INS_ENB OFF)
else()
    set(INS_ENB ON)
endif()

option(LLAMA_SVE                             "llama: enable SVE"                                OFF)
option(LLAMA_AVX                             "llama: enable AVX"                                ${INS_ENB})
option(LLAMA_AVX2                            "llama: enable AVX2"                               ${INS_ENB})
option(LLAMA_AVX512                          "llama: enable AVX512"                             OFF)
option(LLAMA_AVX512_VBMI                     "llama: enable AVX512-VBMI"                        OFF)
option(LLAMA_AVX512_VNNI                     "llama: enable AVX512-VNNI"                        OFF)
option(LLAMA_AVX512_BF16                     "llama: enable AVX512-BF16"                        OFF)
option(LLAMA_FMA                             "llama: enable FMA"                                ${INS_ENB})
# in MSVC F16C is implied with AVX2/AVX512
if (NOT MSVC)
    option(LLAMA_F16C                        "llama: enable F16C"                               ${INS_ENB})
endif()

if (WIN32)
    set(LLAMA_WIN_VER "0x602" CACHE STRING "llama: Windows Version")
endif()

# 3rd party libs
option(LLAMA_ACCELERATE                      "llama: enable Accelerate framework"               ON)
option(LLAMA_BLAS                            "llama: use BLAS"                                  ${LLAMA_BLAS_DEFAULT})
set(LLAMA_BLAS_VENDOR ${LLAMA_BLAS_VENDOR_DEFAULT} CACHE STRING
                                             "llama: BLAS library vendor")
option(LLAMA_LLAMAFILE                       "llama: use llamafile SGEMM"                       ${LLAMA_LLAMAFILE_DEFAULT})
option(LLAMA_CUDA                            "llama: use CUDA"                                  OFF)
option(LLAMA_CUBLAS                          "llama: use CUDA (deprecated, use LLAMA_CUDA)"     OFF)
option(LLAMA_CUDA_FORCE_DMMV                 "llama: use dmmv instead of mmvq CUDA kernels"     OFF)
option(LLAMA_CUDA_FORCE_MMQ                  "llama: use mmq kernels instead of cuBLAS"         OFF)
set(LLAMA_CUDA_DMMV_X      "32" CACHE STRING "llama: x stride for dmmv CUDA kernels")
set(LLAMA_CUDA_MMV_Y        "1" CACHE STRING "llama: y block size for mmv CUDA kernels")
option(LLAMA_CUDA_F16                        "llama: use 16 bit floats for some calculations"   OFF)
set(LLAMA_CUDA_KQUANTS_ITER "2" CACHE STRING "llama: iters./thread per block for Q2_K/Q6_K")
set(LLAMA_CUDA_PEER_MAX_BATCH_SIZE "128" CACHE STRING
                                             "llama: max. batch size for using peer access")
option(LLAMA_CUDA_NO_PEER_COPY               "llama: do not use peer to peer copies"            OFF)
option(LLAMA_CUDA_NO_VMM                     "llama: do not try to use CUDA VMM"                OFF)
option(LLAMA_CUDA_FA_ALL_QUANTS              "llama: compile all quants for FlashAttention"     OFF)

option(LLAMA_CURL                            "llama: use libcurl to download model from an URL" OFF)
option(LLAMA_HIPBLAS                         "llama: use hipBLAS"                               OFF)
option(LLAMA_HIP_UMA                         "llama: use HIP unified memory architecture"       OFF)
option(LLAMA_VULKAN                          "llama: use Vulkan"                                OFF)
option(LLAMA_VULKAN_CHECK_RESULTS            "llama: run Vulkan op checks"                      OFF)
option(LLAMA_VULKAN_DEBUG                    "llama: enable Vulkan debug output"                OFF)
option(LLAMA_VULKAN_MEMORY_DEBUG             "llama: enable Vulkan memory debug output"         OFF)
option(LLAMA_VULKAN_VALIDATE                 "llama: enable Vulkan validation"                  OFF)
option(LLAMA_VULKAN_RUN_TESTS                "llama: run Vulkan tests"                          OFF)
option(LLAMA_METAL                           "llama: use Metal"                                 ${LLAMA_METAL_DEFAULT})
option(LLAMA_METAL_NDEBUG                    "llama: disable Metal debugging"                   OFF)
option(LLAMA_METAL_SHADER_DEBUG              "llama: compile Metal with -fno-fast-math"         OFF)
option(LLAMA_METAL_EMBED_LIBRARY             "llama: embed Metal library"                       OFF)
set(LLAMA_METAL_MACOSX_VERSION_MIN "" CACHE STRING
                                             "llama: metal minimum macOS version")
set(LLAMA_METAL_STD "" CACHE STRING          "llama: metal standard version (-std flag)")
option(LLAMA_KOMPUTE                         "llama: use Kompute"                               OFF)
option(LLAMA_RPC                             "llama: use RPC"                                   OFF)
option(LLAMA_OPENMP                          "llama: use OpenMP"                                ON)
option(LLAMA_SYCL                            "llama: use SYCL"                                  OFF)
option(LLAMA_SYCL_F16                        "llama: use 16 bit floats for sycl calculations"   OFF)
set(LLAMA_SYCL_TARGET   "INTEL" CACHE STRING "llama: sycl target device")
option(LLAMA_CPU_HBM                         "llama: use memkind for CPU HBM"                   OFF)
set(LLAMA_SCHED_MAX_COPIES  "4" CACHE STRING "llama: max input copies for pipeline parallelism")

option(LLAMA_BUILD_TESTS                     "llama: build tests"    ${LLAMA_STANDALONE})
option(LLAMA_BUILD_EXAMPLES                  "llama: build examples" ${LLAMA_STANDALONE})
option(LLAMA_BUILD_SERVER                    "llama: build server example"                      ON)
option(LLAMA_LASX                            "llama: enable lasx"                               ON)
option(LLAMA_LSX                             "llama: enable lsx"                                ON)

# Required for relocatable CMake package
include(${CMAKE_CURRENT_SOURCE_DIR}/scripts/build-info.cmake)

#
# Compile flags
#

if (LLAMA_SYCL)
    set(CMAKE_CXX_STANDARD 17)
else()
    set(CMAKE_CXX_STANDARD 11)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)
include(CheckCXXCompilerFlag)

add_compile_definitions(GGML_SCHED_MAX_COPIES=${LLAMA_SCHED_MAX_COPIES})

# enable libstdc++ assertions for debug builds
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions($<$<CONFIG:Debug>:_GLIBCXX_ASSERTIONS>)
endif()

if (NOT MSVC)
    if (LLAMA_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
        link_libraries     (-fsanitize=thread)
    endif()

    if (LLAMA_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        link_libraries     (-fsanitize=address)
    endif()

    if (LLAMA_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
        link_libraries     (-fsanitize=undefined)
    endif()
endif()

if (APPLE AND LLAMA_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        add_compile_definitions(GGML_USE_ACCELERATE)
        add_compile_definitions(ACCELERATE_NEW_LAPACK)
        add_compile_definitions(ACCELERATE_LAPACK_ILP64)
        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

if (LLAMA_METAL)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(METAL_FRAMEWORK    Metal      REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit   REQUIRED)

    message(STATUS "Metal framework found")
    set(GGML_HEADERS_METAL ggml-metal.h)
    set(GGML_SOURCES_METAL ggml-metal.m)

    add_compile_definitions(GGML_USE_METAL)
    if (LLAMA_METAL_NDEBUG)
        add_compile_definitions(GGML_METAL_NDEBUG)
    endif()

    # copy ggml-common.h and ggml-metal.metal to bin directory
    configure_file(ggml-common.h    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-common.h    COPYONLY)
    configure_file(ggml-metal.metal ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal.metal COPYONLY)

    if (LLAMA_METAL_EMBED_LIBRARY)
        enable_language(ASM)
        add_compile_definitions(GGML_METAL_EMBED_LIBRARY)

        set(METALLIB_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/ggml-common.h")
        set(METALLIB_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ggml-metal.metal")

        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/autogenerated")

        # merge ggml-common.h and ggml-metal.metal into a single file
        set(METALLIB_EMBED_ASM    "${CMAKE_BINARY_DIR}/autogenerated/ggml-metal-embed.s")
        set(METALLIB_SOURCE_EMBED "${CMAKE_BINARY_DIR}/autogenerated/ggml-metal-embed.metal")

        add_custom_command(
            OUTPUT ${METALLIB_EMBED_ASM}
            COMMAND echo "Embedding Metal library"
            COMMAND sed -e '/\#include \"ggml-common.h\"/r ${METALLIB_COMMON}' -e '/\#include \"ggml-common.h\"/d' < ${METALLIB_SOURCE} > ${METALLIB_SOURCE_EMBED}
            COMMAND echo ".section __DATA,__ggml_metallib"          >  ${METALLIB_EMBED_ASM}
            COMMAND echo ".globl _ggml_metallib_start"              >> ${METALLIB_EMBED_ASM}
            COMMAND echo "_ggml_metallib_start:"                    >> ${METALLIB_EMBED_ASM}
            COMMAND echo ".incbin \\\"${METALLIB_SOURCE_EMBED}\\\"" >> ${METALLIB_EMBED_ASM}
            COMMAND echo ".globl _ggml_metallib_end"                >> ${METALLIB_EMBED_ASM}
            COMMAND echo "_ggml_metallib_end:"                      >> ${METALLIB_EMBED_ASM}
            DEPENDS ggml-metal.metal ggml-common.h
            COMMENT "Generate assembly for embedded Metal library"
        )

        set(GGML_SOURCES_METAL ${GGML_SOURCES_METAL} ${METALLIB_EMBED_ASM})
    else()
        if (LLAMA_METAL_SHADER_DEBUG)
            # custom command to do the following:
            #   xcrun -sdk macosx metal    -fno-fast-math -c ggml-metal.metal -o ggml-metal.air
            #   xcrun -sdk macosx metallib                   ggml-metal.air   -o default.metallib
            #
            # note: this is the only way I found to disable fast-math in Metal. it's ugly, but at least it works
            #       disabling fast math is needed in order to pass tests/test-backend-ops
            # note: adding -fno-inline fixes the tests when using MTL_SHADER_VALIDATION=1
            # note: unfortunately, we have to call it default.metallib instead of ggml.metallib
            #       ref: https://github.com/ggerganov/whisper.cpp/issues/1720
            set(XC_FLAGS -fno-fast-math -fno-inline -g)
        else()
            set(XC_FLAGS -O3)
        endif()

        # Append macOS metal versioning flags
        if (LLAMA_METAL_MACOSX_VERSION_MIN)
            message(STATUS "Adding -mmacosx-version-min=${LLAMA_METAL_MACOSX_VERSION_MIN} flag to metal compilation")
            list(APPEND XC_FLAGS -mmacosx-version-min=${LLAMA_METAL_MACOSX_VERSION_MIN})
        endif()
        if (LLAMA_METAL_STD)
            message(STATUS "Adding -std=${LLAMA_METAL_STD} flag to metal compilation")
            list(APPEND XC_FLAGS -std=${LLAMA_METAL_STD})
        endif()

        add_custom_command(
            OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
            COMMAND xcrun -sdk macosx metal    ${XC_FLAGS} -c ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal.metal -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal.air
            COMMAND xcrun -sdk macosx metallib                ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal.air   -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
            COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal.air
            COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-common.h
            COMMAND rm -f ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ggml-metal.metal
            DEPENDS ggml-metal.metal ggml-common.h
            COMMENT "Compiling Metal kernels"
            )

        add_custom_target(
            ggml-metal ALL
            DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
            )
    endif() # LLAMA_METAL_EMBED_LIBRARY

    set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS}
        ${FOUNDATION_LIBRARY}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        )
endif()

if (LLAMA_OPENMP)
    find_package(OpenMP)
    if (OpenMP_FOUND)
        message(STATUS "OpenMP found")
        add_compile_definitions(GGML_USE_OPENMP)
        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
    else()
        message(WARNING "OpenMP not found")
    endif()
endif()

if (LLAMA_BLAS)
    if (LLAMA_STATIC)
        set(BLA_STATIC ON)
    endif()
    #if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.22)
    #    set(BLA_SIZEOF_INTEGER 8)
    #endif()

    set(BLA_VENDOR ${LLAMA_BLAS_VENDOR})
    find_package(BLAS)

    if (BLAS_FOUND)
        message(STATUS "BLAS found, Libraries: ${BLAS_LIBRARIES}")

        if (("${BLAS_INCLUDE_DIRS}" STREQUAL "") AND NOT (${LLAMA_BLAS_VENDOR} MATCHES "Apple"))
            # BLAS_INCLUDE_DIRS is missing in FindBLAS.cmake.
            # see https://gitlab.kitware.com/cmake/cmake/-/issues/20268
            find_package(PkgConfig REQUIRED)
            if (${LLAMA_BLAS_VENDOR} MATCHES "Generic")
                pkg_check_modules(DepBLAS REQUIRED blas)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "OpenBLAS")
                # As of openblas v0.3.22, the 64-bit is named openblas64.pc
                pkg_check_modules(DepBLAS openblas64)
                if (NOT DepBLAS_FOUND)
                    pkg_check_modules(DepBLAS REQUIRED openblas)
                endif()
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "FLAME")
                pkg_check_modules(DepBLAS REQUIRED blis)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "ATLAS")
                pkg_check_modules(DepBLAS REQUIRED blas-atlas)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "FlexiBLAS")
                pkg_check_modules(DepBLAS REQUIRED flexiblas_api)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "Intel")
                # all Intel* libraries share the same include path
                pkg_check_modules(DepBLAS REQUIRED mkl-sdl)
            elseif (${LLAMA_BLAS_VENDOR} MATCHES "NVHPC")
                # this doesn't provide pkg-config
                # suggest to assign BLAS_INCLUDE_DIRS on your own
                if ("${NVHPC_VERSION}" STREQUAL "")
                    message(WARNING "Better to set NVHPC_VERSION")
                else()
                    set(DepBLAS_FOUND ON)
                    set(DepBLAS_INCLUDE_DIRS "/opt/nvidia/hpc_sdk/${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}/${NVHPC_VERSION}/math_libs/include")
                endif()
            endif()
            if (DepBLAS_FOUND)
                set(BLAS_INCLUDE_DIRS ${DepBLAS_INCLUDE_DIRS})
            else()
                message(WARNING "BLAS_INCLUDE_DIRS neither been provided nor been automatically"
                " detected by pkgconfig, trying to find cblas.h from possible paths...")
                find_path(BLAS_INCLUDE_DIRS
                    NAMES cblas.h
                    HINTS
                        /usr/include
                        /usr/local/include
                        /usr/include/openblas
                        /opt/homebrew/opt/openblas/include
                        /usr/local/opt/openblas/include
                        /usr/include/x86_64-linux-gnu/openblas/include
                )
            endif()
        endif()

        message(STATUS "BLAS found, Includes: ${BLAS_INCLUDE_DIRS}")

        add_compile_options(${BLAS_LINKER_FLAGS})

        add_compile_definitions(GGML_USE_BLAS)

        if (${BLAS_INCLUDE_DIRS} MATCHES "mkl" AND (${LLAMA_BLAS_VENDOR} MATCHES "Generic" OR ${LLAMA_BLAS_VENDOR} MATCHES "Intel"))
            add_compile_definitions(GGML_BLAS_USE_MKL)
        endif()

        set(GGML_HEADERS_BLAS ggml-blas.h)
        set(GGML_SOURCES_BLAS ggml-blas.cpp)

        set(LLAMA_EXTRA_LIBS     ${LLAMA_EXTRA_LIBS}     ${BLAS_LIBRARIES})
        set(LLAMA_EXTRA_INCLUDES ${LLAMA_EXTRA_INCLUDES} ${BLAS_INCLUDE_DIRS})
    else()
        message(WARNING "BLAS not found, please refer to "
        "https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors"
        " to set correct LLAMA_BLAS_VENDOR")
    endif()
endif()

if (LLAMA_LLAMAFILE)
    add_compile_definitions(GGML_USE_LLAMAFILE)

    set(GGML_HEADERS_LLAMAFILE sgemm.h)
    set(GGML_SOURCES_LLAMAFILE sgemm.cpp)
endif()

if (LLAMA_CUBLAS)
    message(WARNING "LLAMA_CUBLAS is deprecated and will be removed in the future.\nUse LLAMA_CUDA instead")
    set(LLAMA_CUDA ON)
endif()

if (LLAMA_CUDA)
    cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES

    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        message(STATUS "CUDA found")

        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # 52 == lowest CUDA 12 standard
            # 60 == f16 CUDA intrinsics
            # 61 == integer CUDA intrinsics
            # 70 == compute capability at which unrolling a loop in mul_mat_q kernels is faster
            if (LLAMA_CUDA_F16 OR LLAMA_CUDA_DMMV_F16)
                set(CMAKE_CUDA_ARCHITECTURES "60;61;70") # needed for f16 CUDA intrinsics
            else()
                set(CMAKE_CUDA_ARCHITECTURES "52;61;70") # lowest CUDA 12 standard + lowest for integer intrinsics
                #set(CMAKE_CUDA_ARCHITECTURES "OFF") # use this to compile much faster, but only F16 models work
            endif()
        endif()
        message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

        enable_language(CUDA)

        set(GGML_HEADERS_CUDA ggml-cuda.h)

        file(GLOB GGML_SOURCES_CUDA "ggml-cuda/*.cu")
        list(APPEND GGML_SOURCES_CUDA "ggml-cuda.cu")
        file(GLOB SRCS "ggml-cuda/template-instances/fattn-wmma*.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        file(GLOB SRCS "ggml-cuda/template-instances/mmq*.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})

        add_compile_definitions(GGML_USE_CUDA)
        add_compile_definitions(GGML_CUDA_USE_GRAPHS)
        if (LLAMA_CUDA_FORCE_DMMV)
            add_compile_definitions(GGML_CUDA_FORCE_DMMV)
        endif()
        if (LLAMA_CUDA_FORCE_MMQ)
            add_compile_definitions(GGML_CUDA_FORCE_MMQ)
        endif()
        if (LLAMA_CUDA_NO_VMM)
            add_compile_definitions(GGML_CUDA_NO_VMM)
        endif()
        add_compile_definitions(GGML_CUDA_DMMV_X=${LLAMA_CUDA_DMMV_X})
        add_compile_definitions(GGML_CUDA_MMV_Y=${LLAMA_CUDA_MMV_Y})
        if (DEFINED LLAMA_CUDA_DMMV_Y)
            add_compile_definitions(GGML_CUDA_MMV_Y=${LLAMA_CUDA_DMMV_Y}) # for backwards compatibility
        endif()
        if (LLAMA_CUDA_F16 OR LLAMA_CUDA_DMMV_F16)
            add_compile_definitions(GGML_CUDA_F16)
        endif()
        add_compile_definitions(K_QUANTS_PER_ITERATION=${LLAMA_CUDA_KQUANTS_ITER})
        add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${LLAMA_CUDA_PEER_MAX_BATCH_SIZE})
        if (LLAMA_CUDA_NO_PEER_COPY)
            add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
        endif()
        if (LLAMA_CUDA_FA_ALL_QUANTS)
            file(GLOB SRCS "ggml-cuda/template-instances/fattn-vec*.cu")
            list(APPEND GGML_SOURCES_CUDA ${SRCS})
            add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
        else()
            file(GLOB SRCS "ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu")
            list(APPEND GGML_SOURCES_CUDA ${SRCS})
            file(GLOB SRCS "ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu")
            list(APPEND GGML_SOURCES_CUDA ${SRCS})
            file(GLOB SRCS "ggml-cuda/template-instances/fattn-vec*f16-f16.cu")
            list(APPEND GGML_SOURCES_CUDA ${SRCS})
        endif()

        if (LLAMA_STATIC)
            if (WIN32)
                # As of 12.3.1 CUDA Toolkit for Windows does not offer a static cublas library
                set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas CUDA::cublasLt)
            else ()
                set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
            endif()
        else()
            set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cudart CUDA::cublas CUDA::cublasLt)
        endif()

        if (LLAMA_CUDA_NO_VMM)
            # No VMM requested, no need to link directly with the cuda driver lib (libcuda.so)
        else()
            set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} CUDA::cuda_driver) # required by cuDeviceGetAttribute(), cuMemGetAllocationGranularity(...), ...
        endif()
    else()
        message(WARNING "CUDA not found")
    endif()
endif()

if (LLAMA_RPC)
    add_compile_definitions(GGML_USE_RPC)

    if (WIN32)
        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} ws2_32)
    endif()

    set(GGML_HEADERS_RPC ggml-rpc.h)
    set(GGML_SOURCES_RPC ggml-rpc.cpp)
endif()

if (LLAMA_VULKAN)
    find_package(Vulkan)
    if (Vulkan_FOUND)
        message(STATUS "Vulkan found")

        set(GGML_HEADERS_VULKAN ggml-vulkan.h)
        set(GGML_SOURCES_VULKAN ggml-vulkan.cpp)

        add_compile_definitions(GGML_USE_VULKAN)

        # Workaround to the "can't dereference invalidated vector iterator" bug in clang-cl debug build
        # Posssibly relevant: https://stackoverflow.com/questions/74748276/visual-studio-no-displays-the-correct-length-of-stdvector
        if (MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)
        endif()

        if (LLAMA_VULKAN_CHECK_RESULTS)
            add_compile_definitions(GGML_VULKAN_CHECK_RESULTS)
        endif()

        if (LLAMA_VULKAN_DEBUG)
            add_compile_definitions(GGML_VULKAN_DEBUG)
        endif()

        if (LLAMA_VULKAN_MEMORY_DEBUG)
            add_compile_definitions(GGML_VULKAN_MEMORY_DEBUG)
        endif()

        if (LLAMA_VULKAN_VALIDATE)
            add_compile_definitions(GGML_VULKAN_VALIDATE)
        endif()

        if (LLAMA_VULKAN_RUN_TESTS)
            add_compile_definitions(GGML_VULKAN_RUN_TESTS)
        endif()

        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} Vulkan::Vulkan)
    else()
        message(WARNING "Vulkan not found")
    endif()
endif()

if (LLAMA_HIPBLAS)
    if (NOT EXISTS $ENV{ROCM_PATH})
        if (NOT EXISTS /opt/rocm)
            set(ROCM_PATH /usr)
        else()
            set(ROCM_PATH /opt/rocm)
        endif()
    else()
        set(ROCM_PATH $ENV{ROCM_PATH})
    endif()
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

    # CMake on Windows doesn't support the HIP language yet
    if(WIN32)
        set(CXX_IS_HIPCC TRUE)
    else()
        string(REGEX MATCH "hipcc(\.bat)?$" CXX_IS_HIPCC "${CMAKE_CXX_COMPILER}")
    endif()

    if(CXX_IS_HIPCC)
        if(LINUX)
            if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
                message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
            endif()

            message(WARNING "Setting hipcc as the C++ compiler is legacy behavior."
                    " Prefer setting the HIP compiler directly. See README for details.")
        endif()
    else()
        # Forward AMDGPU_TARGETS to CMAKE_HIP_ARCHITECTURES.
        if(AMDGPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
            set(CMAKE_HIP_ARCHITECTURES ${AMDGPU_TARGETS})
        endif()
        cmake_minimum_required(VERSION 3.21)
        enable_language(HIP)
    endif()
    find_package(hip     REQUIRED)
    find_package(hipblas REQUIRED)
    find_package(rocblas REQUIRED)

    message(STATUS "HIP and hipBLAS found")

    set(GGML_HEADERS_ROCM ggml-cuda.h)

    file(GLOB GGML_SOURCES_ROCM "ggml-cuda/*.cu")
    list(APPEND GGML_SOURCES_ROCM "ggml-cuda.cu")
    file(GLOB SRCS "ggml-cuda/template-instances/fattn-wmma*.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})
    file(GLOB SRCS "ggml-cuda/template-instances/mmq*.cu")
    list(APPEND GGML_SOURCES_ROCM ${SRCS})

    add_compile_definitions(GGML_USE_HIPBLAS GGML_USE_CUDA)

    if (LLAMA_HIP_UMA)
        add_compile_definitions(GGML_HIP_UMA)
    endif()

    if (LLAMA_CUDA_FORCE_DMMV)
        add_compile_definitions(GGML_CUDA_FORCE_DMMV)
    endif()

    if (LLAMA_CUDA_FORCE_MMQ)
        add_compile_definitions(GGML_CUDA_FORCE_MMQ)
    endif()

    if (LLAMA_CUDA_NO_PEER_COPY)
        add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
    endif()

    if (LLAMA_CUDA_FA_ALL_QUANTS)
        file(GLOB SRCS "ggml-cuda/template-instances/fattn-vec*.cu")
        list(APPEND GGML_SOURCES_ROCM ${SRCS})
        add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
    else()
        file(GLOB SRCS "ggml-cuda/template-instances/fattn-vec*q4_0-q4_0.cu")
        list(APPEND GGML_SOURCES_ROCM ${SRCS})
        file(GLOB SRCS "ggml-cuda/template-instances/fattn-vec*q8_0-q8_0.cu")
        list(APPEND GGML_SOURCES_ROCM ${SRCS})
        file(GLOB SRCS "ggml-cuda/template-instances/fattn-vec*f16-f16.cu")
        list(APPEND GGML_SOURCES_ROCM ${SRCS})
    endif()

    add_compile_definitions(GGML_CUDA_DMMV_X=${LLAMA_CUDA_DMMV_X})
    add_compile_definitions(GGML_CUDA_MMV_Y=${LLAMA_CUDA_MMV_Y})
    add_compile_definitions(K_QUANTS_PER_ITERATION=${LLAMA_CUDA_KQUANTS_ITER})

    if (CXX_IS_HIPCC)
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE CXX)
        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} hip::device)
    else()
        set_source_files_properties(${GGML_SOURCES_ROCM} PROPERTIES LANGUAGE HIP)
    endif()

    if (LLAMA_STATIC)
        message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
    endif()

    set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} PUBLIC hip::host roc::rocblas roc::hipblas)
endif()

if (LLAMA_SYCL)
    if (NOT LLAMA_SYCL_TARGET MATCHES "^(INTEL|NVIDIA)$")
        message(FATAL_ERROR "Invalid backend chosen, supported options are INTEL or NVIDIA")
    endif()

    if ( NOT DEFINED ENV{ONEAPI_ROOT})
        message(FATAL_ERROR "Not detect ENV {ONEAPI_ROOT}, please install oneAPI & source it, like: source /opt/intel/oneapi/setvars.sh")
    endif()
    #todo: AOT

    find_package(IntelSYCL REQUIRED)
    find_package(MKL REQUIRED)

    message(STATUS "SYCL found")

    add_compile_definitions(GGML_USE_SYCL)

    if (LLAMA_SYCL_F16)
        add_compile_definitions(GGML_SYCL_F16)
    endif()

    if (LLAMA_CUDA_FORCE_MMQ)
        add_compile_definitions(GGML_SYCL_FORCE_MMQ)
    endif()

    add_compile_options(-I./) #include DPCT

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    if (LLAMA_SYCL_TARGET STREQUAL "NVIDIA")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")
    endif()

    set(GGML_HEADERS_SYCL ggml-sycl.h)
    file(GLOB GGML_SOURCES_SYCL "ggml-sycl/*.cpp")
    list(APPEND GGML_SOURCES_SYCL "ggml-sycl.cpp")

    if (WIN32)
        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} IntelSYCL::SYCL_CXX MKL::MKL MKL::MKL_SYCL)
    else()
        add_compile_options(-I/${SYCL_INCLUDE_DIR})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -L${MKLROOT}/lib")
        if (LLAMA_SYCL_TARGET STREQUAL "INTEL")
            set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} -fsycl OpenCL mkl_core pthread m dl mkl_sycl_blas mkl_intel_ilp64 mkl_tbb_thread)
        elseif (LLAMA_SYCL_TARGET STREQUAL "NVIDIA")
            set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} -fsycl pthread m dl onemkl)
        endif()
    endif()
endif()

if (LLAMA_KOMPUTE)
    add_compile_definitions(VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1)
    find_package(Vulkan COMPONENTS glslc REQUIRED)
    find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)
    if (NOT glslc_executable)
        message(FATAL_ERROR "glslc not found")
    endif()

    function(compile_shader)
        set(options)
        set(oneValueArgs)
        set(multiValueArgs SOURCES)
        cmake_parse_arguments(compile_shader "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
        foreach(source ${compile_shader_SOURCES})
            get_filename_component(filename ${source} NAME)
            set(spv_file ${filename}.spv)
            add_custom_command(
                OUTPUT ${spv_file}
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source}
                ${CMAKE_CURRENT_SOURCE_DIR}/kompute-shaders/common.comp
                ${CMAKE_CURRENT_SOURCE_DIR}/kompute-shaders/op_getrows.comp
                ${CMAKE_CURRENT_SOURCE_DIR}/kompute-shaders/op_mul_mv_q_n_pre.comp
                ${CMAKE_CURRENT_SOURCE_DIR}/kompute-shaders/op_mul_mv_q_n.comp
                COMMAND ${glslc_executable} --target-env=vulkan1.2 -o ${spv_file} ${CMAKE_CURRENT_SOURCE_DIR}/${source}
                COMMENT "Compiling ${source} to ${spv_file}"
                )

            get_filename_component(RAW_FILE_NAME ${spv_file} NAME)
            set(FILE_NAME "shader${RAW_FILE_NAME}")
            string(REPLACE ".comp.spv" ".h" HEADER_FILE ${FILE_NAME})
            string(TOUPPER ${HEADER_FILE} HEADER_FILE_DEFINE)
            string(REPLACE "." "_" HEADER_FILE_DEFINE "${HEADER_FILE_DEFINE}")
            set(OUTPUT_HEADER_FILE "${HEADER_FILE}")
            message(STATUS "${HEADER_FILE} generating ${HEADER_FILE_DEFINE}")
            if(CMAKE_GENERATOR MATCHES "Visual Studio")
                add_custom_command(
                    OUTPUT ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "/*THIS FILE HAS BEEN AUTOMATICALLY GENERATED - DO NOT EDIT*/" > ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo \"\#ifndef ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo \"\#define ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "namespace kp {" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "namespace shader_data {" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/xxd -i ${RAW_FILE_NAME} >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "}}" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo \"\#endif // define ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                    DEPENDS ${spv_file} xxd
                    COMMENT "Converting to hpp: ${FILE_NAME} ${CMAKE_BINARY_DIR}/bin/$<CONFIG>/xxd"
                    )
            else()
                add_custom_command(
                    OUTPUT ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "/*THIS FILE HAS BEEN AUTOMATICALLY GENERATED - DO NOT EDIT*/" > ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo \"\#ifndef ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo \"\#define ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "namespace kp {" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "namespace shader_data {" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_BINARY_DIR}/bin/xxd -i ${RAW_FILE_NAME} >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo "}}" >> ${OUTPUT_HEADER_FILE}
                    COMMAND ${CMAKE_COMMAND} -E echo \"\#endif // define ${HEADER_FILE_DEFINE}\" >> ${OUTPUT_HEADER_FILE}
                    DEPENDS ${spv_file} xxd
                    COMMENT "Converting to hpp: ${FILE_NAME} ${CMAKE_BINARY_DIR}/bin/xxd"
                    )
            endif()
        endforeach()
    endfunction()

    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/kompute/CMakeLists.txt")
        message(STATUS "Kompute found")
        set(KOMPUTE_OPT_LOG_LEVEL Error CACHE STRING "Kompute log level")
        add_subdirectory(kompute)

        # Compile our shaders
        compile_shader(SOURCES
            kompute-shaders/op_scale.comp
            kompute-shaders/op_scale_8.comp
            kompute-shaders/op_add.comp
            kompute-shaders/op_addrow.comp
            kompute-shaders/op_mul.comp
            kompute-shaders/op_silu.comp
            kompute-shaders/op_relu.comp
            kompute-shaders/op_gelu.comp
            kompute-shaders/op_softmax.comp
            kompute-shaders/op_norm.comp
            kompute-shaders/op_rmsnorm.comp
            kompute-shaders/op_diagmask.comp
            kompute-shaders/op_mul_mat_mat_f32.comp
            kompute-shaders/op_mul_mat_f16.comp
            kompute-shaders/op_mul_mat_q8_0.comp
            kompute-shaders/op_mul_mat_q4_0.comp
            kompute-shaders/op_mul_mat_q4_1.comp
            kompute-shaders/op_mul_mat_q6_k.comp
            kompute-shaders/op_getrows_f32.comp
            kompute-shaders/op_getrows_f16.comp
            kompute-shaders/op_getrows_q4_0.comp
            kompute-shaders/op_getrows_q4_1.comp
            kompute-shaders/op_getrows_q6_k.comp
            kompute-shaders/op_rope_f16.comp
            kompute-shaders/op_rope_f32.comp
            kompute-shaders/op_cpy_f16_f16.comp
            kompute-shaders/op_cpy_f16_f32.comp
            kompute-shaders/op_cpy_f32_f16.comp
            kompute-shaders/op_cpy_f32_f32.comp
        )

        # Create a custom target for our generated shaders
        add_custom_target(generated_shaders DEPENDS
            shaderop_scale.h
            shaderop_scale_8.h
            shaderop_add.h
            shaderop_addrow.h
            shaderop_mul.h
            shaderop_silu.h
            shaderop_relu.h
            shaderop_gelu.h
            shaderop_softmax.h
            shaderop_norm.h
            shaderop_rmsnorm.h
            shaderop_diagmask.h
            shaderop_mul_mat_mat_f32.h
            shaderop_mul_mat_f16.h
            shaderop_mul_mat_q8_0.h
            shaderop_mul_mat_q4_0.h
            shaderop_mul_mat_q4_1.h
            shaderop_mul_mat_q6_k.h
            shaderop_getrows_f32.h
            shaderop_getrows_f16.h
            shaderop_getrows_q4_0.h
            shaderop_getrows_q4_1.h
            shaderop_getrows_q6_k.h
            shaderop_rope_f16.h
            shaderop_rope_f32.h
            shaderop_cpy_f16_f16.h
            shaderop_cpy_f16_f32.h
            shaderop_cpy_f32_f16.h
            shaderop_cpy_f32_f32.h
        )

        # Create a custom command that depends on the generated_shaders
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ggml-kompute.stamp
            COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/ggml-kompute.stamp
            DEPENDS generated_shaders
            COMMENT "Ensuring shaders are generated before compiling ggml-kompute.cpp"
        )

        # Add the stamp to the main sources to ensure dependency tracking
        set(GGML_SOURCES_KOMPUTE ggml-kompute.cpp ${CMAKE_CURRENT_BINARY_DIR}/ggml-kompute.stamp)
        set(GGML_HEADERS_KOMPUTE ggml-kompute.h   ${CMAKE_CURRENT_BINARY_DIR}/ggml-kompute.stamp)

        add_compile_definitions(GGML_USE_KOMPUTE)

        set(LLAMA_EXTRA_LIBS ${LLAMA_EXTRA_LIBS} kompute)
        set(LLAMA_EXTRA_INCLUDES ${LLAMA_EXTRA_INCLUDES} ${CMAKE_BINARY_DIR})
    else()
        message(WARNING "Kompute not found")
    endif()
endif()

if (LLAMA_CPU_HBM)
    find_library(memkind memkind REQUIRED)

    add_compile_definitions(GGML_USE_CPU_HBM)

    target_link_libraries(ggml PUBLIC memkind)
endif()

function(get_flags CCID CCVER)
    set(C_FLAGS "")
    set(CXX_FLAGS "")

    if (CCID MATCHES "Clang")
        set(C_FLAGS   -Wunreachable-code-break -Wunreachable-code-return)
        set(CXX_FLAGS -Wunreachable-code-break -Wunreachable-code-return -Wmissing-prototypes -Wextra-semi)

        if (
            (CCID STREQUAL "Clang"      AND CCVER VERSION_GREATER_EQUAL 3.8.0) OR
            (CCID STREQUAL "AppleClang" AND CCVER VERSION_GREATER_EQUAL 7.3.0)
        )
            list(APPEND C_FLAGS -Wdouble-promotion)
        endif()
    elseif (CCID STREQUAL "GNU")
        set(C_FLAGS   -Wdouble-promotion)
        set(CXX_FLAGS -Wno-array-bounds)

        if (CCVER VERSION_GREATER_EQUAL 7.1.0)
            list(APPEND CXX_FLAGS -Wno-format-truncation)
        endif()
        if (CCVER VERSION_GREATER_EQUAL 8.1.0)
            list(APPEND CXX_FLAGS -Wextra-semi)
        endif()
    endif()

    set(GF_C_FLAGS   ${C_FLAGS}   PARENT_SCOPE)
    set(GF_CXX_FLAGS ${CXX_FLAGS} PARENT_SCOPE)
endfunction()

if (LLAMA_FATAL_WARNINGS)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        list(APPEND C_FLAGS   -Werror)
        list(APPEND CXX_FLAGS -Werror)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(/WX)
    endif()
endif()

if (LLAMA_ALL_WARNINGS)
    if (NOT MSVC)
        list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
        list(APPEND C_FLAGS       -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                  -Werror=implicit-int -Werror=implicit-function-declaration)
        list(APPEND CXX_FLAGS     -Wmissing-declarations -Wmissing-noreturn)

        list(APPEND C_FLAGS   ${WARNING_FLAGS})
        list(APPEND CXX_FLAGS ${WARNING_FLAGS})

        get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

        add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                            "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
    else()
        # todo : msvc
        set(C_FLAGS   "")
        set(CXX_FLAGS "")
    endif()
endif()

set(CUDA_CXX_FLAGS "")

if (LLAMA_CUDA)
    set(CUDA_FLAGS -use_fast_math)

    if (LLAMA_FATAL_WARNINGS)
        list(APPEND CUDA_FLAGS -Werror all-warnings)
    endif()

    if (LLAMA_ALL_WARNINGS AND NOT MSVC)
        set(NVCC_CMD ${CMAKE_CUDA_COMPILER} .c)
        if (NOT CMAKE_CUDA_HOST_COMPILER STREQUAL "")
            list(APPEND NVCC_CMD -ccbin ${CMAKE_CUDA_HOST_COMPILER})
        endif()

        execute_process(
            COMMAND ${NVCC_CMD} -Xcompiler --version
            OUTPUT_VARIABLE CUDA_CCFULLVER
            ERROR_QUIET
        )

        if (NOT CUDA_CCFULLVER MATCHES clang)
            set(CUDA_CCID "GNU")
            execute_process(
                COMMAND ${NVCC_CMD} -Xcompiler "-dumpfullversion -dumpversion"
                OUTPUT_VARIABLE CUDA_CCVER
                ERROR_QUIET
            )
        else()
            if (CUDA_CCFULLVER MATCHES Apple)
                set(CUDA_CCID "AppleClang")
            else()
                set(CUDA_CCID "Clang")
            endif()
            string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
        endif()

        message("-- CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")

        get_flags(${CUDA_CCID} ${CUDA_CCVER})
        list(APPEND CUDA_CXX_FLAGS ${CXX_FLAGS} ${GF_CXX_FLAGS})  # This is passed to -Xcompiler later
    endif()

    if (NOT MSVC)
        list(APPEND CUDA_CXX_FLAGS -Wno-pedantic)
    endif()
endif()

if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

if (LLAMA_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

if (LLAMA_CCACHE)
    find_program(LLAMA_CCACHE_FOUND ccache)
    if (LLAMA_CCACHE_FOUND)
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
        set(ENV{CCACHE_SLOPPINESS} time_macros)
        message(STATUS "ccache found, compilation results will be cached. Disable with LLAMA_CCACHE=OFF.")
    else()
        message(STATUS "Warning: ccache not found - consider installing it for faster compilation or disable this warning with LLAMA_CCACHE=OFF")
    endif ()
endif()

# this version of Apple ld64 is buggy
execute_process(
    COMMAND ${CMAKE_C_COMPILER} ${CMAKE_EXE_LINKER_FLAGS} -Wl,-v
    ERROR_VARIABLE output
    OUTPUT_QUIET
)

if (output MATCHES "dyld-1015\.7")
    add_compile_definitions(HAVE_BUGGY_APPLE_LINKER)
endif()

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (MSVC)
    string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" CMAKE_GENERATOR_PLATFORM_LWR)
    message(STATUS "CMAKE_GENERATOR_PLATFORM: ${CMAKE_GENERATOR_PLATFORM}")
else ()
    set(CMAKE_GENERATOR_PLATFORM_LWR "")
endif ()

if (NOT MSVC)
    if (LLAMA_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (LLAMA_GPROF)
        add_compile_options(-pg)
    endif()
endif()

set(ARCH_FLAGS "")

if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR CMAKE_GENERATOR_PLATFORM_LWR STREQUAL "arm64" OR
    (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
     CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm.*|ARM64)$"))
    message(STATUS "ARM detected")
    if (MSVC)
        add_compile_definitions(__aarch64__) # MSVC defines _M_ARM64 instead
        add_compile_definitions(__ARM_NEON)
        add_compile_definitions(__ARM_FEATURE_FMA)

        set(CMAKE_REQUIRED_FLAGS_PREV ${CMAKE_REQUIRED_FLAGS})
        string(JOIN " " CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS} "/arch:armv8.2")
        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int8x16_t _a, _b; int32x4_t _s = vdotq_s32(_s, _a, _b); return 0; }" GGML_COMPILER_SUPPORT_DOTPROD)
        if (GGML_COMPILER_SUPPORT_DOTPROD)
            add_compile_definitions(__ARM_FEATURE_DOTPROD)
        endif ()
        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int8x16_t _a, _b; int32x4_t _s = vmlaq_f32(_s, _a, _b); return 0; }" GGML_COMPILER_SUPPORT_MATMUL_INT8)
        if (GGML_COMPILER_SUPPORT_MATMUL_INT8)
            add_compile_definitions(__ARM_FEATURE_MATMUL_INT8)
        endif ()

        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float16_t _a; float16x8_t _s = vdupq_n_f16(_a); return 0; }" GGML_COMPILER_SUPPORT_FP16_VECTOR_ARITHMETIC)
        if (GGML_COMPILER_SUPPORT_FP16_VECTOR_ARITHMETIC)
            add_compile_definitions(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        endif ()
        set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_PREV})
    else()
        check_cxx_compiler_flag(-mfp16-format=ieee COMPILER_SUPPORTS_FP16_FORMAT_I3E)
        if (NOT "${COMPILER_SUPPORTS_FP16_FORMAT_I3E}" STREQUAL "")
            list(APPEND ARCH_FLAGS -mfp16-format=ieee)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
            # Raspberry Pi 1, Zero
            list(APPEND ARCH_FLAGS -mfpu=neon-fp-armv8 -mno-unaligned-access)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
            if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
                # Android armeabi-v7a
                list(APPEND ARCH_FLAGS -mfpu=neon-vfpv4 -mno-unaligned-access -funsafe-math-optimizations)
            else()
                # Raspberry Pi 2
                list(APPEND ARCH_FLAGS -mfpu=neon-fp-armv8 -mno-unaligned-access -funsafe-math-optimizations)
            endif()
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
            # Android arm64-v8a
            # Raspberry Pi 3, 4, Zero 2 (32-bit)
            list(APPEND ARCH_FLAGS -mno-unaligned-access)
        endif()
        if (LLAMA_SVE)
            list(APPEND ARCH_FLAGS -march=armv8.6-a+sve)
        endif()
    endif()
elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR CMAKE_GENERATOR_PLATFORM_LWR MATCHES "^(x86_64|i686|amd64|x64|win32)$" OR
        (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
         CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|i686|AMD64)$"))
    message(STATUS "x86 detected")
    if (MSVC)
        # instruction set detection for MSVC only
        if (LLAMA_NATIVE)
            include(cmake/FindSIMD.cmake)
        endif ()
        if (LLAMA_AVX512)
            list(APPEND ARCH_FLAGS /arch:AVX512)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
            if (LLAMA_AVX512_VBMI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VBMI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VBMI__>)
            endif()
            if (LLAMA_AVX512_VNNI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VNNI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VNNI__>)
            endif()
            if (LLAMA_AVX512_BF16)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512BF16__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512BF16__>)
            endif()
        elseif (LLAMA_AVX2)
            list(APPEND ARCH_FLAGS /arch:AVX2)
        elseif (LLAMA_AVX)
            list(APPEND ARCH_FLAGS /arch:AVX)
        endif()
    else()
        if (LLAMA_NATIVE)
            list(APPEND ARCH_FLAGS -march=native)
        endif()
        if (LLAMA_F16C)
            list(APPEND ARCH_FLAGS -mf16c)
        endif()
        if (LLAMA_FMA)
            list(APPEND ARCH_FLAGS -mfma)
        endif()
        if (LLAMA_AVX)
            list(APPEND ARCH_FLAGS -mavx)
        endif()
        if (LLAMA_AVX2)
            list(APPEND ARCH_FLAGS -mavx2)
        endif()
        if (LLAMA_AVX512)
            list(APPEND ARCH_FLAGS -mavx512f)
            list(APPEND ARCH_FLAGS -mavx512bw)
        endif()
        if (LLAMA_AVX512_VBMI)
            list(APPEND ARCH_FLAGS -mavx512vbmi)
        endif()
        if (LLAMA_AVX512_VNNI)
            list(APPEND ARCH_FLAGS -mavx512vnni)
        endif()
        if (LLAMA_AVX512_BF16)
            list(APPEND ARCH_FLAGS -mavx512bf16)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
    message(STATUS "PowerPC detected")
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64le")
        list(APPEND ARCH_FLAGS -mcpu=powerpc64le)
    else()
        list(APPEND ARCH_FLAGS -mcpu=native -mtune=native)
        #TODO: Add  targets for Power8/Power9 (Altivec/VSX) and Power10(MMA) and query for big endian systems (ppc64/le/be)
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
    message(STATUS "loongarch64 detected")

    list(APPEND ARCH_FLAGS -march=loongarch64)
    if (LLAMA_LASX)
        list(APPEND ARCH_FLAGS -mlasx)
    endif()
    if (LLAMA_LSX)
        list(APPEND ARCH_FLAGS -mlsx)
    endif()

else()
    message(STATUS "Unknown architecture")
endif()

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${ARCH_FLAGS}>")
add_compile_options("$<$<COMPILE_LANGUAGE:C>:${ARCH_FLAGS}>")

if (LLAMA_CUDA)
    list(APPEND CUDA_CXX_FLAGS ${ARCH_FLAGS})
    list(JOIN CUDA_CXX_FLAGS " " CUDA_CXX_FLAGS_JOINED)  # pass host compiler flags as a single argument
    if (NOT CUDA_CXX_FLAGS_JOINED STREQUAL "")
        list(APPEND CUDA_FLAGS -Xcompiler ${CUDA_CXX_FLAGS_JOINED})
    endif()
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
endif()

if (MINGW)
    # Target Windows 8 for PrefetchVirtualMemory
    add_compile_definitions(_WIN32_WINNT=${LLAMA_WIN_VER})
endif()

#
# POSIX conformance
#

# clock_gettime came in POSIX.1b (1993)
# CLOCK_MONOTONIC came in POSIX.1-2001 / SUSv3 as optional
# posix_memalign came in POSIX.1-2001 / SUSv3
# M_PI is an XSI extension since POSIX.1-2001 / SUSv3, came in XPG1 (1985)
add_compile_definitions(_XOPEN_SOURCE=600)

# Somehow in OpenBSD whenever POSIX conformance is specified
# some string functions rely on locale_t availability,
# which was introduced in POSIX.1-2008, forcing us to go higher
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    remove_definitions(-D_XOPEN_SOURCE=600)
    add_compile_definitions(_XOPEN_SOURCE=700)
endif()

# Data types, macros and functions related to controlling CPU affinity and
# some memory allocation are available on Linux through GNU extensions in libc
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_definitions(_GNU_SOURCE)
endif()

# RLIMIT_MEMLOCK came in BSD, is not specified in POSIX.1,
# and on macOS its availability depends on enabling Darwin extensions
# similarly on DragonFly, enabling BSD extensions is necessary
if (
    CMAKE_SYSTEM_NAME MATCHES "Darwin" OR
    CMAKE_SYSTEM_NAME MATCHES "iOS" OR
    CMAKE_SYSTEM_NAME MATCHES "tvOS" OR
    CMAKE_SYSTEM_NAME MATCHES "DragonFly"
)
    add_compile_definitions(_DARWIN_C_SOURCE)
endif()

# alloca is a non-standard interface that is not visible on BSDs when
# POSIX conformance is specified, but not all of them provide a clean way
# to enable it in such cases
if (CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    add_compile_definitions(__BSD_VISIBLE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "NetBSD")
    add_compile_definitions(_NETBSD_SOURCE)
endif()
if (CMAKE_SYSTEM_NAME MATCHES "OpenBSD")
    add_compile_definitions(_BSD_SOURCE)
endif()

#
# libraries
#

# ggml

add_library(ggml OBJECT
            ggml.c
            ggml.h
            ggml-alloc.c
            ggml-alloc.h
            ggml-backend.c
            ggml-backend.h
            ggml-quants.c
            ggml-quants.h
            ${GGML_SOURCES_CUDA}      ${GGML_HEADERS_CUDA}
            ${GGML_SOURCES_METAL}     ${GGML_HEADERS_METAL}
            ${GGML_SOURCES_RPC}       ${GGML_HEADERS_RPC}
            ${GGML_SOURCES_EXTRA}     ${GGML_HEADERS_EXTRA}
            ${GGML_SOURCES_SYCL}      ${GGML_HEADERS_SYCL}
            ${GGML_SOURCES_KOMPUTE}   ${GGML_HEADERS_KOMPUTE}
            ${GGML_SOURCES_VULKAN}    ${GGML_HEADERS_VULKAN}
            ${GGML_SOURCES_ROCM}      ${GGML_HEADERS_ROCM}
            ${GGML_SOURCES_BLAS}      ${GGML_HEADERS_BLAS}
            ${GGML_SOURCES_LLAMAFILE} ${GGML_HEADERS_LLAMAFILE}
            )

target_include_directories(ggml PUBLIC . ${LLAMA_EXTRA_INCLUDES})
target_compile_features   (ggml PUBLIC c_std_11) # don't bump

target_link_libraries(ggml PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})

add_library(ggml_static STATIC $<TARGET_OBJECTS:ggml>)

if (BUILD_SHARED_LIBS)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(ggml_shared SHARED $<TARGET_OBJECTS:ggml>)
    target_link_libraries(ggml_shared PUBLIC Threads::Threads ${LLAMA_EXTRA_LIBS})
    install(TARGETS ggml_shared LIBRARY)
endif()

# llama

add_library(llama
            llama.cpp
            llama.h
            unicode.h
            unicode.cpp
            unicode-data.cpp
            )

target_include_directories(llama PUBLIC .)
target_compile_features   (llama PUBLIC cxx_std_11) # don't bump

target_link_libraries(llama PRIVATE
    ggml
    ${LLAMA_EXTRA_LIBS}
    )

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_SHARED LLAMA_BUILD)
    if (LLAMA_METAL)
        set_target_properties(llama PROPERTIES RESOURCE "${CMAKE_CURRENT_SOURCE_DIR}/ggml-metal.metal")
    endif()
endif()


#
# install
#

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(LLAMA_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}
    CACHE PATH "Location of header files")
set(LLAMA_LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}
    CACHE PATH "Location of library files")
set(LLAMA_BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR}
    CACHE PATH "Location of binary files")
set(LLAMA_BUILD_NUMBER ${BUILD_NUMBER})
set(LLAMA_BUILD_COMMIT ${BUILD_COMMIT})
set(LLAMA_INSTALL_VERSION 0.0.${BUILD_NUMBER})
get_directory_property(LLAMA_TRANSIENT_DEFINES COMPILE_DEFINITIONS)

configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/scripts/LlamaConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/LlamaConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Llama
    PATH_VARS LLAMA_INCLUDE_INSTALL_DIR
              LLAMA_LIB_INSTALL_DIR
              LLAMA_BIN_INSTALL_DIR )

write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/LlamaConfigVersion.cmake
    VERSION ${LLAMA_INSTALL_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/LlamaConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/LlamaConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Llama)

set(GGML_PUBLIC_HEADERS "ggml.h" "ggml-alloc.h" "ggml-backend.h"
        "${GGML_HEADERS_CUDA}"
        "${GGML_HEADERS_METAL}"
        "${GGML_HEADERS_EXTRA}")

set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${GGML_PUBLIC_HEADERS}")
install(TARGETS ggml PUBLIC_HEADER)

set_target_properties(llama PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/llama.h)
install(TARGETS llama LIBRARY PUBLIC_HEADER)

install(
    FILES convert-hf-to-gguf.py
    PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
    DESTINATION ${CMAKE_INSTALL_BINDIR})
if (LLAMA_METAL)
    install(
        FILES ggml-metal.metal
        PERMISSIONS
            OWNER_READ
            OWNER_WRITE
            GROUP_READ
            WORLD_READ
        DESTINATION ${CMAKE_INSTALL_BINDIR})
    if (NOT LLAMA_METAL_EMBED_LIBRARY)
        install(
            FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/default.metallib
            DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()
endif()

configure_file(cmake/llama.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/llama.pc"
        @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/llama.pc"
        DESTINATION lib/pkgconfig)

#
# programs, examples and tests
#

add_subdirectory(common)

if (LLAMA_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    include(CTest)
    add_subdirectory(tests)
endif ()

if (LLAMA_BUILD_EXAMPLES)
    add_subdirectory(examples)
    add_subdirectory(pocs)
endif()
