cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES

find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found")

    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # native == GPUs available at build time
        # 50     == Maxwell, lowest CUDA 12 standard
        # 60     == P100, FP16 CUDA intrinsics
        # 61     == Pascal, __dp4a instruction (per-byte integer dot product)
        # 70     == V100, FP16 tensor cores
        # 75     == Turing, int8 tensor cores
        # 80     == Ampere, asynchronous data loading, faster tensor core instructions
        # 86     == RTX 3000, needs CUDA v11.1
        # 89     == RTX 4000, needs CUDA v11.8
        # 90     == Hopper H100/200, needs CUDA v11.8
        # 120    == Blackwell, needs CUDA v12.8, FP4 tensor cores
        #
        # XX-virtual == compile CUDA code as PTX, do JIT compilation to binary code on first run
        # XX-real    == compile CUDA code as device code for this specific architecture
        # no suffix  == compile as both PTX and device code
        #
        # The default behavior for a non-native is to build virtual architectures as needed to cover all features needed
        #     for best performance and to also build real architectures for the most commonly used GPUs.
        if (GGML_NATIVE AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.6" AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(CMAKE_CUDA_ARCHITECTURES "native")
        else()
            if (CUDAToolkit_VERSION VERSION_LESS "13")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 50-virtual 61-virtual 70-virtual)
            endif ()

            list(APPEND CMAKE_CUDA_ARCHITECTURES 75-virtual 80-virtual 86-real)

            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 89-real 90-virtual)
            endif()

            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
                # The CUDA architecture 120f-virtual would in principle work for Blackwell support
                #     but the newly added "f" suffix conflicted with a preexising regex for validating CUDA architectures in CMake.
                # So either a recent CMake version or one with the backported fix is needed.
                # The following versions should work:
                #   - CMake >= v3.31.8 && CMake < v4.0.0
                #   - CMake >= v4.0.2
                # This is NOT documented in the CMake release notes,
                #     check Modules/Internal/CMakeCUDAArchitecturesValidate.cmake in the CMake git repository instead.
                # However, the architectures 120a-real and 121a-real should work with basically any CMake version and
                #     until the release of e.g. Rubin there is no benefit to shipping virtual architectures for Blackwell.
                list(APPEND CMAKE_CUDA_ARCHITECTURES 120a-real)
            endif()
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 121a-real)
            endif()

            # Consumer Blackwell: 100a = SM100 (RTX 5090), 101a = SM101 (RTX 5060 Ti)
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 100a-real 101a-real)
            endif()
        endif()
    endif()

    enable_language(CUDA)

    # TODO: Remove once CCCL 3.2 has been released and bundled with CUDA Toolkit
    if (GGML_CUDA_CUB_3DOT2)
        include(FetchContent)

        FetchContent_Declare(
            CCCL
            GIT_REPOSITORY https://github.com/nvidia/cccl.git
            GIT_TAG        v3.2.0
            GIT_SHALLOW    TRUE
        )

        FetchContent_MakeAvailable(CCCL)
    endif()

    # Replace any plain 12X CUDA architectures with their "architecture-specific" equivalents 12Xa.
    # 12X is forwards-compatible, 12Xa is not.
    # Notably the Blackwell FP4 tensor core instructions are not forwards compatible and therefore need 12Xa.
    # But while 12X vs. 12Xa can be checked in device code there is (to my knowledge) no easy way to do the same check in host code.
    # So for now just replace all instances of 12X with 12Xa, this should be fine until Rubin is released.
    foreach(ARCHS IN ITEMS CMAKE_CUDA_ARCHITECTURES CMAKE_CUDA_ARCHITECTURES_NATIVE)
        set(FIXED_ARCHS "")
        foreach(ARCH IN LISTS ${ARCHS})
            if (ARCH MATCHES "^12[0-9](-real|-virtual)?$")
                string(REGEX REPLACE "^(12[0-9])((-real|-virtual)?)$" "\\1a\\2" FIXED_ARCH ${ARCH})
                message(STATUS "Replacing ${ARCH} in ${ARCHS} with ${FIXED_ARCH}")
                list(APPEND FIXED_ARCHS "${FIXED_ARCH}")
            else()
                list(APPEND FIXED_ARCHS "${ARCH}")
            endif()
        endforeach()
        set(${ARCHS} ${FIXED_ARCHS})
    endforeach()

    # If we try to compile a "native" build it will use the 12X architectures and fail.
    # So we should instead use the native architectures as determined by CMake after replacing 12X with 12Xa.
    # But if at the time of the build no GPUs are connected at all CMAKE_CUDA_ARCHITECTURES will contain garbage that we should not use.
    if (CMAKE_CUDA_ARCHITECTURES STREQUAL "native" AND CMAKE_CUDA_ARCHITECTURES_NATIVE MATCHES "^[0-9]+(a|f)?(-real|-virtual)?(;[0-9]+(a|f)?(-real|-virtual)?|;)*$")
        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
    endif()
    message(STATUS "Using CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} CMAKE_CUDA_ARCHITECTURES_NATIVE=${CMAKE_CUDA_ARCHITECTURES_NATIVE}")

    # Detect WGMMA PTX support (SM 100/120 warpgroup MMA).
    # CUDA 13.2+ recognizes SM 120a but does not support wgmma PTX instructions.
    # This try_compile test auto-detects support and defines GGML_CUDA_HAS_WGMMA.
    if (CMAKE_CUDA_ARCHITECTURES MATCHES "1[02][01]a?")
        include(CheckLanguage)
        set(CMAKE_CUDA_ARCHITECTURES_SAVE "${CMAKE_CUDA_ARCHITECTURES}")
        # Use the first Blackwell architecture for the test
        if (CMAKE_CUDA_ARCHITECTURES MATCHES "12[01]a?")
            set(CMAKE_CUDA_ARCHITECTURES "120a")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "100a")
        endif()

        file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/check_wgmma/test.cu"
            "__device__ void test_wgmma() {"
            "    uint32_t d[64] = {};"
            "    __shared__ alignas(16) uint8_t sa[64][32], sb[32][8];"
            "    const uint32_t *a = reinterpret_cast<uint32_t*>(sa);"
            "    const uint32_t *b = reinterpret_cast<uint32_t*>(sb);"
            "    asm volatile("
            "        \"wgmma.mma_sync.sync.aligned.m64n32h8.s32.s8.s8.s32 \""
            "        \"{%%0<16>}, [%%1], [%%2], {%%3<16>}, %%4;\""
            "        : \"+r\"(d[0]), \"+r\"(d[1]), \"+r\"(d[2]), \"+r\"(d[3]),"
            "          \"+r\"(d[4]), \"+r\"(d[5]), \"+r\"(d[6]), \"+r\"(d[7]),"
            "          \"+r\"(d[8]), \"+r\"(d[9]), \"+r\"(d[10]), \"+r\"(d[11]),"
            "          \"+r\"(d[12]), \"+r\"(d[13]), \"+r\"(d[14]), \"+r\"(d[15]),"
            "          \"+r\"(d[16]), \"+r\"(d[17]), \"+r\"(d[18]), \"+r\"(d[19]),"
            "          \"+r\"(d[20]), \"+r\"(d[21]), \"+r\"(d[22]), \"+r\"(d[23]),"
            "          \"+r\"(d[24]), \"+r\"(d[25]), \"+r\"(d[26]), \"+r\"(d[27]),"
            "          \"+r\"(d[28]), \"+r\"(d[29]), \"+r\"(d[30]), \"+r\"(d[31]),"
            "          \"+r\"(d[32]), \"+r\"(d[33]), \"+r\"(d[34]), \"+r\"(d[35]),"
            "          \"+r\"(d[36]), \"+r\"(d[37]), \"+r\"(d[38]), \"+r\"(d[39]),"
            "          \"+r\"(d[40]), \"+r\"(d[41]), \"+r\"(d[42]), \"+r\"(d[43]),"
            "          \"+r\"(d[44]), \"+r\"(d[45]), \"+r\"(d[46]), \"+r\"(d[47]),"
            "          \"+r\"(d[48]), \"+r\"(d[49]), \"+r\"(d[50]), \"+r\"(d[51]),"
            "          \"+r\"(d[52]), \"+r\"(d[53]), \"+r\"(d[54]), \"+r\"(d[55]),"
            "          \"+r\"(d[56]), \"+r\"(d[57]), \"+r\"(d[58]), \"+r\"(d[59]),"
            "          \"+r\"(d[60]), \"+r\"(d[61]), \"+r\"(d[62]), \"+r\"(d[63])"
            "        : \"l\"(reinterpret_cast<unsigned long long>(__cvta_generic_to_shared(a))),"
            "          \"l\"(reinterpret_cast<unsigned long long>(__cvta_generic_to_shared(b))),"
            "          \"r\"(0)"
            "        : \"memory\");"
            "}"
            "int main() { return 0; }"
            )

        try_compile(GGML_CUDA_HAS_WGMMA_RESULT
            "${CMAKE_CURRENT_BINARY_DIR}/check_wgmma"
            "${CMAKE_CURRENT_BINARY_DIR}/check_wgmma"
            SOURCES "${CMAKE_CURRENT_BINARY_DIR}/check_wgmma/test.cu"
            COMPILE_DEFINITIONS -std=c++17
            )

        if (GGML_CUDA_HAS_WGMMA_RESULT)
            message(STATUS "WGMMA PTX support detected for Blackwell")
            set(GGML_CUDA_HAS_WGMMA ON CACHE INTERNAL "WGMMA PTX supported by this CUDA toolkit")
        else()
            message(STATUS "WGMMA PTX not yet supported by this CUDA toolkit (will use fallback)")
            set(GGML_CUDA_HAS_WGMMA OFF CACHE INTERNAL "WGMMA PTX supported by this CUDA toolkit")
        endif()

        set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_SAVE}")
    endif()

    file(GLOB   GGML_HEADERS_CUDA "*.cuh")
    list(APPEND GGML_HEADERS_CUDA "../../include/ggml-cuda.h")

    file(GLOB   GGML_SOURCES_CUDA "*.cu")
    file(GLOB   SRCS "template-instances/fattn-tile*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/fattn-mma*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/mmq*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS "template-instances/mmf*.cu")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})

    if (GGML_CUDA_FA_ALL_QUANTS)
        file(GLOB   SRCS "template-instances/fattn-vec*.cu")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        add_compile_definitions(GGML_CUDA_FA_ALL_QUANTS)
    else()
        list(APPEND GGML_SOURCES_CUDA
            template-instances/fattn-vec-instance-f16-f16.cu
            template-instances/fattn-vec-instance-q4_0-q4_0.cu
            template-instances/fattn-vec-instance-q8_0-q8_0.cu
            template-instances/fattn-vec-instance-bf16-bf16.cu)
    endif()

    ggml_add_backend_library(ggml-cuda
                             ${GGML_HEADERS_CUDA}
                             ${GGML_SOURCES_CUDA}
                            )

    if (GGML_CUDA_HAS_WGMMA)
        target_compile_definitions(ggml-cuda PUBLIC GGML_CUDA_HAS_WGMMA)
    endif()

    add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE})

    if (GGML_CUDA_GRAPHS)
        add_compile_definitions(GGML_CUDA_USE_GRAPHS)
    endif()

    if (GGML_CUDA_FORCE_MMQ)
        add_compile_definitions(GGML_CUDA_FORCE_MMQ)
    endif()

    if (GGML_CUDA_FORCE_CUBLAS)
        add_compile_definitions(GGML_CUDA_FORCE_CUBLAS)
    endif()

    if (GGML_CUDA_NO_VMM)
        add_compile_definitions(GGML_CUDA_NO_VMM)
    endif()

    if (NOT GGML_CUDA_FA)
        add_compile_definitions(GGML_CUDA_NO_FA)
    endif()

    if (GGML_CUDA_NO_PEER_COPY)
        add_compile_definitions(GGML_CUDA_NO_PEER_COPY)
    endif()

    if (GGML_STATIC)
        if (WIN32)
            # As of 12.3.1 CUDA Toolkit for Windows does not offer a static cublas library
            target_link_libraries(ggml-cuda PRIVATE CUDA::cudart_static CUDA::cublas)
        else ()
            if (GGML_CUDA_CUB_3DOT2)
                target_link_libraries(ggml-cuda PRIVATE  CCCL::CCCL)
            endif()
            if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "10.1")
                target_link_libraries(ggml-cuda PRIVATE  CUDA::cudart_static CUDA::cublas_static CUDA::cublasLt_static)
            else()
                target_link_libraries(ggml-cuda PRIVATE  CUDA::cudart_static CUDA::cublas_static)
            endif()
        endif()
    else()
        if (GGML_CUDA_CUB_3DOT2)
            target_link_libraries(ggml-cuda PRIVATE  CCCL::CCCL)
        endif()
        target_link_libraries(ggml-cuda PRIVATE CUDA::cudart CUDA::cublas)
    endif()

    if (GGML_CUDA_NO_VMM)
        # No VMM requested, no need to link directly with the cuda driver lib (libcuda.so)
    else()
        target_link_libraries(ggml-cuda PRIVATE CUDA::cuda_driver)
    endif()

    if (GGML_CUDA_NCCL)
        find_package(NCCL)
        if (NCCL_FOUND)
            add_compile_definitions(GGML_USE_NCCL)
            target_link_libraries(ggml-cuda PRIVATE NCCL::NCCL)
        else()
            message(STATUS "Warning: NCCL not found, performance for multiple CUDA GPUs will be suboptimal")
        endif()
    endif()

    set(CUDA_CXX_FLAGS "")

    set(CUDA_FLAGS -use_fast_math -extended-lambda)

    if (GGML_CUDA_DEBUG)
        list(APPEND CUDA_FLAGS -lineinfo)
        add_compile_definitions(GGML_CUDA_DEBUG)
    endif()

    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
        # Options are:
        # - none (not recommended)
        # - speed (nvcc's default)
        # - balance
        # - size
        list(APPEND CUDA_FLAGS -compress-mode=${GGML_CUDA_COMPRESSION_MODE})
    endif()

    if (GGML_FATAL_WARNINGS)
        list(APPEND CUDA_FLAGS -Werror all-warnings)
    endif()

    if (GGML_ALL_WARNINGS AND NOT MSVC)
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
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        else()
            if (CUDA_CCFULLVER MATCHES Apple)
                set(CUDA_CCID "AppleClang")
            else()
                set(CUDA_CCID "Clang")
            endif()
            string(REGEX REPLACE "^.* version ([0-9.]*).*$" "\\1" CUDA_CCVER ${CUDA_CCFULLVER})
        endif()

        message(STATUS "CUDA host compiler is ${CUDA_CCID} ${CUDA_CCVER}")

        ggml_get_flags(${CUDA_CCID} ${CUDA_CCVER})
        list(APPEND CUDA_CXX_FLAGS ${CXX_FLAGS} ${GF_CXX_FLAGS})  # This is passed to -Xcompiler later
    endif()

    if (NOT MSVC)
        list(APPEND CUDA_CXX_FLAGS -Wno-pedantic)
    else()
        # CCCL 3.2 onwards will require a cpp-standard-compliant preprocessor for MSVC
        # https://github.com/NVIDIA/cccl/pull/6827
        list(APPEND CUDA_CXX_FLAGS /Zc:preprocessor)
    endif()

    list(JOIN   CUDA_CXX_FLAGS " " CUDA_CXX_FLAGS_JOINED)  # pass host compiler flags as a single argument

    if (NOT CUDA_CXX_FLAGS_JOINED STREQUAL "")
        list(APPEND CUDA_FLAGS -Xcompiler ${CUDA_CXX_FLAGS_JOINED})
    endif()

    target_compile_options(ggml-cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
else()
    message(FATAL_ERROR "CUDA Toolkit not found")
endif()
