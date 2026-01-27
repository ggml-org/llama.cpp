if (NOT DEFINED GGML_SYCL_TARGET)
    set(GGML_SYCL_TARGET "INTEL")
endif()

message(STATUS  "GGML_SYCL_TARGET=${GGML_SYCL_TARGET}")

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
option(GGML_SYCL_MMQ_XMX "Enable experimental XMX MMQ kernels for SYCL backend" OFF)
option(GGML_SYCL_XMX_GEMM "Enable experimental XMX GEMM for quantized matmul (known to be slower)" OFF)
option(GGML_SYCL_DEBUG_SOA_LAYOUT "Enable SoA layout verification debug checks" OFF)
option(GGML_SYCL_QUANTIZER_DEBUG "Enable quantizer kernel debug output" OFF)
option(GGML_SYCL_ASAN_PARTIAL "Enable ASan for ggml-sycl.cpp and unified-cache.cpp only" OFF)

file(GLOB   GGML_SOURCES_SYCL "*.cpp")
list(FILTER GGML_SOURCES_SYCL EXCLUDE REGEX "test_joint_matrix.*\\.cpp$")
list(FILTER GGML_SOURCES_SYCL EXCLUDE REGEX "test_tensor_parallel\\.cpp$")

if (GGML_SYCL_MMQ_XMX)
    message(STATUS "Compiling with experimental GGML_SYCL_MMQ_XMX kernels.")
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_MMQ_XMX)
    list(APPEND GGML_SOURCES_SYCL mmq_xmx.cpp)
    list(APPEND GGML_HEADERS_SYCL mmq_xmx.hpp)
endif()

if (GGML_SYCL_DEBUG_SOA_LAYOUT)
    message(STATUS "Compiling with GGML_SYCL_DEBUG_SOA_LAYOUT enabled - will verify SoA layout at runtime.")
    add_compile_definitions(GGML_SYCL_DEBUG_SOA_LAYOUT=1)
endif()

if (GGML_SYCL_QUANTIZER_DEBUG)
    message(STATUS "Compiling with GGML_SYCL_QUANTIZER_DEBUG enabled - will print quantizer d values.")
    add_compile_definitions(GGML_SYCL_QUANTIZER_DEBUG=1)
endif()

target_sources(ggml-sycl PRIVATE ${GGML_HEADERS_SYCL} ${GGML_SOURCES_SYCL})

if (GGML_SYCL_ASAN_PARTIAL)
    message(STATUS "GGML_SYCL_ASAN_PARTIAL: enabling ASan for ggml-sycl.cpp + unified-cache.cpp")
    set_source_files_properties(ggml-sycl.cpp unified-cache.cpp
        PROPERTIES COMPILE_OPTIONS "-fsanitize=address;-fno-omit-frame-pointer")
    target_link_options(ggml-sycl PRIVATE "-fsanitize=address")
endif()

if (WIN32)
    # To generate a Visual Studio solution, using Intel C++ Compiler for ggml-sycl is mandatory
    if( ${CMAKE_GENERATOR} MATCHES "Visual Studio" AND NOT (${CMAKE_GENERATOR_TOOLSET} MATCHES "Intel C"))
        set_target_properties(ggml-sycl PROPERTIES VS_PLATFORM_TOOLSET "Intel C++ Compiler 2025")
        set(CMAKE_CXX_COMPILER "icx")
        set(CMAKE_CXX_COMPILER_ID "IntelLLVM")
    endif()
endif()

find_package(IntelSYCL)
if (IntelSYCL_FOUND)
    # Use oneAPI CMake when possible
    target_link_libraries(ggml-sycl PRIVATE IntelSYCL::SYCL_CXX)
else()
    # Fallback to the simplest way of enabling SYCL when using intel/llvm nightly for instance
    target_compile_options(ggml-sycl PRIVATE "-fsycl")
    target_link_options(ggml-sycl PRIVATE "-fsycl")
endif()

# Add profiling-friendly compiler flags for VTune kernel name visibility
# -fsycl-instrument-device-code: enables profiling tools to map GPU activities to kernel function names
# -fdebug-info-for-profiling: generates debug info optimized for profilers (line tables, etc.)
# -gline-tables-only: required for accurate source attribution in VTune
target_compile_options(ggml-sycl PRIVATE "-Wno-narrowing" "-fsycl-instrument-device-code" "-fdebug-info-for-profiling" "-gline-tables-only")
target_link_options(ggml-sycl PRIVATE "-fsycl-instrument-device-code")

# Link against oneDNN (primary math backend for Intel targets)
set(GGML_SYCL_DNNL 0)
set(GGML_SYCL_MKL_FALLBACK_ENABLED 0)

# Deprecation warning for old GGML_SYCL_DNN flag
if(DEFINED GGML_SYCL_DNN)
    message(DEPRECATION "GGML_SYCL_DNN is deprecated. oneDNN is now the primary backend for Intel targets.")
endif()

# New option for MKL fallback
option(GGML_SYCL_USE_MKL_FALLBACK "Enable oneMKL as optional fallback math backend" OFF)

if (GGML_SYCL_TARGET STREQUAL "INTEL")
    # oneDNN is REQUIRED for Intel targets (primary math backend)
    find_package(DNNL REQUIRED)
    if(DNNL_FOUND)
        if (NOT DEFINED DNNL_GPU_VENDOR)
            set(DNNL_GPU_VENDOR "INTEL")
        endif()

        if("${GGML_SYCL_TARGET}" STREQUAL "${DNNL_GPU_VENDOR}")
            target_link_libraries(ggml-sycl PRIVATE DNNL::dnnl)
            set(GGML_SYCL_DNNL 1)
            get_target_property(CONFIGS DNNL::dnnl IMPORTED_CONFIGURATIONS)
            foreach(CONFIG ${CONFIGS})
                get_target_property(DNNL_LIB DNNL::dnnl IMPORTED_LOCATION_${CONFIG})
                message(STATUS "Found oneDNN (primary): ${DNNL_LIB}")
            endforeach()
        else()
            message(FATAL_ERROR
                "oneDNN must be compiled for INTEL target.
                 llama.cpp: ${GGML_SYCL_TARGET}, oneDNN: ${DNNL_GPU_VENDOR}")
        endif()
    endif()

    # Optional MKL fallback for Intel targets
    if(GGML_SYCL_USE_MKL_FALLBACK)
        find_package(MKL QUIET)
        if(MKL_FOUND)
            target_link_libraries(ggml-sycl PRIVATE MKL::MKL_SYCL::BLAS)
            set(GGML_SYCL_MKL_FALLBACK_ENABLED 1)
            message(STATUS "Found oneMKL: enabled as optional fallback")
        else()
            message(WARNING "MKL fallback requested but MKL not found - using oneDNN only")
        endif()
    endif()
else()
    # For NVIDIA/AMD: oneDNN is optional (they use oneMath with cuBLAS/rocBLAS)
    find_package(DNNL QUIET)
    if(DNNL_FOUND)
        if (NOT DEFINED DNNL_GPU_VENDOR)
            set(DNNL_GPU_VENDOR "INTEL")
        endif()
        if("${GGML_SYCL_TARGET}" STREQUAL "${DNNL_GPU_VENDOR}")
            target_link_libraries(ggml-sycl PRIVATE DNNL::dnnl)
            set(GGML_SYCL_DNNL 1)
            message(STATUS "Found oneDNN for ${GGML_SYCL_TARGET}")
        endif()
    endif()
endif()

target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_DNNL=${GGML_SYCL_DNNL})
target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_MKL_FALLBACK=${GGML_SYCL_MKL_FALLBACK_ENABLED})
if (GGML_SYCL_XMX_GEMM)
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_XMX_GEMM)
    message(STATUS "XMX GEMM enabled (experimental, known to be 5-11x slower for quantized models)")
endif()

# Link against oneCCL for tensor parallelism collective operations
set(GGML_SYCL_CCL 0)
option(GGML_SYCL_ONECCL "Enable oneCCL for tensor parallelism collectives" ON)
if(GGML_SYCL_ONECCL)
    find_package(oneCCL QUIET)
    if(oneCCL_FOUND)
        target_link_libraries(ggml-sycl PRIVATE ccl)
        set(GGML_SYCL_CCL 1)
        message(STATUS "Found oneCCL: enabled for tensor parallelism")
    else()
        message(STATUS "oneCCL not found, tensor parallelism will use manual ALL_REDUCE")
    endif()
else()
    message(STATUS "oneCCL support disabled by the user")
endif()
target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_CCL=${GGML_SYCL_CCL})

if (GGML_SYCL_F16)
    if (GGML_SYCL_TARGET STREQUAL "AMD")
        message(WARNING "AMD target does not entirely support FP16 in the SYCL backend.")
    endif()
    add_compile_definitions(GGML_SYCL_F16)
endif()

if (GGML_SYCL_TARGET STREQUAL "INTEL")
    # Enable AOT for Battlemage (B580/G21) and JIT fallback (spir64)
    # This prevents runtime JIT compilation overhead
    target_compile_options(ggml-sycl PRIVATE "-fsycl-targets=intel_gpu_bmg_g21,spir64")
    target_link_options(ggml-sycl PRIVATE "-fsycl-targets=intel_gpu_bmg_g21,spir64")

    # Force WARP_SIZE=32 for Intel to match MMQ tile sizes (32) and fix division-by-zero
    # issues in load_tiles_q4_0. Intel hardware supports subgroup size 32.
    add_compile_definitions(GGML_SYCL_WARP_SIZE=32)
    # target_link_options(ggml-sycl PRIVATE  -Xs   -ze-intel-greater-than-4GB-buffer-required)
elseif (GGML_SYCL_TARGET STREQUAL "NVIDIA")
    add_compile_definitions(GGML_SYCL_WARP_SIZE=32)
elseif (GGML_SYCL_TARGET STREQUAL "AMD")
    # INFO: Allowed Sub_group_sizes are not consistent through all
    # hip targets. For example, 64 is used for certain models, but the backend
    # does not support it.
    # Target archs tested working: gfx1030, gfx1031, (Only tested sub_group_size = 32)
    add_compile_definitions(GGML_SYCL_WARP_SIZE=32)
else()
    # default for other target
    add_compile_definitions(GGML_SYCL_WARP_SIZE=32)
endif()

if (GGML_SYCL_GRAPH)
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_GRAPH)
endif()

# Optional VTune ITT API profiling support
option(GGML_SYCL_PROFILE "Enable VTune ITT API profiling instrumentation" OFF)
if (GGML_SYCL_PROFILE)
    # Find ITT library from oneAPI installation
    if (DEFINED ENV{ONEAPI_ROOT})
        set(ITT_ROOT "$ENV{ONEAPI_ROOT}/vtune/latest" CACHE PATH "Path to VTune installation")
    else()
        set(ITT_ROOT "/opt/intel/oneapi/vtune/latest" CACHE PATH "Path to VTune installation")
    endif()

    find_path(ITT_INCLUDE_DIR ittnotify.h
        PATHS ${ITT_ROOT}/sdk/include ${ITT_ROOT}/include
        NO_DEFAULT_PATH)
    find_library(ITT_LIBRARY NAMES ittnotify
        PATHS ${ITT_ROOT}/sdk/lib64 ${ITT_ROOT}/lib64
        NO_DEFAULT_PATH)

    if (ITT_INCLUDE_DIR AND ITT_LIBRARY)
        target_include_directories(ggml-sycl PRIVATE ${ITT_INCLUDE_DIR})
        target_link_libraries(ggml-sycl PRIVATE ${ITT_LIBRARY})
        target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_PROFILE)
        message(STATUS "VTune ITT profiling enabled: ${ITT_LIBRARY}")
    else()
        message(WARNING "VTune ITT library not found. Profiling disabled.
            ITT_INCLUDE_DIR=${ITT_INCLUDE_DIR}
            ITT_LIBRARY=${ITT_LIBRARY}
            Set ITT_ROOT to your VTune installation path.")
    endif()
endif()

# Link against oneMath for NVIDIA/AMD (Intel uses oneDNN as primary)
if (GGML_SYCL_TARGET STREQUAL "INTEL")
    # Intel uses oneDNN as primary math backend (configured above)
    # MKL is only used if GGML_SYCL_USE_MKL_FALLBACK=ON
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(SYCL_COMPILER ON)
    endif()
    if(GGML_SYCL_MKL_FALLBACK_ENABLED)
        target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_USE_INTEL_ONEMKL)
    endif()
    message(STATUS "Intel SYCL backend: oneDNN primary, MKL fallback=${GGML_SYCL_MKL_FALLBACK_ENABLED}")

else()
    find_package(oneMath QUIET)
    if (NOT oneMath_FOUND)
        message(STATUS "oneMath not found: oneMath will be automatically downloaded")
        # Use FetchContent to automatically pull and build oneMath
        include(FetchContent)
        set(BUILD_FUNCTIONAL_TESTS False)
        set(BUILD_EXAMPLES False)
        set(TARGET_DOMAINS blas)
        if (GGML_SYCL_TARGET STREQUAL "NVIDIA")
            set(ENABLE_MKLCPU_BACKEND False)
            set(ENABLE_MKLGPU_BACKEND False)
            set(ENABLE_CUBLAS_BACKEND True)
        elseif (GGML_SYCL_TARGET STREQUAL "AMD")
            set(ENABLE_MKLCPU_BACKEND False)
            set(ENABLE_MKLGPU_BACKEND False)
            set(ENABLE_ROCBLAS_BACKEND True)
            # Ensure setting a string variable here is not overriden by oneMath CACHE variables
            cmake_policy(SET CMP0126 NEW)
            # Setting the device architecture is only needed and useful for AMD devices in oneMath
            set(HIP_TARGETS ${GGML_SYCL_DEVICE_ARCH} CACHE STRING "oneMath HIP target" FORCE)
        endif()
        FetchContent_Declare(
            ONEMATH
            GIT_REPOSITORY https://github.com/uxlfoundation/oneMath.git
            GIT_TAG 8efe85f5aaebb37f1d8c503b7af66315feabf142
        )
        FetchContent_MakeAvailable(ONEMATH)
        # Create alias to match with find_package targets name
        function(onemath_alias target)
            if (TARGET ${target}_obj)
                # Silence verbose warnings from external libraries
                target_compile_options(${target}_obj PRIVATE -w)
            endif()
            if (TARGET ${target})
                add_library(ONEMATH::${target} ALIAS ${target})
            endif()
        endfunction()
        onemath_alias(onemath)
        onemath_alias(onemath_blas_mklcpu)
        onemath_alias(onemath_blas_mklgpu)
        onemath_alias(onemath_blas_cublas)
        onemath_alias(onemath_blas_rocblas)
    endif()

    # Below oneMath compile-time dispatching is used for better performance
    if (GGML_SYCL_TARGET STREQUAL "NVIDIA")
        target_link_libraries(ggml-sycl PRIVATE ONEMATH::onemath_blas_cublas)
        target_compile_options(ggml-sycl PRIVATE "-fsycl-targets=nvptx64-nvidia-cuda")
        target_link_options(ggml-sycl PRIVATE "-fsycl-targets=nvptx64-nvidia-cuda")
        target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_NVIDIA)
    elseif (GGML_SYCL_TARGET STREQUAL "AMD")
        if (NOT GGML_SYCL_DEVICE_ARCH)
            message(FATAL_ERROR "Can't enable SYCL hip backend, GGML_SYCL_DEVICE_ARCH has not been set.")
        endif()
        target_link_libraries(ggml-sycl PRIVATE ONEMATH::onemath_blas_rocblas)
        target_compile_options(ggml-sycl PRIVATE "-fsycl-targets=amdgcn-amd-amdhsa")
        target_link_options(ggml-sycl PRIVATE "-fsycl-targets=amdgcn-amd-amdhsa")
        target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_AMD)
    else()
        # Fallback to oneMath runtime dispatcher
        target_link_libraries(ggml-sycl PRIVATE ONEMATH::onemath)
        target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_GENERIC)
    endif()
endif()

if (GGML_SYCL_DEVICE_ARCH)
    target_compile_options(ggml-sycl PRIVATE -Xsycl-target-backend --offload-arch=${GGML_SYCL_DEVICE_ARCH})
    target_link_options(ggml-sycl PRIVATE -Xsycl-target-backend --offload-arch=${GGML_SYCL_DEVICE_ARCH})
endif()

# =============================================================================
# XMX ESIMD Tests
# =============================================================================
option(GGML_SYCL_BUILD_XMX_TESTS "Build XMX ESIMD unit tests" OFF)

if (GGML_SYCL_BUILD_XMX_TESTS AND GGML_SYCL_TARGET STREQUAL "INTEL")
    message(STATUS "Building XMX ESIMD unit tests")

    # Common SYCL compile/link options for tests
    set(XMX_TEST_SYCL_OPTIONS
        "-fsycl"
        "-fsycl-targets=intel_gpu_bmg_g21,spir64"
        "-Wno-narrowing"
    )

    # Test 1: XMX Hardware Detection Tests
    # This test is standalone and only needs xmx-esimd-common.hpp
    add_executable(test-xmx-hardware-detect
        tests/test-xmx-hardware-detect.cpp
    )
    target_include_directories(test-xmx-hardware-detect PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-xmx-hardware-detect PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_link_options(test-xmx-hardware-detect PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_compile_definitions(test-xmx-hardware-detect PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
        XMX_TEST_STANDALONE=1
    )
    if (IntelSYCL_FOUND)
        target_link_libraries(test-xmx-hardware-detect PRIVATE IntelSYCL::SYCL_CXX)
    endif()

    # Test 2: XMX ESIMD Basic Operations Tests
    # This test is standalone and only needs xmx-esimd-common.hpp
    add_executable(test-xmx-esimd-basic
        tests/test-xmx-esimd-basic.cpp
    )
    target_include_directories(test-xmx-esimd-basic PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-xmx-esimd-basic PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_link_options(test-xmx-esimd-basic PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_compile_definitions(test-xmx-esimd-basic PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
        XMX_TEST_STANDALONE=1
    )
    if (IntelSYCL_FOUND)
        target_link_libraries(test-xmx-esimd-basic PRIVATE IntelSYCL::SYCL_CXX)
    endif()

    # Test 3: XMX ESIMD Q8_0 GEMM Kernel Tests - DISABLED (API mismatch)
    # Tests the ESIMD-based Q8_0 GEMM kernel with dynamic hardware detection
    # add_executable(test-xmx-esimd-q8-gemm
    #     tests/test-xmx-esimd-q8-gemm.cpp
    # )
    # target_include_directories(test-xmx-esimd-q8-gemm PRIVATE
    #     ${CMAKE_CURRENT_SOURCE_DIR}
    #     ${CMAKE_CURRENT_SOURCE_DIR}/..
    #     ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    # )
    # target_compile_options(test-xmx-esimd-q8-gemm PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    # target_link_options(test-xmx-esimd-q8-gemm PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    # target_compile_definitions(test-xmx-esimd-q8-gemm PRIVATE
    #     GGML_SYCL_WARP_SIZE=32
    #     GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    #     XMX_TEST_STANDALONE=1
    # )
    # if (IntelSYCL_FOUND)
    #     target_link_libraries(test-xmx-esimd-q8-gemm PRIVATE IntelSYCL::SYCL_CXX)
    # endif()

    # Test 4: XMX DPAS Basic Infrastructure Tests (DISABLED - uses old XMXDpasConfig API)
    # Superseded by unified-kernel.hpp joint_matrix approach
    # TODO(llama.cpp-6hp): Update or remove when XMX kernel optimization resumes
    # add_executable(test-xmx-dpas-basic
    #     tests/test-xmx-dpas-basic.cpp
    # )
    # target_include_directories(test-xmx-dpas-basic PRIVATE
    #     ${CMAKE_CURRENT_SOURCE_DIR}
    #     ${CMAKE_CURRENT_SOURCE_DIR}/..
    #     ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    # )
    # target_compile_options(test-xmx-dpas-basic PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    # target_link_options(test-xmx-dpas-basic PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    # target_compile_definitions(test-xmx-dpas-basic PRIVATE
    #     GGML_SYCL_WARP_SIZE=32
    #     GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    #     XMX_TEST_STANDALONE=1
    # )
    # if (IntelSYCL_FOUND)
    #     target_link_libraries(test-xmx-dpas-basic PRIVATE IntelSYCL::SYCL_CXX)
    # endif()

    # Test 5: XMX ESIMD Q4_0 GEMM Kernel Tests (DISABLED - uses old XMXDpasConfig API)
    # Superseded by unified-kernel.hpp joint_matrix approach
    # TODO(llama.cpp-6hp): Update or remove when XMX kernel optimization resumes
    # add_executable(test-xmx-esimd-q4-gemm
    #     tests/test-xmx-esimd-q4-gemm.cpp
    # )
    # target_include_directories(test-xmx-esimd-q4-gemm PRIVATE
    #     ${CMAKE_CURRENT_SOURCE_DIR}
    #     ${CMAKE_CURRENT_SOURCE_DIR}/..
    #     ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    # )
    # target_compile_options(test-xmx-esimd-q4-gemm PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    # target_link_options(test-xmx-esimd-q4-gemm PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    # target_compile_definitions(test-xmx-esimd-q4-gemm PRIVATE
    #     GGML_SYCL_WARP_SIZE=32
    #     GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    #     XMX_TEST_STANDALONE=1
    # )
    # if (IntelSYCL_FOUND)
    #     target_link_libraries(test-xmx-esimd-q4-gemm PRIVATE IntelSYCL::SYCL_CXX)
    # endif()

    # Test 6: XMX MoE MXFP4 Fused Kernel Tests
    # Tests the fused MXFP4 MoE GEMM kernel with E8M0 exponents and kvalues LUT
    add_executable(test-xmx-moe-mxfp4
        tests/test-xmx-moe-mxfp4.cpp
    )
    target_include_directories(test-xmx-moe-mxfp4 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-xmx-moe-mxfp4 PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_link_options(test-xmx-moe-mxfp4 PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_compile_definitions(test-xmx-moe-mxfp4 PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
        XMX_TEST_STANDALONE=1
    )
    if (IntelSYCL_FOUND)
        target_link_libraries(test-xmx-moe-mxfp4 PRIVATE IntelSYCL::SYCL_CXX)
    endif()

    # Register tests with CTest
    include(CTest)
    add_test(NAME xmx-hardware-detect COMMAND test-xmx-hardware-detect)
    add_test(NAME xmx-esimd-basic COMMAND test-xmx-esimd-basic)
    # add_test(NAME xmx-esimd-q8-gemm COMMAND test-xmx-esimd-q8-gemm)  # DISABLED
    # add_test(NAME xmx-dpas-basic COMMAND test-xmx-dpas-basic)  # DISABLED - old API
    # add_test(NAME xmx-esimd-q4-gemm COMMAND test-xmx-esimd-q4-gemm)  # DISABLED - old API
    add_test(NAME xmx-moe-mxfp4 COMMAND test-xmx-moe-mxfp4)

    # Set test properties
    set_tests_properties(xmx-hardware-detect xmx-esimd-basic xmx-moe-mxfp4 PROPERTIES
        LABELS "sycl;xmx"
        TIMEOUT 120
    )

    # MoE MXFP4 tests need longer timeout for large matrix tests
    set_tests_properties(xmx-moe-mxfp4 PROPERTIES
        TIMEOUT 300
    )

    # Test 7: MMQ XMX Dispatch Integration Tests
    # Verifies dispatch logic, thresholds, and edge cases (empty batch, non-aligned dims)
    add_executable(test-mmq-xmx-dispatch
        tests/test-mmq-xmx-dispatch.cpp
    )
    target_include_directories(test-mmq-xmx-dispatch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-mmq-xmx-dispatch PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_link_options(test-mmq-xmx-dispatch PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_compile_definitions(test-mmq-xmx-dispatch PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
        XMX_TEST_STANDALONE=1
    )
    if (IntelSYCL_FOUND)
        target_link_libraries(test-mmq-xmx-dispatch PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME mmq-xmx-dispatch COMMAND test-mmq-xmx-dispatch)
    set_tests_properties(mmq-xmx-dispatch PROPERTIES
        LABELS "sycl;xmx;dispatch"
        TIMEOUT 120
    )

    # Test 8: Unified Kernel Tests
    # Tests the unified matmul kernel architecture with Q4_0 support
    add_executable(test-unified-kernel
        tests/test-unified-kernel.cpp
        unified-kernel.cpp
    )
    target_include_directories(test-unified-kernel PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-unified-kernel PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_link_options(test-unified-kernel PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_compile_definitions(test-unified-kernel PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    if (IntelSYCL_FOUND)
        target_link_libraries(test-unified-kernel PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME unified-kernel COMMAND test-unified-kernel)
    set_tests_properties(unified-kernel PROPERTIES
        LABELS "sycl;unified;matmul"
        TIMEOUT 180
    )

    # Test 9: XMX Compute Path Tests
    # Tests the XMX dpas compute path using joint_matrix extensions
    add_executable(test-xmx-compute
        tests/test-xmx-compute.cpp
        unified-kernel.cpp
    )
    target_include_directories(test-xmx-compute PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-xmx-compute PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_link_options(test-xmx-compute PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_compile_definitions(test-xmx-compute PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    if (IntelSYCL_FOUND)
        target_link_libraries(test-xmx-compute PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME xmx-compute COMMAND test-xmx-compute)
    set_tests_properties(xmx-compute PROPERTIES
        LABELS "sycl;xmx;unified;matmul"
        TIMEOUT 300
    )

    # Test 10: XMX Default Enable Tests
    # Tests that XMX is enabled by default and uses XMXCapabilities
    add_executable(test-xmx-default-enable
        tests/test-xmx-default-enable.cpp
        unified-kernel.cpp
    )
    target_include_directories(test-xmx-default-enable PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )
    target_compile_options(test-xmx-default-enable PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_link_options(test-xmx-default-enable PRIVATE ${XMX_TEST_SYCL_OPTIONS})
    target_compile_definitions(test-xmx-default-enable PRIVATE
        GGML_SYCL_WARP_SIZE=32
        GGML_SYCL_DNNL=${GGML_SYCL_DNNL}
    )
    if (IntelSYCL_FOUND)
        target_link_libraries(test-xmx-default-enable PRIVATE IntelSYCL::SYCL_CXX)
    endif()
    add_test(NAME xmx-default-enable COMMAND test-xmx-default-enable)
    set_tests_properties(xmx-default-enable PROPERTIES
        LABELS "sycl;xmx;unified;default"
        TIMEOUT 120
    )
endif()
