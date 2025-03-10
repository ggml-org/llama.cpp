message(STATUS  "GGML_SYCL_TARGET=${GGML_SYCL_TARGET}")

if (NOT GGML_SYCL_TARGET MATCHES "^(INTEL.*|NVIDIA|AMD)$")
    message(FATAL_ERROR "Invalid backend chosen, supported options are INTEL, NVIDIA, or AMD")
endif()

check_cxx_compiler_flag("-fsycl" SUPPORTS_SYCL)

if (DEFINED ENV{ONEAPI_ROOT})
    message(STATUS "Using oneAPI Release SYCL compiler (icpx).")
elseif(SUPPORTS_SYCL)
    message(WARNING "Using open-source SYCL compiler (clang++). Didn't detect ENV {ONEAPI_ROOT}.
        If you expected the oneAPI Release compiler, please install oneAPI & source it, like:
        source /opt/intel/oneapi/setvars.sh")
else()
    message(FATAL_ERROR, "C++ compiler lacks SYCL support.")
endif()
message(STATUS "SYCL found")
#todo: AOT

ggml_add_backend_library(ggml-sycl
                         ggml-sycl.cpp
                         ../../include/ggml-sycl.h
                        )

if (GGML_SYCL_F16)
    if (GGML_SYCL_TARGET STREQUAL "AMD")
        message(WARNING "AMD target does not entirely support FP16 in the SYCL backend.")
    endif()
    add_compile_definitions(GGML_SYCL_F16)
endif()

if (GGML_SYCL_TARGET STREQUAL "NVIDIA")
    add_compile_definitions(GGML_SYCL_WARP_SIZE=32)
elseif (GGML_SYCL_TARGET STREQUAL "AMD")
    # INFO: Allowed Sub_group_sizes are not consistent through all
    # hip targets. For example, 64 is used for certain models, but the backend
    # does not support it.
    # Target archs tested working: gfx1030, gfx1031, (Only tested sub_group_size = 32)
    add_compile_definitions(GGML_SYCL_WARP_SIZE=32)
else()
    add_compile_definitions(GGML_SYCL_WARP_SIZE=16)
endif()

file(GLOB   GGML_HEADERS_SYCL "*.hpp")
file(GLOB   GGML_SOURCES_SYCL "*.cpp")
target_sources(ggml-sycl PRIVATE ${GGML_HEADERS_SYCL} ${GGML_SOURCES_SYCL})

find_package(DNNL)
message("-- DNNL found:" ${DNNL_FOUND})

if (GGML_SYCL_TARGET MATCHES "INTEL.*")
    add_compile_definitions(GGML_SYCL_DNNL=${DNNL_FOUND})
else()
    add_compile_definitions(GGML_SYCL_DNNL=0)
endif()

if (${DNNL_FOUND} AND GGML_SYCL_TARGET MATCHES "INTEL.*")
    target_link_libraries(ggml-sycl PRIVATE DNNL::dnnl)
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

target_compile_options(ggml-sycl PRIVATE "-Wno-narrowing")

find_package(oneMath QUIET)
if (NOT oneMath_FOUND)
    message("-- oneMath not found: oneMath will be automatically downloaded")
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
        GIT_TAG develop
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
if (GGML_SYCL_TARGET STREQUAL "INTEL_CPU")
    target_link_libraries(ggml-sycl PRIVATE ONEMATH::onemath_blas_mklcpu)
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_INTEL GGML_SYCL_INTEL_CPU)
elseif (GGML_SYCL_TARGET STREQUAL "INTEL_GPU")
    target_link_libraries(ggml-sycl PRIVATE ONEMATH::onemath_blas_mklgpu)
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_INTEL GGML_SYCL_INTEL_GPU)
elseif (GGML_SYCL_TARGET STREQUAL "NVIDIA")
    target_link_libraries(ggml-sycl PRIVATE ONEMATH::onemath_blas_cublas)
    target_compile_options(ggml-sycl PRIVATE "-fsycl-targets=nvptx64-nvidia-cuda")
    target_link_options(ggml-sycl PRIVATE "-fsycl-targets=nvptx64-nvidia-cuda")
    target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_NVIDIA)
elseif (GGML_SYCL_TARGET STREQUAL "AMD")
    if (NOT GGML_SYCL_DEVICE_ARCH)
        message(ERROR "Can't enable SYCL hip backend, GGML_SYCL_DEVICE_ARCH has not been set.")
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

if (GGML_SYCL_DEVICE_ARCH)
    target_compile_options(ggml-sycl PRIVATE -Xsycl-target-backend --offload-arch=${GGML_SYCL_DEVICE_ARCH})
    target_link_options(ggml-sycl PRIVATE -Xsycl-target-backend --offload-arch=${GGML_SYCL_DEVICE_ARCH})
endif()
