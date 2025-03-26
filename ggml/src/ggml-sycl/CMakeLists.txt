message(STATUS  "GGML_SYCL_TARGET=${GGML_SYCL_TARGET}")

if (NOT GGML_SYCL_TARGET MATCHES "^(INTEL|NVIDIA|AMD)$")
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

find_package(DNNL)
set(GGML_SYCL_DNNL 0)
if(DNNL_FOUND)
    if (DEFINED ENV{ONEAPI_ROOT} AND NOT DEFINED DNNL_GPU_VENDOR)
        # Assuming oneDNN packaged with oneapi release is used which
        # supports only intel target
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
target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_DNNL=${GGML_SYCL_DNNL})

if (GGML_SYCL_F16)
    if (GGML_SYCL_TARGET STREQUAL "AMD")
        message(WARNING "AMD target does not entirely support FP16 in the SYCL backend.")
    endif()
    add_compile_definitions(GGML_SYCL_F16)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing -fsycl")

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


if (WIN32)
    find_package(IntelSYCL REQUIRED)
    find_package(MKL REQUIRED)
    target_link_libraries(ggml-sycl PRIVATE IntelSYCL::SYCL_CXX MKL::MKL MKL::MKL_SYCL)
else()
    if (GGML_SYCL_GRAPH)
        add_compile_definitions(GGML_SYCL_GRAPH)
    endif()
    if (GGML_SYCL_TARGET STREQUAL "INTEL")
        target_link_libraries(ggml-sycl PRIVATE sycl OpenCL mkl_core pthread m dl mkl_sycl_blas mkl_intel_ilp64 mkl_tbb_thread)
    elseif (GGML_SYCL_TARGET STREQUAL "NVIDIA")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda")
        add_compile_definitions(GGML_SYCL_NVIDIA)
        target_link_libraries(ggml-sycl PRIVATE sycl pthread m dl onemkl_blas_cublas)
    elseif (GGML_SYCL_TARGET STREQUAL "AMD")
        if (NOT GGML_SYCL_DEVICE_ARCH)
            message(ERROR "Can't enable SYCL hip backend, GGML_SYCL_DEVICE_ARCH has not been set.")
        endif()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=amdgcn-amd-amdhsa")
        target_link_libraries(ggml-sycl PRIVATE sycl pthread m dl onemkl)
    endif()

    if (GGML_SYCL_DEVICE_ARCH)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xsycl-target-backend --offload-arch=${GGML_SYCL_DEVICE_ARCH}")
  endif()
endif()
