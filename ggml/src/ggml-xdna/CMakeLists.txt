# AMD XDNA (Ryzen AI NPU) backend: dispatches precompiled IRON/mlir-aie kernels through XRT.
#
# Host side links against xrt_coreutil (the XRT runtime that ships with the NPU driver).
# Device side (the *.xclbin / *.insts.bin in kernels/) is built separately with
# mlir-aie, see kernels/README.md.

set(GGML_XDNA_XRT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/xrt-include" CACHE PATH
    "ggml: directory containing the XRT public headers (xrt/xrt_bo.h, ...)")
set(GGML_XDNA_XRT_LIB "" CACHE FILEPATH
    "ggml: xrt_coreutil import/shared library; on Windows it is generated from the driver DLL when empty")
set(GGML_XDNA_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels" CACHE PATH
    "ggml: default directory searched for precompiled XDNA kernels (overridable with $GGML_XDNA_KERNEL_DIR)")

ggml_add_backend_library(ggml-xdna
                         ggml-xdna.cpp
                        )

target_include_directories(ggml-xdna PRIVATE ${GGML_XDNA_XRT_INCLUDE_DIR})

# the CPU half of the CPU+NPU matmul split needs a private CPU backend instance; backends only
# link ggml-base (the registry lives in ggml), so link ggml-cpu directly when it is a library
if (TARGET ggml-cpu AND NOT GGML_BACKEND_DL)
    target_link_libraries(ggml-xdna PRIVATE ggml-cpu)
    target_compile_definitions(ggml-xdna PRIVATE GGML_XDNA_HAVE_CPU)
else()
    message(STATUS "ggml-xdna: no linkable ggml-cpu target (GGML_BACKEND_DL?) - matmuls run NPU-only")
endif()

# likewise a private Vulkan backend instance (the integrated GPU) can take a share of each matmul
if (TARGET ggml-vulkan AND NOT GGML_BACKEND_DL)
    target_link_libraries(ggml-xdna PRIVATE ggml-vulkan)
    target_compile_definitions(ggml-xdna PRIVATE GGML_XDNA_HAVE_VULKAN)
    message(STATUS "ggml-xdna: Vulkan worker enabled")
endif()

file(TO_CMAKE_PATH "${GGML_XDNA_KERNEL_DIR}" _ggml_xdna_kernel_dir)
target_compile_definitions(ggml-xdna PRIVATE GGML_XDNA_DEFAULT_KERNEL_DIR="${_ggml_xdna_kernel_dir}")

if (MSVC)
    # the XRT headers select std::any over boost::any from __cplusplus, which MSVC
    # only reports correctly with this switch (otherwise they demand boost)
    target_compile_options(ggml-xdna PRIVATE /Zc:__cplusplus)
endif()

if (WIN32)
    # NOMINMAX is set by the XRT headers themselves
    target_compile_definitions(ggml-xdna PRIVATE WIN32_LEAN_AND_MEAN)
    if (NOT GGML_XDNA_XRT_LIB)
        file(TO_CMAKE_PATH "$ENV{SystemRoot}/System32/xrt_coreutil.dll" _xrt_dll)
        if (NOT EXISTS "${_xrt_dll}")
            message(FATAL_ERROR "ggml-xdna: ${_xrt_dll} not found - install the AMD NPU (Ryzen AI) driver, "
                                "or pass -DGGML_XDNA_XRT_LIB=<path to xrt_coreutil.lib>")
        endif()
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/xrt_implib.cmake)
        ggml_xdna_make_implib("${_xrt_dll}" "${CMAKE_CURRENT_BINARY_DIR}/xrt_coreutil.lib")
        set(GGML_XDNA_XRT_LIB "${CMAKE_CURRENT_BINARY_DIR}/xrt_coreutil.lib")
    endif()
else()
    if (NOT EXISTS "${GGML_XDNA_XRT_INCLUDE_DIR}/xrt/xrt_bo.h" AND EXISTS "/opt/xilinx/xrt/include/xrt/xrt_bo.h")
        set(GGML_XDNA_XRT_INCLUDE_DIR "/opt/xilinx/xrt/include")
        target_include_directories(ggml-xdna PRIVATE ${GGML_XDNA_XRT_INCLUDE_DIR})
    endif()
    if (NOT GGML_XDNA_XRT_LIB)
        find_library(GGML_XDNA_XRT_LIB NAMES xrt_coreutil HINTS /opt/xilinx/xrt/lib)
        if (NOT GGML_XDNA_XRT_LIB)
            message(FATAL_ERROR "ggml-xdna: libxrt_coreutil not found - install XRT (amd/xdna-driver) "
                                "or pass -DGGML_XDNA_XRT_LIB=<path>")
        endif()
    endif()
endif()

message(STATUS "ggml-xdna: XRT include ${GGML_XDNA_XRT_INCLUDE_DIR}")
message(STATUS "ggml-xdna: XRT library ${GGML_XDNA_XRT_LIB}")
message(STATUS "ggml-xdna: default kernel dir ${_ggml_xdna_kernel_dir}")

target_link_libraries(ggml-xdna PRIVATE ${GGML_XDNA_XRT_LIB})
