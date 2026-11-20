find_package(hrx CONFIG REQUIRED)
find_package(loomc CONFIG REQUIRED)

set(GGML_HRX_KERNEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
set(GGML_HRX_KERNELS_CPP "${CMAKE_CURRENT_BINARY_DIR}/ggml-hrx-kernels.cpp")

file(GLOB_RECURSE GGML_HRX_KERNEL_FILES CONFIGURE_DEPENDS "${GGML_HRX_KERNEL_DIR}/*.loom")

add_custom_command(
    OUTPUT  "${GGML_HRX_KERNELS_CPP}"
    COMMAND "${CMAKE_COMMAND}"
            -DKERNEL_DIR=${GGML_HRX_KERNEL_DIR}
            -DOUTPUT=${GGML_HRX_KERNELS_CPP}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/embed-kernels.cmake"
    DEPENDS ${GGML_HRX_KERNEL_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/embed-kernels.cmake"
    COMMENT "Embedding HRX Loom kernels"
    VERBATIM
)

file(GLOB GGML_HRX_SOURCES CONFIGURE_DEPENDS "*.cpp" "ops/*.cpp")
file(GLOB GGML_HRX_HEADERS CONFIGURE_DEPENDS "*.h")

ggml_add_backend_library(ggml-hrx
    ${GGML_HRX_HEADERS}
    ${GGML_HRX_SOURCES}
    "${GGML_HRX_KERNELS_CPP}"
    ../../include/ggml-hrx.h
)

target_link_libraries(ggml-hrx PRIVATE hrx::hrx loomc::loomc)
target_include_directories(ggml-hrx PRIVATE .)
target_compile_features(ggml-hrx PRIVATE cxx_std_17)
