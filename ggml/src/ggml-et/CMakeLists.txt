
message(STATUS "Using ET backend")

# Configure ET platform path
if(DEFINED ENV{ET_PLATFORM})
    set(ET_PLATFORM_PATH $ENV{ET_PLATFORM})
else()
    set(ET_PLATFORM_PATH "/opt/et")
endif()

# Use sysemu for ET backend if compiled with `-DGGML_ET_SYSEMU=ON`
if(GGML_ET_SYSEMU)
    message(STATUS "Using ET backend with sysemu instead of hardware")
    add_compile_definitions(GGML_ET_SYSEMU=1)
else()
    message(STATUS "Using ET backend with hardware device")
endif()

# Add ET platform CMake modules and config files to search paths
list(APPEND CMAKE_PREFIX_PATH ${ET_PLATFORM_PATH}/lib/cmake)
list(APPEND CMAKE_MODULE_PATH ${ET_PLATFORM_PATH}/lib/cmake)
include(aifoundry-utils/ProjectFunctions)

message("Using ET Platform at ${ET_PLATFORM_PATH}")

find_package(runtime REQUIRED)

# --- Kernel list (single source of truth, shared with et-kernels) ---
set(KERNELS
    el_map_f32
    flash_attn_ext_f32
    glu_f32
    scale_f32
    mul_mat_f32
    mul_mat_f32_matrix_engine
    mul_mat_id_f32
    mul_mat_Q8_0
    mul_mat_f16
    mul_mat_f16_matrix_engine
    rope_f32
    unary_f32
    sqr_f32
    sum_rows_f32
    norm_f32
    l2_norm_f32
    rms_norm_f32
    rms_norm_mul_f32
    softmax_f32
    get_rows_f32
    concat_f32
    repeat_f32
    rwkv_wkv6_f32
    rwkv_wkv7_f32
    gated_delta_net_f32
    cont_f32
    cont_f16
    cpy_f32_f16
    flash_attn_ext_f16_me
    set_rows_f32
    fill_f32
    diag_f32
    memops
)

# Build ET kernels (cross-compiled in subdirectory scope)
add_subdirectory(et-kernels)

# --- Embed kernels into C++ source for the host library ---
find_package(Python3 REQUIRED)

set(EMBED_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/et-kernels/scripts/embed_kernels.py)
set(EMBED_HPP ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-kernels-embed.hpp)
set(EMBED_CPP ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-kernels-embed.cpp)

set(KERNEL_ELF_TARGETS)
foreach(KERNEL ${KERNELS})
    list(APPEND KERNEL_ELF_TARGETS ${KERNEL}.elf)
endforeach()

add_custom_command(
    OUTPUT ${EMBED_HPP} ${EMBED_CPP}
    COMMAND ${Python3_EXECUTABLE} ${EMBED_SCRIPT}
        --input-dir ${CMAKE_CURRENT_BINARY_DIR}/et-kernels
        --output-hpp ${EMBED_HPP}
        --output-cpp ${EMBED_CPP}
        --kernels ${KERNELS}
    DEPENDS ${KERNEL_ELF_TARGETS} ${EMBED_SCRIPT}
    COMMENT "Embedding ET kernels into source code"
    VERBATIM
)

add_custom_target(et-kernels-embed ALL
    DEPENDS ${EMBED_HPP} ${EMBED_CPP}
)

# --- Host backend library ---
ggml_add_backend_library(ggml-et
                         ggml-et.cpp
                         ggml-et-kernels.cpp
                         ggml-et-memops.cpp
                         ggml-et-ops.cpp
                         ggml-et-cpu-compare.cpp
                        )

# Mark generated files as such
set_source_files_properties(
    ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-kernels-embed.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-kernels-embed.hpp
    PROPERTIES GENERATED TRUE
)

# Add embedded kernel sources
target_sources(ggml-et PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-kernels-embed.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-kernels-embed.hpp
)

# Include directory for embedded headers
target_include_directories(ggml-et PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/et-kernels)

target_link_libraries(ggml-et PRIVATE runtime::etrt_static deviceLayer::deviceLayer)

# Ensure kernels are built and embedded before the backend library
add_dependencies(ggml-et et-kernels-embed)
