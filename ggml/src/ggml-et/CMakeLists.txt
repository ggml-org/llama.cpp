
message(STATUS "Using ET backend")

# Configure ET platform path
if (DEFINED ENV{ET_PLATFORM})
    set(ET_PLATFORM_PATH $ENV{ET_PLATFORM})
else()
    set(ET_PLATFORM_PATH "/opt/et")
endif()

# Use sysemu for ET backend if compiled with `-DGGML_ET_SYSEMU=ON`
if (GGML_ET_SYSEMU)
    message(STATUS "Using ET backend with sysemu instead of hardware")
else()
    message(STATUS "Using ET backend with hardware device")
endif()

# Add ET platform CMake modules and config files to search paths
list(APPEND CMAKE_PREFIX_PATH ${ET_PLATFORM_PATH}/lib/cmake)
list(APPEND CMAKE_MODULE_PATH ${ET_PLATFORM_PATH}/lib/cmake)
include(aifoundry-utils/ProjectFunctions)

message(STATUS "Using ET Platform at ${ET_PLATFORM_PATH}")

find_package(runtime REQUIRED)

# Kernel list
set(KERNELS
    el_map_f32
    flash_attn_ext_f32
    glu_f32
    scale_f32
    mul_mat_f32
    mul_mat_f32_matrix_engine
    mul_mat_id_f32
    mul_mat_id_Q4_0
    mul_mat_id_Q8_0
    mul_mat_Q8_0
    mul_mat_Q4_0
    mul_mat_Q4_0_matrix_engine
    mul_mat_f16
    mul_mat_f16_matrix_engine
    rope_f32
    unary_f32
    sqr_f32
    clamp_f32
    sum_rows_f32
    mean_f32
    cumsum_f32
    norm_f32
    l2_norm_f32
    group_norm_f32
    rms_norm_f32
    rms_norm_mul_f32
    softmax_f32
    im2col
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
    set_f32
    fill_f32
    pad_f32
    diag_f32
    tri_f32
    solve_tri_f32
    ssm_conv_f32
    ssm_scan_f32
    conv_2d_f32_me
    memops
    uberkernel
)

# Kernels that we support dispatch form Uberkernel
set(UBERKERNEL_SUPPORTED_KERNELS
    el_map_f32
    # unary_f32
    # cpy_f32_f16
    # cont_f32
    # get_rows_f32
    concat_f32
    cont_f16
    cumsum_f32
    diag_f32
    fill_f32
    flash_attn_ext_f16_me
    flash_attn_ext_f32
    gated_delta_net_f32
    glu_f32
    group_norm_f32
    im2col
    l2_norm_f32
    mul_mat_f16
    mul_mat_f16_matrix_engine
    mul_mat_f32
    mul_mat_f32_matrix_engine
    mul_mat_id_f32
    mul_mat_Q4_0
    mul_mat_Q8_0
    norm_f32
    pad_f32
    repeat_f32
    rms_norm_f32
    rms_norm_mul_f32
    rope_f32
    rwkv_wkv6_f32
    rwkv_wkv7_f32
    scale_f32
    set_f32
    set_rows_f32
    softmax_f32
    solve_tri_f32
    sqr_f32
    # ssm_conv_f32
    ssm_scan_f32
    sum_rows_f32
    tri_f32
)

set(UBERKERNEL_MAP_HPP ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-uberkernel-kernel-map.h)
set(UBERKERNEL_MAP_CPP ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-uberkernel-kernel-map.cpp)

set(UBERKERNEL_KERNELS_SORTED ${UBERKERNEL_SUPPORTED_KERNELS})
list(SORT UBERKERNEL_KERNELS_SORTED)

set(UBERKERNEL_ENUM_ENTRIES "")
set(UBERKERNEL_MAP_ENTRIES "")
set(_uk_idx 1)
foreach(KERNEL ${UBERKERNEL_KERNELS_SORTED})
    string(TOUPPER ${KERNEL} _uk_upper)
    string(APPEND UBERKERNEL_ENUM_ENTRIES
        "    GGML_ET_UBERKERNEL_KERNEL_${_uk_upper} = ${_uk_idx},\n")
    string(APPEND UBERKERNEL_MAP_ENTRIES
        "        {\"${KERNEL}\", GGML_ET_UBERKERNEL_KERNEL_${_uk_upper}},\n")
    math(EXPR _uk_idx "${_uk_idx} + 1")
endforeach()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ggml-et-uberkernel-kernel-map.h.in
    ${UBERKERNEL_MAP_HPP}
    @ONLY)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ggml-et-uberkernel-kernel-map.cpp.in
    ${UBERKERNEL_MAP_CPP}
    @ONLY)

add_custom_target(et-uberkernel-map
    DEPENDS ${UBERKERNEL_MAP_HPP} ${UBERKERNEL_MAP_CPP}
)

# Build ET kernels (cross-compiled in subdirectory scope)
add_subdirectory(et-kernels)

# Embed kernels into C++ source
set(EMBED_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_one_kernel.cmake)
set(EMBED_HPP ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-kernels-embed.hpp)
set(EMBED_CPP ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-kernels-embed.cpp)
set(EMBED_DIR ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/embed)
file(MAKE_DIRECTORY ${EMBED_DIR})

set(EMBED_KERNEL_SOURCES)
set(EMBED_EXTERNS "")
set(EMBED_MAP_ENTRIES "")

foreach(KERNEL ${KERNELS})
    set(ELF_PATH ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/${KERNEL}.elf)
    set(OUT_CPP ${EMBED_DIR}/${KERNEL}.cpp)

    add_custom_command(
        OUTPUT ${OUT_CPP}
        COMMAND ${CMAKE_COMMAND}
            -DELF_FILE=${ELF_PATH}
            -DOUT_FILE=${OUT_CPP}
            -DVAR_NAME=${KERNEL}
            -P ${EMBED_SCRIPT}
        DEPENDS ${KERNEL}.elf ${EMBED_SCRIPT}
        COMMENT "Embedding ${KERNEL}.elf"
        VERBATIM
    )
    list(APPEND EMBED_KERNEL_SOURCES ${OUT_CPP})

    string(APPEND EMBED_EXTERNS
        "extern unsigned char ${KERNEL}_data[];\n"
        "extern const uint64_t ${KERNEL}_len;\n")
    string(APPEND EMBED_MAP_ENTRIES
        "    {\"${KERNEL}\", {${KERNEL}_data, ${KERNEL}_len}},\n")
endforeach()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ggml-et-kernels-embed.hpp.in
    ${EMBED_HPP}
    @ONLY)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ggml-et-kernels-embed.cpp.in
    ${EMBED_CPP}
    @ONLY)

add_custom_target(et-kernels-embed ALL
    DEPENDS ${EMBED_KERNEL_SOURCES} ${EMBED_HPP} ${EMBED_CPP} et-uberkernel-map
)

ggml_add_backend_library(ggml-et
                         ggml-et.cpp
                         ggml-et-kernels.cpp
                         ggml-et-memops.cpp
                         ggml-et-ops.cpp
                         ggml-et-cpu-compare.cpp
                        )

# Mark generated files as such
set_source_files_properties(
    ${EMBED_CPP}
    ${EMBED_HPP}
    ${EMBED_KERNEL_SOURCES}
    ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-uberkernel-kernel-map.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-uberkernel-kernel-map.h
    PROPERTIES GENERATED TRUE
)

# Add embedded kernel sources
target_sources(ggml-et PRIVATE
    ${EMBED_CPP}
    ${EMBED_HPP}
    ${EMBED_KERNEL_SOURCES}
    ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-uberkernel-kernel-map.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-uberkernel-kernel-map.h
)

# Include directory for embedded headers
target_include_directories(ggml-et PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/et-kernels)

target_link_libraries(ggml-et PRIVATE runtime::etrt_static deviceLayer::deviceLayer)
target_compile_definitions(ggml-et PRIVATE GGML_ET_UBERKERNEL_HOST_LOOKUP)
if (GGML_ET_SYSEMU)
    target_compile_definitions(ggml-et PRIVATE GGML_ET_SYSEMU=1)
endif()

# Ensure kernels are built and embedded before the backend library
add_dependencies(ggml-et et-kernels-embed et-uberkernel-map)
