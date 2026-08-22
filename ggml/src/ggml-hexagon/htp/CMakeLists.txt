cmake_minimum_required(VERSION 3.22.2)
project(ggml-htp C CXX ASM)

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

include_directories(
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})

# Op sources shared by both skel variants (dspqueue entry via main.c and
# mempool/FastRPC-invoke entry via entry.c). Entry points and glue differ per variant.
set(HTP_OPS_SRCS
    work-queue.c
    dma-queue.c
    hmx-queue.c
    matmul-ops.c
    flash-attn-ops.c
    gated-delta-net-ops.c
    binary-ops.c
    unary-ops.c
    sum-rows-ops.c
    softmax-ops.c
    act-ops.c
    rope-ops.c
    set-rows-ops.c
    get-rows-ops.c
    cpy-ops.c
    repeat-ops.c
    ssm-conv.c
    cumsum-ops.c
    fill-ops.c
    concat-ops.c
    diag-ops.c
    solve-tri-ops.c
    pad-ops.c
    argsort-ops.c
    im2col-ops.c
)

# Both variants build ggml-htp-${DSP_VERSION} (libggml-htp-vXX.so); only one
# is built per configuration.
option(GGML_HEXAGON_USE_MEMPOOL "Build the mempool/FastRPC-invoke skel variant (entry.c) instead of the dspqueue variant (main.c)" OFF)

if (GGML_HEXAGON_USE_MEMPOOL)
    set(HTP_LIB ggml-htp-${DSP_VERSION})

    add_library(${HTP_LIB} SHARED
        entry.c
        ${CMAKE_CURRENT_BINARY_DIR}/ggml_dsp_skel.c
        ${HTP_OPS_SRCS}
    )

    target_compile_definitions(${HTP_LIB} PRIVATE
        GGML_HEXAGON_USE_MEMPOOL=1
        $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,HTP_DEBUG=1,NDEBUG=1>
        $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,FARF_HIGH=1,>
    )

    target_compile_options(${HTP_LIB} PRIVATE -Wno-error)

    build_idl(ggml_dsp.idl ${HTP_LIB})

    set_target_properties(${HTP_LIB} PROPERTIES EXPORT_COMPILE_COMMANDS ON)

    install(TARGETS ${HTP_LIB})
else()
    set(HTP_LIB ggml-htp-${DSP_VERSION})

    add_library(${HTP_LIB} SHARED
        main.c
        htp_iface_skel.c
        htp-tensor.c
        ${HTP_OPS_SRCS}
    )

    target_compile_definitions(${HTP_LIB} PRIVATE
        $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,HTP_DEBUG=1,NDEBUG=1>
        $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,FARF_HIGH=1,>
    )

    build_idl(htp_iface.idl ${HTP_LIB})

    set_target_properties(${HTP_LIB} PROPERTIES EXPORT_COMPILE_COMMANDS ON)

    install(TARGETS ${HTP_LIB})
endif()
