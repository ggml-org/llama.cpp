cmake_minimum_required(VERSION 3.22.2)
project(ggml-htp C CXX ASM)

include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)

# Override qaic executable path based on host OS (SDK defaults to bin/qaic which doesn't exist on Windows)
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    set(QAIC_EXECUTABLE "${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/WinNT/qaic.exe")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
    set(QAIC_EXECUTABLE "${HEXAGON_SDK_ROOT}/ipc/fastrpc/qaic/Ubuntu/qaic")
else()
    set(QAIC_EXECUTABLE "${proj-qaic_src}/bin/qaic")
endif()

function(build_idl idlFile currentTarget)
    string(RANDOM _random)

    set(QIDLINC "")
    set(defaultIncs
        ${HEXAGON_SDK_ROOT}/incs/
        ${HEXAGON_SDK_ROOT}/incs/stddef/
        )
    foreach(path ${defaultIncs})
        set(QIDLINC ${QIDLINC} "-I${path}")
    endforeach(path)

    get_target_property(dirs ${currentTarget} INCLUDE_DIRECTORIES)
    foreach(path ${dirs})
        set(QIDLINC ${QIDLINC} "-I${path}")
    endforeach(path)

    get_filename_component(fileName ${idlFile} NAME_WE)
    set(cmdLineOptions -mdll -o ${CMAKE_CURRENT_BINARY_DIR} ${QIDLINC} ${idlFile})
    set(qidlTarget qidlTarget${fileName}${_random})

    if(NOT TARGET ${qidlTarget})
        idl_generated_files(${idlFile} byproduct_files)

        add_custom_command(
            OUTPUT ${byproduct_files} ${CMAKE_CURRENT_BINARY_DIR}/${fileName}.h
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
            COMMAND "${QAIC_EXECUTABLE}" ${cmdLineOptions}
            DEPENDS ${idlFile}
        )

        add_custom_target( ${qidlTarget}
            DEPENDS ${byproduct_files} ${CMAKE_CURRENT_BINARY_DIR}/${fileName}.h
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
        )

        set_source_files_properties(
            ${byproduct_files}
            ${CMAKE_CURRENT_BINARY_DIR}/${fileName}.h
            PROPERTIES
            GENERATED TRUE
        )
    endif()

    add_dependencies(${currentTarget} ${qidlTarget})
    foreach(target ${ARGN})
        if (TARGET target)
            add_dependencies(${target} ${qidlTarget})
        else()
            MESSAGE(STATUS "build_idl: ${target} is not a target type.")
        endif()
    endforeach()
endfunction()

include_directories(
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR})

set(HTP_LIB ggml-htp-${DSP_VERSION})

add_library(${HTP_LIB} SHARED
    main.c
    htp_iface_skel.c
    worker-pool.c
    hex-dma.c
    matmul-ops.c
    binary-ops.c
    unary-ops.c
    sum-rows-ops.c
    softmax-ops.c
    act-ops.c
    rope-ops.c
    flash-attn-ops.c
    set-rows-ops.c
    get-rows-ops.c
    cpy-ops.c
    repeat-ops.c
    argsort-ops.c
    ssm-conv.c
)

target_compile_definitions(${HTP_LIB} PRIVATE
    $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,HTP_DEBUG=1,NDEBUG=1>
    $<IF:$<BOOL:${HEXAGON_HTP_DEBUG}>,FARF_HIGH=1,>
    FP32_QUANTIZE_GROUP_SIZE=${GGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE})

# VTCM release strategy (ON = hold for entire session, OFF = per-op acquire/release)
option(HTP_VTCM_SESSION_HOLD
    "Hold VTCM for entire session instead of per-op acquire/release" ON)

if (HTP_VTCM_SESSION_HOLD)
    target_compile_definitions(${HTP_LIB} PRIVATE HTP_VTCM_SESSION_HOLD=1)
endif()

# HMX acceleration: available on v73+ architectures
set(HTP_HMX_VERSIONS v73 v75 v79 v81)
list(FIND HTP_HMX_VERSIONS ${DSP_VERSION} _hmx_idx)

if (_hmx_idx GREATER_EQUAL 0)
    target_sources(${HTP_LIB} PRIVATE
        hmx/hmx-matmul-ops.c
        hmx/hmx-flash-attn-ops.c
        hmx/hmx-rms-norm-ops.c
        hmx/precompute-table.c
    )

    # -mhmx enables HMX instruction set (needed by files that include hmx-utils.h)
    set_source_files_properties(
        hmx/hmx-matmul-ops.c
        hmx/hmx-flash-attn-ops.c
        hmx/hmx-rms-norm-ops.c
        PROPERTIES COMPILE_OPTIONS "-mhmx"
    )

    target_compile_definitions(${HTP_LIB} PRIVATE HTP_HAS_HMX=1)
endif()

build_idl(htp_iface.idl ${HTP_LIB})

set_target_properties(${HTP_LIB} PROPERTIES EXPORT_COMPILE_COMMANDS ON)

install(TARGETS ${HTP_LIB})
