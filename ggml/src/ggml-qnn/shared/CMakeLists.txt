
file(GLOB common_srcs "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(runtime-common STATIC
    ${common_srcs}
)

target_include_directories(runtime-common PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/
    ${CMAKE_CURRENT_LIST_DIR}/../
    ${CMAKE_CURRENT_LIST_DIR}/../../
    ${CMAKE_CURRENT_LIST_DIR}/../../../include/ # TODO: figure out how to remove this
)

if(GGML_HEXAGON_NPU_ONLY)
    add_compile_definitions(GGML_HEXAGON_NPU_ONLY)
endif()

if(GGML_QNN_ENABLE_HEXAGON_BACKEND)
    if(DEFINED ENV{QNN_SDK_PATH})
        set(HEXAGON_SDK_ROOT $ENV{HEXAGON_SDK_ROOT})
        message("found HEXAGON_SDK_ROOT, setting to ${HEXAGON_SDK_ROOT}")
    elseif(EXISTS ${HEXAGON_SDK_ROOT})
        message("HEXAGON_SDK_ROOT: ${HEXAGON_SDK_ROOT}")
    else()
        message(FATAL_ERROR "HEXAGON_SDK_ROOT not defined")
    endif()

    target_include_directories(runtime-common PUBLIC
        ${HEXAGON_SDK_ROOT}/incs/
        ${HEXAGON_SDK_ROOT}/incs/stddef/
        ${HEXAGON_SDK_ROOT}/incs/HAP/
        ${HEXAGON_SDK_ROOT}/rtos/qurt/
        ${HEXAGON_SDK_ROOT}/utils/examples/
    )
    target_compile_definitions(runtime-common PRIVATE
        GGML_QNN_ENABLE_HEXAGON_BACKEND
    )
else()
    message("HEXAGON_SDK_ROOT not defined, not appending to include directories")
endif()
