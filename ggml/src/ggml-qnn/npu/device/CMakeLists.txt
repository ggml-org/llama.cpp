
# hexagon npu build, this section will run inside the `build_cmake` script
cmake_minimum_required(VERSION 3.14.3)
project(hexagon_npu C CXX ASM)

enable_language(ASM)
cmake_policy(SET CMP0115 OLD)

if(DEFINED ENV{HEXAGON_SDK_ROOT})
    set(HEXAGON_SDK_ROOT $ENV{HEXAGON_SDK_ROOT})
    message("HEXAGON_SDK_ROOT (from environment): ${HEXAGON_SDK_ROOT}")
elseif(DEFINED HEXAGON_SDK_ROOT)
    message("HEXAGON_SDK_ROOT: ${HEXAGON_SDK_ROOT}")
else()
    message(FATAL_ERROR "HEXAGON_SDK_ROOT not defined")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
    set(PREBUILT_LIB_DIR "android_aarch64")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(PREBUILT_LIB_DIR "UbuntuARM_aarch64")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    # Windows
    set(PREBUILT_LIB_DIR "windows_aarch64")
endif()

if(HEXAGON_SDK_ROOT)
    include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
else()
    include(${HEXAGON_CMAKE_ROOT}/hexagon_fun.cmake)
endif()

# Base Include dirs for the Project
set(common_incs
    ${CMAKE_CURRENT_BINARY_DIR}/
    ${HEXAGON_SDK_ROOT}/incs/
    ${HEXAGON_SDK_ROOT}/incs/stddef/
    ${HEXAGON_SDK_ROOT}/incs/HAP/
    ${HEXAGON_SDK_ROOT}/rtos/qurt/
    ${HEXAGON_SDK_ROOT}/utils/examples/
)

include_directories(${common_incs})

# check if QNN_SDK_ROOT is set
if(NOT DEFINED ENV{QNN_SDK_ROOT})
    message(FATAL_ERROR "QNN_SDK_ROOT not defined")
endif()

set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})
message("QNN_SDK_ROOT: ${QNN_SDK_ROOT}")
message("GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS: ${GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS}")

include_directories(
    ${QNN_SDK_ROOT}/include/QNN/
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

file(GLOB common_srcs "${CMAKE_CURRENT_LIST_DIR}/common/*.cpp")
file(GLOB device_srcs "${CMAKE_CURRENT_LIST_DIR}/device/*.cpp")
file(GLOB device_op_srcs "${CMAKE_CURRENT_LIST_DIR}/device/op/*.cpp")
set(skel_srcs "${CMAKE_CURRENT_BINARY_DIR}/npu_device_skel.c")
add_library(hexagon_npu_skel_OBJS OBJECT
    ${common_srcs}
    ${device_srcs}
    ${device_op_srcs}
    ${skel_srcs}
)

if(CMAKE_BUILD_TYPE MATCHES "Debug|Dbg")
    message("Debug build, enable all logging")
    target_compile_definitions(hexagon_npu_skel_OBJS PUBLIC
        _DEBUG
        DEBUG_LOGGING
    )
else()
    message("Release build, disable debug logging")
    target_compile_definitions(hexagon_npu_skel_OBJS PUBLIC
        NDEBUG
        RELEASE_LOGGING
    )
endif()

if(GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS)
    message("GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS is enabled")
    target_compile_definitions(hexagon_npu_skel_OBJS PUBLIC
        GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS
    )
endif()

if(GGML_HEXAGON_NPU_SANITIZE_ADDRESS)
    message("GGML_HEXAGON_NPU_SANITIZE_ADDRESS is enabled")
    target_compile_options(hexagon_npu_skel_OBJS PUBLIC
        -fsanitize=address -fno-omit-frame-pointer
    )
    target_link_options(hexagon_npu_skel_OBJS PUBLIC
        -fsanitize=address
    )
endif()

if(GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING)
    message("GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING is enabled")
    target_compile_definitions(hexagon_npu_skel_OBJS PUBLIC
        GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING
    )
endif()

build_idl(../idl/hexagon_npu.idl hexagon_npu_skel_OBJS)

add_subdirectory(${HEXAGON_SDK_ROOT}/libs/qprintf qprintf_dir)
target_include_directories(hexagon_npu_skel_OBJS PUBLIC
    ${HEXAGON_SDK_ROOT}/libs/qprintf/inc/

    # TODO: find a better way to include these
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/op/
)

# disable warnings for the skel
set_source_files_properties(
    ${skel_srcs}
    PROPERTIES
    COMPILE_FLAGS "-w"
)

add_library(hexagon_npu_skel SHARED $<TARGET_OBJECTS:hexagon_npu_skel_OBJS>)
target_link_libraries(hexagon_npu_skel
    ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++abi.so.1
    ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++.so.1
    ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc.so
)
set_target_properties(hexagon_npu_skel PROPERTIES OUTPUT_NAME "hexagon_npu_skel_${HEXAGON_ARCH}")
target_link_libraries(hexagon_npu_skel qprintf_static)

copy_binaries(hexagon_npu_skel)
