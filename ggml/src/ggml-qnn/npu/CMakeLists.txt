enable_language(ASM)
cmake_policy(SET CMP0115 OLD)

if(DEFINED ENV{HEXAGON_SDK_ROOT})
    set(HEXAGON_SDK_ROOT $ENV{HEXAGON_SDK_ROOT})
    message("HEXAGON_SDK_ROOT: ${HEXAGON_SDK_ROOT}")
else()
    message(FATAL_ERROR "HEXAGON_SDK_ROOT not defined")
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

if(${CMAKE_SYSTEM_NAME} MATCHES "Android|Linux|Windows")
    # host build
    file(GLOB common_srcs "${CMAKE_CURRENT_LIST_DIR}/common/*.cpp")
    file(GLOB host_srcs "${CMAKE_CURRENT_LIST_DIR}/host/*.cpp")
    set(stub_srcs "${CMAKE_CURRENT_BINARY_DIR}/npu_device_stub.c")
    add_library(hexagon-npu-host STATIC
        ${common_srcs}
        ${host_srcs}
        ${stub_srcs}
    )

    # disable warnings for the stub
    set_source_files_properties(
        ${stub_srcs}
        PROPERTIES
        COMPILE_FLAGS "-w"
    )

    build_idl(idl/hexagon_npu.idl hexagon-npu-host)

    # Add compile definitions to the target
    target_compile_definitions(hexagon-npu-host PUBLIC
        VERIFY_PRINT_ERROR
        GGML_QNN_ENABLE_HEXAGON_BACKEND
    )

    target_include_directories(hexagon-npu-host PRIVATE
        ${HEXAGON_SDK_ROOT}/ipc/fastrpc/rpcmem/inc/
        ${QNN_SDK_ROOT}/include/QNN/
        ${CMAKE_CURRENT_LIST_DIR}/host/
        ${CMAKE_CURRENT_LIST_DIR}/
    )

    target_include_directories(hexagon-npu-host PUBLIC
        ${HEXAGON_SDK_ROOT}/incs/ # TODO: this is for rpc-mem
    )

    if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        set_target_properties(hexagon-npu-host PROPERTIES OUTPUT_NAME "hexagon_npu")
    endif()

    if(${CMAKE_SYSTEM_NAME} MATCHES "Android|Linux")
        target_link_options(hexagon-npu-host PUBLIC -pie)
    endif()

    link_options(hexagon-npu-host)

    if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
        set(PREBUILT_LIB_DIR "android_aarch64")
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        set(PREBUILT_LIB_DIR "UbuntuARM_aarch64")
    else()
        # Windows
        set(PREBUILT_LIB_DIR "windows_aarch64")
    endif()

    choose_dsprpc("3" dsprpc) # cdsprpc
    link_custom_library(hexagon-npu-host ${dsprpc})
else()
    # hexagon npu build
    cmake_minimum_required(VERSION 3.14.3)
    project(hexagon_npu C CXX ASM)

    # check if QNN_SDK_ROOT is set
    if(NOT DEFINED ENV{QNN_SDK_ROOT})
        message(FATAL_ERROR "QNN_SDK_ROOT not defined")
    endif()

    set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})
    message("QNN_SDK_ROOT: ${QNN_SDK_ROOT}")
    include_directories(
        ${QNN_SDK_ROOT}/include/QNN/
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

    file(GLOB common_srcs "${CMAKE_CURRENT_LIST_DIR}/common/*.cpp")
    file(GLOB device_srcs "${CMAKE_CURRENT_LIST_DIR}/device/*.cpp")
    set(skel_srcs "${CMAKE_CURRENT_BINARY_DIR}/npu_device_skel.c")
    add_library(hexagon_npu_skel_OBJS OBJECT
        ${common_srcs}
        ${device_srcs}
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

    build_idl(idl/hexagon_npu.idl hexagon_npu_skel_OBJS)

    # disable warnings for the skel
    set_source_files_properties(
        ${skel_srcs}
        PROPERTIES
        COMPILE_FLAGS "-w"
    )

    add_library(hexagon_npu_skel SHARED $<TARGET_OBJECTS:hexagon_npu_skel_OBJS>)

    target_link_libraries(hexagon_npu_skel
        ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++abi.a
        ${HEXAGON_LIB_DIR}/${HEXAGON_ARCH}/G0/pic/libc++.a
    )
    set_target_properties(hexagon_npu_skel PROPERTIES OUTPUT_NAME "hexagon_npu_skel_${HEXAGON_ARCH}")

    copy_binaries(hexagon_npu_skel)
endif()

# vim: set noet fenc=utf-8 ff=unix ft=cmake :
