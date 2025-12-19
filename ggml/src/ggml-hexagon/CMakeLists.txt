include(${HEXAGON_SDK_ROOT}/build/cmake/hexagon_fun.cmake)
include(ExternalProject)

option(GGML_HEXAGON_HTP_DEBUG "ggml-hexagon: enable HTP debug output" OFF)
set(GGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE 128 CACHE STRING "ggml-hexagon: quantize group size (32, 64, or 128)")

add_library(htp_iface OBJECT
    ${CMAKE_CURRENT_BINARY_DIR}/htp_iface_stub.c)

set_target_properties(htp_iface PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(htp_iface PUBLIC
    ${HEXAGON_SDK_ROOT}/incs
    ${HEXAGON_SDK_ROOT}/incs/stddef
    ${HEXAGON_SDK_ROOT}/utils/examples
    ${CMAKE_CURRENT_SOURCE_DIR}/htp
    ${CMAKE_CURRENT_BINARY_DIR})

build_idl(htp/htp_iface.idl htp_iface)

if (CMAKE_SYSTEM_NAME MATCHES Android)
    target_link_options(htp_iface PUBLIC -llog -ldl)
elseif (CMAKE_SYSTEM_NAME MATCHES Windows)
    target_precompile_headers(htp_iface PUBLIC <sal.h>)
else()
    target_link_options(htp_iface PUBLIC -ldl)
endif()

link_custom_library(htp_iface cdsprpc)
link_custom_library(htp_iface rpcmem)

set(TARGET_NAME ggml-hexagon)
ggml_add_backend_library(${TARGET_NAME}
    ggml-hexagon.cpp htp-utils.c htp-utils.h ../../include/ggml-hexagon.h)

target_link_libraries(${TARGET_NAME} PRIVATE htp_iface)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/htp ${CMAKE_CURRENT_BINARY_DIR})

# Build HTP bits
set(HTP_CMAKE_ARGS
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_CURRENT_SOURCE_DIR}/htp/cmake-toolchain.cmake
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_LIBDIR=${CMAKE_CURRENT_BINARY_DIR}
    -DHEXAGON_SDK_ROOT=$ENV{HEXAGON_SDK_ROOT}
    -DHEXAGON_TOOLS_ROOT=$ENV{HEXAGON_TOOLS_ROOT}
    -DHEXAGON_HTP_DEBUG=${GGML_HEXAGON_HTP_DEBUG}
    -DGGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE=${GGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE})

ExternalProject_Add(htp-v68
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/htp BUILD_ALWAYS ON
    CMAKE_ARGS ${HTP_CMAKE_ARGS} -DDSP_VERSION=v68 -DPREBUILT_LIB_DIR="toolv19_v68")

ExternalProject_Add(htp-v69
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/htp BUILD_ALWAYS ON
    CMAKE_ARGS ${HTP_CMAKE_ARGS} -DDSP_VERSION=v69 -DPREBUILT_LIB_DIR="toolv19_v69")

ExternalProject_Add(htp-v73
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/htp BUILD_ALWAYS ON
    CMAKE_ARGS ${HTP_CMAKE_ARGS} -DDSP_VERSION=v73 -DPREBUILT_LIB_DIR="toolv19_v73")

ExternalProject_Add(htp-v75
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/htp BUILD_ALWAYS ON
    CMAKE_ARGS ${HTP_CMAKE_ARGS} -DDSP_VERSION=v75 -DPREBUILT_LIB_DIR="toolv19_v75")

ExternalProject_Add(htp-v79
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/htp BUILD_ALWAYS ON
    CMAKE_ARGS ${HTP_CMAKE_ARGS} -DDSP_VERSION=v79 -DPREBUILT_LIB_DIR="toolv19_v79")

ExternalProject_Add(htp-v81
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/htp BUILD_ALWAYS ON
    CMAKE_ARGS ${HTP_CMAKE_ARGS} -DDSP_VERSION=v81 -DPREBUILT_LIB_DIR="toolv19_v81")

# Install Hexagon skels required at runtime
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/libggml-htp-v68.so
    ${CMAKE_CURRENT_BINARY_DIR}/libggml-htp-v69.so
    ${CMAKE_CURRENT_BINARY_DIR}/libggml-htp-v73.so
    ${CMAKE_CURRENT_BINARY_DIR}/libggml-htp-v75.so
    ${CMAKE_CURRENT_BINARY_DIR}/libggml-htp-v79.so
    ${CMAKE_CURRENT_BINARY_DIR}/libggml-htp-v81.so
    TYPE LIB)
