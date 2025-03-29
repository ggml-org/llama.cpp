project(ggml-hexagon)
message(STATUS "Using HEXAGON backend")
message("CMAKE_SYSTEM_NAME : ${CMAKE_SYSTEM_NAME}")

if(NOT DEFINED QNN_SDK_PATH)
    message(FATAL_ERROR "QNN_SDK_PATH not defined")
endif()

if(NOT DEFINED HEXAGON_SDK_PATH)
    message(FATAL_ERROR "HEXAGON_SDK_PATH not defined")
endif()

message("QNN_SDK_PATH    : ${QNN_SDK_PATH}")
message("HEXAGON_SDK_PATH: ${HEXAGON_SDK_PATH}")
message("HTP_ARCH_VERSION: ${HTP_ARCH_VERSION}")

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    find_library(LOG_LIB log)

    add_library(cdsprpc
        SHARED
        IMPORTED)
    set_target_properties(cdsprpc
        PROPERTIES
        IMPORTED_LOCATION
        ${HEXAGON_SDK_PATH}/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so)

    set(QNN_LINK_LIBRARIES ${LOG_LIB} cdsprpc)
    set(QNN_DEFAULT_LIB_SEARCH_PATH "/data/local/tmp/" CACHE STRING "customized library search path for QNN backend")

    include_directories(${HEXAGON_SDK_PATH}/incs)
    include_directories(${HEXAGON_SDK_PATH}/incs/stddef)
    include_directories(${HEXAGON_SDK_PATH}/ipc/fastrpc/incs)
    include_directories(${HEXAGON_SDK_PATH}/ipc/fastrpc/rpcmem/inc)
    include_directories(${HEXAGON_SDK_PATH}/ipc/fastrpc/remote/ship/android_Debug_aarch64)
    include_directories(${HEXAGON_SDK_PATH}/utils/examples)
    include_directories(${HEXAGON_SDK_PATH}/ipc/fastrpc/rtld/ship/android_aarch64)
    include_directories(${HEXAGON_SDK_PATH}/libs/atomic/inc)
    include_directories(${HEXAGON_SDK_PATH}/libs/atomic/android_Debug_aarch64/ship)
    include_directories(${CMAKE_SOURCE_DIR}/ggml/src/ggml-hexagon/)
    include_directories(${CMAKE_SOURCE_DIR}/ggml/src/ggml-hexagon/kernels/)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(QNN_DEFAULT_LIB_SEARCH_PATH "C:\\" CACHE STRING "customized library search path for QNN backend")
else()
    message(FATAL_ERROR "QNN now only available on Android and Windows(Windows on ARM)")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGGML_USE_QNN")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

file(GLOB QNN_SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp" "${CMAKE_CURRENT_LIST_DIR}/kernels/ggmlop_ap_skel.c")
ggml_add_backend_library(ggml-hexagon ${QNN_SOURCES})

target_include_directories(ggml-hexagon PRIVATE ${QNN_SDK_PATH}/include/QNN ${HEXAGON_SDK_PATH} ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(ggml-hexagon PRIVATE ${QNN_LINK_LIBRARIES})

string(REGEX REPLACE "/$" "" QNN_DEFAULT_LIB_SEARCH_PATH "${QNN_DEFAULT_LIB_SEARCH_PATH}")
target_compile_definitions(ggml-hexagon PRIVATE QNN_DEFAULT_LIB_SEARCH_PATH="${QNN_DEFAULT_LIB_SEARCH_PATH}/")

function(ggml_hexagon_build_kernel KNAME)
    message(STATUS "ggml_hexagon: build kernel ${KNAME}")

    add_custom_command(
        TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND echo "current working path:`pwd`\n"
        COMMAND echo "${CMAKE_CURRENT_LIST_DIR}/kernels"
        COMMAND make -C ${CMAKE_CURRENT_LIST_DIR}/kernels/ clean
        COMMAND make -C ${CMAKE_CURRENT_LIST_DIR}/kernels/ HEXAGON_SDK_PATH=${HEXAGON_SDK_PATH} HTP_ARCH_VERSION=${HTP_ARCH_VERSION}
        COMMAND echo "`pwd`"
        COMMENT "build hexagon-kernel"
    )
endfunction()

ggml_hexagon_build_kernel("cdsp")
