

file(GLOB qnn_srcs "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(qnn-backend STATIC
    ${qnn_srcs}
)

target_include_directories(qnn-backend PRIVATE
    ${GGML_QNN_SDK_PATH}/include/QNN/
    ${CMAKE_CURRENT_LIST_DIR}/
    ${CMAKE_CURRENT_LIST_DIR}/../
    ${CMAKE_CURRENT_LIST_DIR}/../../
    ${CMAKE_CURRENT_LIST_DIR}/../../../include/ # TODO: figure out how to remove this
    ${CMAKE_CURRENT_LIST_DIR}/../shared/
)

target_link_directories(qnn-backend PRIVATE
    runtime-common
)

if(GGML_QNN_ENABLE_CPU_BACKEND)
    message("GGML_QNN_ENABLE_CPU_BACKEND is enabled")
    target_compile_definitions(qnn-backend PUBLIC GGML_QNN_ENABLE_CPU_BACKEND)
else()
    message("GGML_QNN_ENABLE_CPU_BACKEND is disabled")
endif()

if(GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING)
    message("GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING is enabled")
    target_compile_definitions(qnn-backend PUBLIC GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING)
else()
    message("GGML_HEXAGON_ENABLE_PERFORMANCE_TRACKING is disabled")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(QNN_DEFAULT_LIB_SEARCH_PATH "/data/local/tmp/" CACHE STRING "customized library search path for QNN backend")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(QNN_DEFAULT_LIB_SEARCH_PATH "" CACHE STRING "customized library search path for QNN backend")
else()
    message(FATAL_ERROR "QNN now only available on Android, Windows and Linux")
endif()

if(NOT "${QNN_DEFAULT_LIB_SEARCH_PATH}" STREQUAL "")
    string(REGEX REPLACE "/$" "" QNN_DEFAULT_LIB_SEARCH_PATH "${QNN_DEFAULT_LIB_SEARCH_PATH}")
endif()

message("GGML_QNN_DEFAULT_LIB_SEARCH_PATH: ${QNN_DEFAULT_LIB_SEARCH_PATH}")
target_compile_definitions(qnn-backend PUBLIC GGML_QNN_DEFAULT_LIB_SEARCH_PATH="${QNN_DEFAULT_LIB_SEARCH_PATH}")
