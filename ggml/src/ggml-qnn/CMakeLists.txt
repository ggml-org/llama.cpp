message(STATUS "Using QNN backend")

if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    find_library(LOG_LIB log)
    set(QNN_LINK_LIBRARIES ${LOG_LIB})
    set(QNN_DEFAULT_LIB_SEARCH_PATH "/data/local/tmp/" CACHE STRING "customized library search path for QNN backend")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(QNN_DEFAULT_LIB_SEARCH_PATH "C:\\" CACHE STRING "customized library search path for QNN backend")
elseif(CMAKE_SYSTEM_NAME STREQUAL "CYGWIN")
    set(QNN_DEFAULT_LIB_SEARCH_PATH "/cygdrive/c/qairt/2.31.0.250130/" CACHE STRING "customized library search path for QNN backend")
else()
    message(FATAL_ERROR "QNN now only available on Android and Windows(Windows on ARM)")
endif()

if(NOT DEFINED GGML_QNN_SDK_PATH)
# try read from environment variable
    if(DEFINED ENV{QNN_SDK_PATH})
        set(GGML_QNN_SDK_PATH $ENV{QNN_SDK_PATH})
    else()
        message(FATAL_ERROR "GGML_QNN_SDK_PATH not defined")
    endif()
endif()

message("QNN_SDK_PATH: ${GGML_QNN_SDK_PATH}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGGML_USE_QNN")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

file(GLOB QNN_SOURCES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
    ggml_add_backend_library(ggml-qnn
    ${QNN_SOURCES}
)

target_include_directories(ggml-qnn PRIVATE ${GGML_QNN_SDK_PATH}/include/QNN ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(ggml-qnn PRIVATE ${QNN_LINK_LIBRARIES})

string(REGEX REPLACE "/$" "" GGML_QNN_DEFAULT_LIB_SEARCH_PATH "${QNN_DEFAULT_LIB_SEARCH_PATH}")
target_compile_definitions(ggml-qnn PRIVATE GGML_QNN_DEFAULT_LIB_SEARCH_PATH="${QNN_DEFAULT_LIB_SEARCH_PATH}/")
