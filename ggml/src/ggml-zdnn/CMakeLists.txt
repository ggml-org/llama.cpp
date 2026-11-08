if (DEFINED ZDNN_ROOT)
    message(STATUS "zdnn: using ZDNN_ROOT override: ${ZDNN_ROOT}")
    set(ZDNN_HINT "${ZDNN_ROOT}")
else()
    set(ZDNN_HINT "")
endif()

find_path(ZDNN_INCLUDE
            NAMES zdnn.h
            HINTS ${ZDNN_HINT} /usr /usr/local
            PATH_SUFFIXES include)
if (ZDNN_INCLUDE)
    message(STATUS "zdnn: found include: ${ZDNN_INCLUDE}")
else()
    message(FATAL_ERROR "zdnn: include directory not found, please set ZDNN_ROOT to the proper path if necessary")
endif()

find_library(ZDNN_LIB
                NAMES zdnn
                HINTS ${ZDNN_HINT} /usr /usr/local
                PATH_SUFFIXES lib lib64)
if (ZDNN_LIB)
    message(STATUS "zdnn: found library: ${ZDNN_LIB}")
else()
    message(FATAL_ERROR "zdnn: library not found, please set ZDNN_ROOT to the proper path if necessary")
endif()

file(GLOB GGML_SOURCES_ZDNN "*.c" "*.cpp")
file(GLOB GGML_HEADERS_ZDNN "*.h" "*.hpp")

ggml_add_backend_library(ggml-zdnn ${GGML_HEADERS_ZDNN} ${GGML_SOURCES_ZDNN})
target_link_libraries(ggml-zdnn PRIVATE ${ZDNN_LIB})
target_include_directories(ggml-zdnn PRIVATE ${ZDNN_INCLUDE})
target_link_directories(ggml-zdnn PRIVATE ${ZDNN_LIB})

target_compile_definitions(ggml-zdnn PRIVATE GGML_USE_ZDNN)
