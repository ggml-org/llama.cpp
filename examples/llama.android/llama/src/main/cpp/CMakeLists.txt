cmake_minimum_required(VERSION 3.31.6)

project("kleidi-llama" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)

# --------------------------------------------------------------------------
# 1. CPU feature detection library
# --------------------------------------------------------------------------
add_subdirectory(
        ${CMAKE_CURRENT_LIST_DIR}/../../../../../../include/cpu_features
        ${CMAKE_BINARY_DIR}/cpu_features_build)
add_library(kleidi-llama-cpu-detector SHARED cpu_detector.cpp)
target_link_libraries(kleidi-llama-cpu-detector
        PRIVATE CpuFeatures::cpu_features
        android
        log)

# --------------------------------------------------------------------------
# 2. Kleidi Llama library
# --------------------------------------------------------------------------

set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../)
add_subdirectory(${LLAMA_SRC} build-llama)

add_library(${CMAKE_PROJECT_NAME} SHARED
        kleidi-llama.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
        ${LLAMA_SRC}
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/ggml/src)

target_link_libraries(${CMAKE_PROJECT_NAME}
        llama
        common
        android
        log)
