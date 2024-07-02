cmake_minimum_required(VERSION 3.12)
project("llama-cli-cmake-pkg" C CXX)
set(TARGET llama-cli-cmake-pkg)

find_package(Llama 0.0.1 REQUIRED)

# Bake common functionality in with target. Because applications
# using the relocatable Llama package should be outside of the
# source tree, llama-cli-cmake-pkg pretends the dependencies are built-in.
set(_common_path "${CMAKE_CURRENT_LIST_DIR}/../../common")
add_library(common OBJECT)
file(GLOB _common_files
    "${_common_path}/*.h"
    "${_common_path}/*.cpp"
)
target_sources(common PRIVATE ${_common_files})

# If the common project was part of "llama-cli-cmake-pkg" the transient
# defines would automatically be attached. Because the common func-
# tionality is separate, but dependent upon the defines, it must be
# explicitly extracted from the "llama" target.
#
get_target_property(_llama_transient_defines llama
    INTERFACE_COMPILE_DEFINITIONS)

target_compile_definitions(common PRIVATE "${_llama_transient_defines}")

add_executable(${TARGET} ${CMAKE_CURRENT_LIST_DIR}/../main/main.cpp)
target_include_directories(${TARGET} PRIVATE ${_common_path})
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
