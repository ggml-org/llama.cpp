# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

project("llama_android" VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(GGML_CPU_KLEIDIAI     ON  CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE        OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON    ON  CACHE BOOL "" FORCE)
set(LLAMA_CURL            OFF CACHE BOOL "" FORCE)

string(APPEND CMAKE_C_FLAGS     " -march=armv8.7-a+dotprod+i8mm+sve")
string(APPEND CMAKE_CXX_FLAGS   " -march=armv8.7-a+dotprod+i8mm+sve")

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)

## CMake definitions
#add_compile_definitions(
#    GGML_CPU_KLEIDIAI=ON
#    GGML_LLAMAFILE=OFF
#    LLAMA_BUILD_COMMON=ON
#    LLAMA_CURL=OFF
#)

# Force Release only when Gradle didn't specify one
if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

## Also provides "common"
#include(FetchContent)
#FetchContent_Declare(
#        llama
#        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
#        GIT_TAG        master
#)
#FetchContent_MakeAvailable(llama)

# Add local llama.cpp as subdirectory
add_subdirectory(../../../../../../ build-llama)

#function(build_llama_tier tier march_flag)
#    set(LIB_NAME "${CMAKE_PROJECT_NAME}_${tier}")
#
##    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.7-a+i8mm")
##    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.7-a+i8mm")
#
#    add_library(${LIB_NAME} SHARED llama-android.cpp)
#

# MIGHT NOT WORK????
# target_include_directories(${LIB_NAME} PRIVATE
#        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../common)
# ?????

#    target_compile_options(${LIB_NAME} PRIVATE "-march=${march_flag}")
#
#    target_link_libraries(${LIB_NAME}
#            llama
#            common
#            android
#            log
#    )
#
#    set_target_properties(${LIB_NAME} PROPERTIES
#            OUTPUT_NAME "llama_android_${tier}"
#    )
#endfunction()
#
#build_llama_tier(t0 "armv8-a+simd")
#build_llama_tier(t1 "armv8.2-a+dotprod")
#build_llama_tier(t2 "armv8.6-a+dotprod+i8mm")
#build_llama_tier(t3 "armv9-a+dotprod+i8mm+sve+sve2")
#build_llama_tier(t4 "armv9.2-a+dotprod+i8mm+sve+sve2+sme")


add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        llama-android.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME}
        # List libraries link to the target library
        llama
        common
        android
        log)
