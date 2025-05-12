cmake_minimum_required(VERSION 3.22.1)
project("llama_android" VERSION 1.0.0 LANGUAGES C CXX)

# --------------------------------------------------------------------
# 0. Global language & toolchain options
# --------------------------------------------------------------------
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

## ---- ggml / llama feature switches (configure-time) ----------------
#set(GGML_CPU_KLEIDIAI     ON  CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_CURL         OFF CACHE BOOL "" FORCE)

# baseline ISA used for every source that is NOT in a tier wrapper
string(APPEND CMAKE_C_FLAGS   " -march=armv9.2-a+dotprod+i8mm+sve+sve2+sme")
string(APPEND CMAKE_CXX_FLAGS " -march=armv9.2-a+dotprod+i8mm+sve+sve2+sme")
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)

# NOT NEEDED WHEN CMAKE_BUILD_TYPE PASSED IN BY GRADLE
#message(DEBUG "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
#if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#    string(APPEND CMAKE_C_FLAGS   " -O3 -DNDEBUG")
#    string(APPEND CMAKE_CXX_FLAGS " -O3 -DNDEBUG")
#endif()

# default to Release if Gradle hasn’t set a variant
if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

### Also provides "common"
##include(FetchContent)
##FetchContent_Declare(
##        llama
##        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
##        GIT_TAG        master
##)
##FetchContent_MakeAvailable(llama)

# --------------------------------------------------------------------
# 1. Pull in llama.cpp (creates targets "llama" and "common")
# --------------------------------------------------------------------
set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../)
add_subdirectory(${LLAMA_SRC} build-llama)

# -----------------------------------------------------------
# 2.  Helper to create one wrapper .so for a hardware tier
# -----------------------------------------------------------
function(build_llama_tier tier march)
    set(wrapper llama_android_${tier})
    add_library(${wrapper} SHARED llama-android.cpp)

    # ISA tuning that belongs only to this tier
    target_compile_options(${wrapper} PRIVATE "-march=${march}")

# NOT NEEDED!
#    # sampling.h lives in common/, ggml-impl.h lives in ggml/src
#    target_include_directories(${wrapper} PRIVATE
#            ${LLAMA_SRC}/common
#            ${LLAMA_SRC}/ggml/src)

    target_link_libraries(${wrapper} PRIVATE
            llama
            common
            android
            log)

    set_target_properties(${wrapper} PROPERTIES OUTPUT_NAME ${wrapper})
endfunction()

# -----------------------------------------------------------
# 3.  Build the five tiers (no KleidiAI for now)
# -----------------------------------------------------------
build_llama_tier(t0 "armv8-a+simd")
build_llama_tier(t1 "armv8.2-a+dotprod")
build_llama_tier(t2 "armv8.6-a+dotprod+i8mm")
build_llama_tier(t3 "armv9-a+dotprod+i8mm+sve+sve2")
build_llama_tier(t4 "armv9.2-a+dotprod+i8mm+sve+sve2+sme")
