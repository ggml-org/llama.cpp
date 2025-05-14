# ============================================================================
#  Multi-tier Android build for llama.cpp
#  --------------------------------------
#  Produces five DSOs, each compiled with an increasingly aggressive
#  -march string.  At runtime you pick the highest tier the device
#  supports and call `System.loadLibrary("llama_android_tX")`.
# ============================================================================

cmake_minimum_required(VERSION 3.22.1)
project("llama_android" LANGUAGES C CXX)

# --------------------------------------------------------------------------
# 0. Language / toolchain defaults
# --------------------------------------------------------------------------
set(CMAKE_C_STANDARD   11  CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD 17  CACHE STRING "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --------------------------------------------------------------------------
# 1. Locate the root of the llama.cpp source tree
#    (six levels up from this CMakeLists.txt).
# --------------------------------------------------------------------------
set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../../../)

# --------------------------------------------------------------------------
# 2. Build helper  –  one invocation = one hardware tier
# --------------------------------------------------------------------------
include(ExternalProject)

function(build_llama_tier tier march)
    # ---------- 2.1  configure & build core code in an external project -----
    set(build_dir ${CMAKE_BINARY_DIR}/llama_build_${tier})

    # KleidiAI requires dotprod and i8mm
    if(${tier} STREQUAL "t0" OR ${tier} STREQUAL "t1")
        set(kleidi OFF)
    else()
        set(kleidi ON)
    endif()

    ExternalProject_Add(llama_build_${tier}
        SOURCE_DIR   ${LLAMA_SRC}
        BINARY_DIR   ${build_dir}
        # ---- pass Android cross-compile context straight through ----------
        CMAKE_ARGS
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DANDROID_ABI=${ANDROID_ABI}
            -DANDROID_PLATFORM=${ANDROID_PLATFORM}
            -DANDROID_STL=${ANDROID_STL}
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            # ---- llama / ggml feature switches ----------------------------
            -DGGML_CPU_KLEIDIAI=${kleidi}
            -DGGML_LLAMAFILE=OFF
            -DGGML_OPENMP=OFF
            -DLLAMA_BUILD_COMMON=ON
            -DLLAMA_CURL=OFF
            -DBUILD_SHARED_LIBS=OFF       # we want static libs to embed
            # ---- tier-specific ISA flags ----------------------------------
            -DCMAKE_C_FLAGS=-march=${march}
            -DCMAKE_CXX_FLAGS=-march=${march}
            # ---- put the .a files right in ${build_dir} for easy pick-up --
            -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=${build_dir}

        INSTALL_COMMAND ""                # nothing to install

        BUILD_BYPRODUCTS
            ${build_dir}/libllama.a
            ${build_dir}/libcommon.a
            ${build_dir}/libggml.a
            ${build_dir}/libggml-base.a
            ${build_dir}/libggml-cpu.a
    )

    # ---------- 2.2  make the static libs produced above visible ------------
    set(llama_a   ${build_dir}/libllama.a)
    set(common_a  ${build_dir}/libcommon.a)
    set(ggml_a    ${build_dir}/libggml.a)
    set(ggml_base_a ${build_dir}/libggml-base.a)
    set(ggml_cpu_a  ${build_dir}/libggml-cpu.a)

    add_library(llama_core_${tier}  STATIC IMPORTED GLOBAL)
    set_target_properties(llama_core_${tier}  PROPERTIES
        IMPORTED_LOCATION ${llama_a})
    add_dependencies(llama_core_${tier} llama_build_${tier})

    add_library(common_core_${tier} STATIC IMPORTED GLOBAL)
    set_target_properties(common_core_${tier} PROPERTIES
        IMPORTED_LOCATION ${common_a})
    add_dependencies(common_core_${tier} llama_build_${tier})

    add_library(ggml_core_${tier}   STATIC IMPORTED GLOBAL)
    set_target_properties(ggml_core_${tier} PROPERTIES
        IMPORTED_LOCATION ${ggml_a})
    add_dependencies(ggml_core_${tier} llama_build_${tier})

    add_library(ggml_base_core_${tier} STATIC IMPORTED GLOBAL)
    set_target_properties(ggml_base_core_${tier} PROPERTIES
        IMPORTED_LOCATION ${ggml_base_a})
    add_dependencies(ggml_base_core_${tier} llama_build_${tier})

    add_library(ggml_cpu_core_${tier}  STATIC IMPORTED GLOBAL)
    set_target_properties(ggml_cpu_core_${tier} PROPERTIES
        IMPORTED_LOCATION ${ggml_cpu_a})
    add_dependencies(ggml_cpu_core_${tier} llama_build_${tier})

    # ---------- 2.3  JNI wrapper DSO ---------------------------------------
    add_library(llama_android_${tier} SHARED llama-android.cpp)

    target_compile_options(llama_android_${tier} PRIVATE "-march=${march}")

    target_include_directories(llama_android_${tier} PRIVATE
        ${LLAMA_SRC}
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/ggml/src)

    target_link_libraries(llama_android_${tier} PRIVATE
        llama_core_${tier}
        common_core_${tier}
        ggml_core_${tier}        # umbrella (brings in few weak deps)
        ggml_cpu_core_${tier}    # back-end & scheduler
        ggml_base_core_${tier}   # core math
        android
        log)

    # ---------- 2.4  nice SONAME & filename -------------------------------
    set_target_properties(llama_android_${tier} PROPERTIES
        OUTPUT_NAME "llama_android_${tier}")
endfunction()

# --------------------------------------------------------------------------
# 3.  Build all five tiers
# --------------------------------------------------------------------------
build_llama_tier(t0 "armv8-a+simd")
build_llama_tier(t1 "armv8.2-a+dotprod")
build_llama_tier(t2 "armv8.6-a+dotprod+i8mm")
build_llama_tier(t3 "armv9-a+dotprod+i8mm+sve+sve2")
#build_llama_tier(t4 "armv9.2-a+dotprod+i8mm+sve+sve2+sme")

add_dependencies(llama_build_t1 llama_build_t0)
add_dependencies(llama_build_t2 llama_build_t1)
add_dependencies(llama_build_t3 llama_build_t2)
#add_dependencies(llama_build_t4 llama_build_t3)

# --------------------------------------------------------------------------
# 4.  Default variant when Gradle hasn’t told us (keeps IDE happy)
# --------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
