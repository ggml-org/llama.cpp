cmake_minimum_required(VERSION 3.14)

file(REMOVE_RECURSE "${TEST_BINARY_DIR}")
file(MAKE_DIRECTORY "${TEST_BINARY_DIR}/source")
file(WRITE "${TEST_BINARY_DIR}/source/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.14)
project(test-cmake-video-options LANGUAGES C CXX)
set(LLAMA_BUILD_TOOLS OFF)
set(LLAMA_VERSION_BASE 0.0.0)
set(LLAMA_VERSION_MAJOR 0)
# Configure the real mtmd target; its unrelated link dependencies need no build.
foreach(dependency ggml llama vendor::hash vendor::miniaudio vendor::stb vendor::sheredom)
    add_library(${dependency} INTERFACE IMPORTED)
endforeach()
add_subdirectory("${LLAMA_SOURCE_DIR}/tools/mtmd" mtmd)
get_target_property(definitions mtmd COMPILE_DEFINITIONS)
if ("MTMD_VIDEO" IN_LIST definitions)
    file(WRITE "${CMAKE_BINARY_DIR}/video-enabled" "ON")
else()
    file(WRITE "${CMAKE_BINARY_DIR}/video-enabled" "OFF")
endif()
]=])

function(check_video subprocess requested effective)
    execute_process(COMMAND "${CMAKE_COMMAND}"
        -S "${TEST_BINARY_DIR}/source" -B "${TEST_BINARY_DIR}/build"
        "-DLLAMA_SOURCE_DIR=${LLAMA_SOURCE_DIR}" "-DLLAMA_SUBPROCESS=${subprocess}"
        ${ARGN}
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if (NOT result EQUAL 0)
        message(FATAL_ERROR "mtmd configure failed: ${output}\n${error}")
    endif()
    load_cache("${TEST_BINARY_DIR}/build" READ_WITH_PREFIX cached_ MTMD_VIDEO)
    file(READ "${TEST_BINARY_DIR}/build/video-enabled" actual)
    if (NOT cached_MTMD_VIDEO STREQUAL requested OR NOT actual STREQUAL effective)
        message(FATAL_ERROR "subprocess=${subprocess}: requested=${cached_MTMD_VIDEO}, effective=${actual}; expected ${requested}/${effective}")
    endif()
endfunction()

# Reuse one cache: dependency toggles must not replace the user's preference.
check_video(OFF ON OFF)
check_video(ON ON ON)
check_video(OFF OFF OFF -DMTMD_VIDEO=OFF)
check_video(ON OFF OFF)
