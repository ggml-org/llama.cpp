llama_add_compile_flags()

#
# llama-tessera-debug
#
# Tessera Layer 1 sidecar writer. Lives below the backends
# (ggml-cpu / ggml-cuda / ggml-metal) so the backends can link against
# it directly without dragging in the whole common/ library. Common
# itself links against this target so arg.cpp's --tessera-dequant-dir
# CLI flag still resolves the same symbols.
#

# Auto-populated build info. The values are baked into the binary at
# compile time so the sidecar's provenance JSON can record the exact
# branch + commit and main tip used to build the dump tool, without an
# out-of-band config. When git is not present at configure time (e.g.,
# release tarballs) the fallback values ("unknown" for both) are used.
#
# Reconfigure on HEAD change: the .git/HEAD file and the ref it
# points at both feed CMake's configure-depends mechanism so a
# commit on this branch triggers a re-run of configure_file.
set(TESSERA_KERNEL_VERSION "unknown")
set(TESSERA_MAIN_TIP        "unknown")

find_package(Git QUIET)
if(NOT Git_FOUND)
    find_program(GIT_EXECUTABLE NAMES git git.exe)
    if(GIT_EXECUTABLE)
        set(Git_FOUND TRUE)
    endif()
endif()

if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..
        OUTPUT_VARIABLE TESSERA_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TESSERA_BRANCH_RC
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..
        OUTPUT_VARIABLE TESSERA_HEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TESSERA_HEAD_RC
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short main
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..
        OUTPUT_VARIABLE TESSERA_MAIN_SHA
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TESSERA_MAIN_RC
    )
    if(TESSERA_BRANCH_RC EQUAL 0 AND TESSERA_HEAD_RC EQUAL 0)
        set(TESSERA_KERNEL_VERSION "${TESSERA_BRANCH} ${TESSERA_HEAD}")
    endif()
    if(TESSERA_MAIN_RC EQUAL 0)
        set(TESSERA_MAIN_TIP "${TESSERA_MAIN_SHA}")
    endif()

    # Make CMake re-run configure_file when the branch tip changes.
    # Both HEAD (the symbolic ref) and the ref it points at are
    # dependencies; either one changing triggers a re-configure.
    set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../.git")
    if(IS_DIRECTORY "${GIT_DIR}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${GIT_DIR}/HEAD")
        # The ref file lives at .git/refs/heads/<branch>; we add the
        # current branch's ref so a commit on this branch re-runs
        # the configure step.
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${GIT_DIR}/refs/heads/${TESSERA_BRANCH}")
    endif()
endif()

set(TESSERA_BUILD_INFO_IN  "${CMAKE_CURRENT_SOURCE_DIR}/tessera-build-info.h.in")
set(TESSERA_BUILD_INFO_OUT "${CMAKE_BINARY_DIR}/tessera-build-info.h")
configure_file(${TESSERA_BUILD_INFO_IN} ${TESSERA_BUILD_INFO_OUT} @ONLY)

set(TARGET llama-tessera-debug)

add_library(${TARGET} STATIC
    tessera-debug.cpp
    tessera-debug.h
    tessera-matmul-output.cpp
    tessera-matmul-output.h
    ${TESSERA_BUILD_INFO_OUT}
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}       # tessera-debug.h via "tessera-debug.h"
    ${CMAKE_BINARY_DIR}               # tessera-build-info.h via "tessera-build-info.h"
)

target_compile_features(${TARGET} PUBLIC cxx_std_17)

# LOG_WRN/LOG_ERR in tessera-debug.cpp need the low-level log macros
# from llama-log. Guarded so the first caller wins; either common/
# (which already links llama-log) or the backends may bring it in
# first.
if (NOT TARGET llama-log)
    add_subdirectory(${CMAKE_SOURCE_DIR}/common/log ${CMAKE_BINARY_DIR}/_deps/llama-log)
endif()
target_link_libraries(${TARGET} PUBLIC llama-log)

# Standalone smoke test for the L2 matmul-output sidecar writer
# (tessera-matmul-output.{h,cpp}). Mirrors the L1 sidecar test pattern
# (test_sidecar_v3.cpp) - kept as a sibling file rather than wired to
# ctest so a missing link (e.g. building outside tests/) does not break
# the sidecar library itself. Run as: ./bin/test-tessera-matmul-output.
if (LLAMA_BUILD_TESTS)
    set(TS_MM_OUT_TEST_TARGET test-tessera-matmul-output)
    add_executable(${TS_MM_OUT_TEST_TARGET} test_matmul_output.cpp)
    target_link_libraries(${TS_MM_OUT_TEST_TARGET} PRIVATE ${TARGET})
    target_compile_features(${TS_MM_OUT_TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${TS_MM_OUT_TEST_TARGET} COMMAND ${TS_MM_OUT_TEST_TARGET})
endif()
