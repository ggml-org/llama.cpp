find_package(Threads REQUIRED)

llama_add_compile_flags()

#
# llama-common-base
#

# Build info header

if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
    set(GIT_DIR "${PROJECT_SOURCE_DIR}/.git")

    # Is git submodule
    if(NOT IS_DIRECTORY "${GIT_DIR}")
        file(READ ${GIT_DIR} REAL_GIT_DIR_LINK)
        string(REGEX REPLACE "gitdir: (.*)\n$" "\\1" REAL_GIT_DIR ${REAL_GIT_DIR_LINK})
        string(FIND "${REAL_GIT_DIR}" "/" SLASH_POS)
        if (SLASH_POS EQUAL 0)
            set(GIT_DIR "${REAL_GIT_DIR}")
        else()
            set(GIT_DIR "${PROJECT_SOURCE_DIR}/${REAL_GIT_DIR}")
        endif()
    endif()

    if(EXISTS "${GIT_DIR}/index")
        # For build-info.cpp below
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${GIT_DIR}/index")
    else()
        message(WARNING "Git index not found in git repository.")
    endif()
else()
    message(WARNING "Git repository not found; to enable automatic generation of build info, make sure Git is installed and the project is a Git repository.")
endif()

set(TEMPLATE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/build-info.cpp.in")
set(OUTPUT_FILE   "${CMAKE_CURRENT_BINARY_DIR}/build-info.cpp")

configure_file(${TEMPLATE_FILE} ${OUTPUT_FILE})

set(TARGET llama-common-base)
add_library(${TARGET} STATIC ${OUTPUT_FILE})

target_include_directories(${TARGET} PUBLIC .)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

#
# llama-common
#

set(TARGET llama-common)

add_library(${TARGET}
    arg.cpp
    arg.h
    base64.hpp
    chat-auto-parser-generator.cpp
    chat-auto-parser-helpers.cpp
    chat-auto-parser.h
    chat-diff-analyzer.cpp
    chat-peg-parser.cpp
    chat-peg-parser.h
    chat.cpp
    chat.h
    common.cpp
    common.h
    console.cpp
    console.h
    debug.cpp
    debug.h
    download.cpp
    download.h
    fit.cpp
    fit.h
    hf-cache.cpp
    hf-cache.h
    http.h
    imatrix-loader.cpp
    imatrix-loader.h
    json-schema-to-grammar.cpp
    llguidance.cpp
    ngram-cache.cpp
    ngram-cache.h
    ngram-map.cpp
    ngram-map.h
    ngram-mod.cpp
    ngram-mod.h
    peg-parser.cpp
    peg-parser.h
    preset.cpp
    preset.h
    reasoning-budget.cpp
    reasoning-budget.h
    sampling.cpp
    sampling.h
    speculative-calibration.cpp
    speculative-calibration.h
    speculative.cpp
    speculative.h
    tessera-analytical-io.cpp
    tessera-analytical-io.h
    tessera-config.cpp
    tessera-config.h
    tessera-runtime.cpp
    tessera-runtime.h
    trie.cpp
    trie.h
    unicode.cpp
    unicode.h
    jinja/lexer.cpp
    jinja/lexer.h
    jinja/parser.cpp
    jinja/parser.h
    jinja/runtime.cpp
    jinja/runtime.h
    jinja/value.cpp
    jinja/value.h
    jinja/string.cpp
    jinja/string.h
    jinja/caps.cpp
    jinja/caps.h
    )

if (APPLE)
    target_sources(${TARGET} PRIVATE
        ane-mtp.mm ane-mtp.h
        ane-pump.mm ane-pump.h)
    set_source_files_properties(ane-mtp.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    set_source_files_properties(ane-pump.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    # ggml-metal provides ggml_mtl_shared_event_t for the W7
    # MTLSharedEvent handoff per state slot. The events are
    # signaled by the ANE dispatch path on OUTPUT_READY and
    # consumed by the Metal backend (or by the host via
    # ggml_mtl_shared_event_wait). The dependency is
    # architecturally justified: the events are the cross-timeline
    # lock-free primitive between the ANE side (in common) and
    # the Metal side (in ggml). ggml-base pulls in nothing extra.
    target_link_libraries(${TARGET} PRIVATE
        ggml ggml-base
        "-framework CoreML" "-framework Foundation" "-framework IOSurface")
endif()

set_target_properties(${TARGET} PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(${TARGET} PUBLIC . ./tessera-debug ../vendor)
target_compile_features   (${TARGET} PUBLIC cxx_std_17)

# Bake the in-repo schema path into the analytical-IO helper so it can
# locate the schema files at runtime without an env-var set. Installed
# builds override this via TESSERA_SCHEMA_DIR at runtime.
target_compile_definitions(${TARGET} PRIVATE
    TESSERA_SCHEMA_DIR="${CMAKE_SOURCE_DIR}/common/schemas")

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # TODO: make fine-grained exports in the future
    set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_link_libraries(${TARGET} PUBLIC  llama-common-base)
target_link_libraries(${TARGET} PUBLIC  llama-log)
target_link_libraries(${TARGET} PRIVATE cpp-httplib)

if (LLAMA_LLGUIDANCE)
    include(ExternalProject)
    set(LLGUIDANCE_SRC ${CMAKE_BINARY_DIR}/llguidance/source)
    set(LLGUIDANCE_PATH ${LLGUIDANCE_SRC}/target/release)
    set(LLGUIDANCE_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}llguidance${CMAKE_STATIC_LIBRARY_SUFFIX}")

    ExternalProject_Add(llguidance_ext
        GIT_REPOSITORY https://github.com/guidance-ai/llguidance
        # v1.0.1:
        GIT_TAG d795912fedc7d393de740177ea9ea761e7905774
        PREFIX ${CMAKE_BINARY_DIR}/llguidance
        SOURCE_DIR ${LLGUIDANCE_SRC}
        BUILD_IN_SOURCE TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND cargo build --release --package llguidance
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS ${LLGUIDANCE_PATH}/${LLGUIDANCE_LIB_NAME} ${LLGUIDANCE_PATH}/llguidance.h
        UPDATE_COMMAND ""
    )
    target_compile_definitions(${TARGET} PUBLIC LLAMA_USE_LLGUIDANCE)

    add_library(llguidance STATIC IMPORTED)
    set_target_properties(llguidance PROPERTIES IMPORTED_LOCATION ${LLGUIDANCE_PATH}/${LLGUIDANCE_LIB_NAME})
    add_dependencies(llguidance llguidance_ext)

    target_include_directories(${TARGET} PRIVATE ${LLGUIDANCE_PATH})
    target_link_libraries(${TARGET} PRIVATE llguidance)
    if (WIN32)
        target_link_libraries(${TARGET} PRIVATE ws2_32 userenv ntdll bcrypt)
    endif()
endif()

target_link_libraries(${TARGET} PUBLIC llama Threads::Threads)

# Low-level logging primitives. Built as its own static target so the
# backends (ggml-cpu, ggml-cuda, ggml-metal) and the tessera-debug
# sidecar writer can use the LOG_* macros without dragging in the
# whole common/ library. Guarded so the first caller wins.
if (NOT TARGET llama-log)
    add_subdirectory(${CMAKE_SOURCE_DIR}/common/log ${CMAKE_BINARY_DIR}/_deps/llama-log)
endif()

# Tessera Layer 1 sidecar writer. Built as its own static target so
# the backends (ggml-cpu, ggml-cuda, ggml-metal) can link against it
# directly without dragging in the whole common/ library. The first
# caller (whichever backend configures first) wins; subsequent calls
# from the other backends are no-ops.
if (NOT TARGET llama-tessera-debug)
    add_subdirectory(${CMAKE_SOURCE_DIR}/common/tessera-debug ${CMAKE_BINARY_DIR}/_deps/llama-tessera-debug)
endif()

target_link_libraries(${TARGET} PUBLIC llama-tessera-debug)

# Standalone smoke test for the analytical-IO helper
# (tessera-analytical-io.{h,cpp}). Sibling of the tessera-debug
# smoke test pattern: kept here (rather than in a subdir) because
# the helper lives in the main common/ library, and a missing link
# should not break the rest of common/. Run as
# ./bin/test-tessera-analytical-io.
if (LLAMA_BUILD_TESTS)
    set(TS_ANALYTICAL_IO_TEST_TARGET test-tessera-analytical-io)
    add_executable(${TS_ANALYTICAL_IO_TEST_TARGET} test_tessera_analytical_io.cpp)
    target_link_libraries(${TS_ANALYTICAL_IO_TEST_TARGET} PRIVATE ${TARGET})
    target_compile_features(${TS_ANALYTICAL_IO_TEST_TARGET} PRIVATE cxx_std_17)
    add_test(NAME ${TS_ANALYTICAL_IO_TEST_TARGET} COMMAND ${TS_ANALYTICAL_IO_TEST_TARGET})

    # Companion emit binary: writes a sample L3-outlier NDJSON file
    # via the helper. Used by the Python end-to-end test
    # (tools/tessera/test_analytical_io.py) to verify the cross-
    # language C++ writer -> polars reader contract.
    set(TS_EMIT_L3_TARGET test-emit-l3-ndjson)
    add_executable(${TS_EMIT_L3_TARGET} test_emit_l3_ndjson.cpp)
    target_link_libraries(${TS_EMIT_L3_TARGET} PRIVATE ${TARGET})
    target_compile_features(${TS_EMIT_L3_TARGET} PRIVATE cxx_std_17)
endif()
