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
    dist-comm.cpp
    dist-comm.h
    dist-comm-nd.cpp
    dist-comm-nd.h
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
    log.cpp
    log.h
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
    speculative.cpp
    speculative.h
    subproc.cpp
    subproc.h
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

set_target_properties(${TARGET} PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(${TARGET} PUBLIC . ../vendor)
target_compile_features   (${TARGET} PUBLIC cxx_std_17)

if (LLAMA_SUBPROCESS)
    target_compile_definitions(${TARGET} PUBLIC LLAMA_SUBPROCESS)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # TODO: make fine-grained exports in the future
    set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

target_link_libraries(${TARGET} PUBLIC  llama-common-base)
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

if (WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()

#
# NetworkDirect (Windows RDMA) data path for distributed tensor parallelism.
#
# Opt in at build time with -DLLAMA_DIST_TP_ND=ON -DLLAMA_DIST_TP_ND_SDK=<NetworkDirect checkout>.
# At run time an edge then uses NetworkDirect when both of its ranks have an ND capable address on
# one subnet; LLAMA_DIST_TP_RDMA=0 keeps a rank's edges on TCP. Without the option every edge runs
# TCP, as before.
#
if (WIN32)
    option(LLAMA_DIST_TP_ND "llama: enable the NetworkDirect RDMA data path for distributed TP" OFF)
    # GGML_RPC_ND_SDK is accepted as a default so the ggml-rpc and dist-tp builds can share one flag
    set(LLAMA_DIST_TP_ND_SDK "${GGML_RPC_ND_SDK}" CACHE PATH
        "llama: path to a NetworkDirect checkout providing src/ndutil")
else()
    set(LLAMA_DIST_TP_ND OFF CACHE BOOL "NetworkDirect is Windows only" FORCE)
endif()

if (LLAMA_DIST_TP_ND)
    if (NOT LLAMA_DIST_TP_ND_SDK)
        message(FATAL_ERROR "LLAMA_DIST_TP_ND requires LLAMA_DIST_TP_ND_SDK to point at a NetworkDirect checkout")
    endif()
    set(LLAMA_ND_UTIL_DIR "${LLAMA_DIST_TP_ND_SDK}/src/ndutil")
    foreach (ND_SRC ndaddr.cpp ndfrmwrk.cpp ndprov.cpp)
        if (NOT EXISTS "${LLAMA_ND_UTIL_DIR}/${ND_SRC}")
            message(FATAL_ERROR "LLAMA_DIST_TP_ND_SDK does not contain src/ndutil/${ND_SRC}")
        endif()
    endforeach()
    # ndstatus.h is generated from ndstatus.mc into the SDK's build output tree
    find_path(LLAMA_DIST_TP_ND_STATUS_DIR ndstatus.h
              PATHS "${LLAMA_DIST_TP_ND_SDK}/out"
              PATH_SUFFIXES Release-x64/include Debug-x64/include
              NO_DEFAULT_PATH)
    if (NOT LLAMA_DIST_TP_ND_STATUS_DIR)
        message(FATAL_ERROR "ndstatus.h not found under ${LLAMA_DIST_TP_ND_SDK}/out - build the NetworkDirect SDK first")
    endif()
    # the prebuilt ndutil.lib is /MT, so build the loader sources here instead of linking it.
    # They go in a static library of their own, outside the export-all scan of this DLL: MSVC
    # emits two external symbols in ndprov.cpp named plainly == and !=, and once they are
    # written to the generated .def file the link fails with LNK2001.
    add_library(llama-nd-util STATIC
        "${LLAMA_ND_UTIL_DIR}/ndaddr.cpp"
        "${LLAMA_ND_UTIL_DIR}/ndfrmwrk.cpp"
        "${LLAMA_ND_UTIL_DIR}/ndprov.cpp")
    target_include_directories(llama-nd-util PRIVATE "${LLAMA_ND_UTIL_DIR}" "${LLAMA_DIST_TP_ND_STATUS_DIR}")
    target_compile_definitions(llama-nd-util PRIVATE _CRT_SECURE_NO_WARNINGS)
    if (MSVC)
        target_compile_options(llama-nd-util PRIVATE /W0)
    else()
        target_compile_options(llama-nd-util PRIVATE -w)
    endif()
    target_link_libraries(${TARGET} PRIVATE llama-nd-util iphlpapi)
    target_include_directories(${TARGET} PRIVATE "${LLAMA_ND_UTIL_DIR}" "${LLAMA_DIST_TP_ND_STATUS_DIR}")
    target_compile_definitions(${TARGET} PRIVATE LLAMA_DIST_TP_ND _CRT_SECURE_NO_WARNINGS)
    message(STATUS "llama: NetworkDirect dist-TP transport enabled (${LLAMA_DIST_TP_ND_SDK})")
else()
    message(STATUS "llama: NetworkDirect dist-TP transport disabled")
endif()
