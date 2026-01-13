# common

find_package(Threads REQUIRED)

llama_add_compile_flags()

# Build info header
#

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

set(TARGET build_info)
add_library(${TARGET} OBJECT ${OUTPUT_FILE})
if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

set(TARGET common)

add_library(${TARGET} STATIC
    arg.cpp
    arg.h
    base64.hpp
    chat-parser.cpp
    chat-parser.h
    chat-parser-xml-toolcall.h
    chat-parser-xml-toolcall.cpp
    chat-peg-parser.cpp
    chat-peg-parser.h
    chat.cpp
    chat.h
    common.cpp
    common.h
    console.cpp
    console.h
    download.cpp
    download.h
    http.h
    json-partial.cpp
    json-partial.h
    json-schema-to-grammar.cpp
    llguidance.cpp
    log.cpp
    log.h
    ngram-cache.cpp
    ngram-cache.h
    peg-parser.cpp
    peg-parser.h
    preset.cpp
    preset.h
    regex-partial.cpp
    regex-partial.h
    sampling.cpp
    sampling.h
    speculative.cpp
    speculative.h
    unicode.cpp
    unicode.h
    )

target_include_directories(${TARGET} PUBLIC . ../vendor)
target_compile_features   (${TARGET} PUBLIC cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# TODO: use list(APPEND LLAMA_COMMON_EXTRA_LIBS ...)
set(LLAMA_COMMON_EXTRA_LIBS build_info)

if (LLAMA_CURL)
    # Use curl to download model url
    find_package(CURL)
    if (NOT CURL_FOUND)
        message(FATAL_ERROR "Could NOT find CURL. Hint: to disable this feature, set -DLLAMA_CURL=OFF")
    endif()
    target_compile_definitions(${TARGET} PUBLIC LLAMA_USE_CURL)
    include_directories(${CURL_INCLUDE_DIRS})
    set(LLAMA_COMMON_EXTRA_LIBS ${LLAMA_COMMON_EXTRA_LIBS} ${CURL_LIBRARIES})
elseif (LLAMA_HTTPLIB)
    # otherwise, use cpp-httplib
    target_compile_definitions(${TARGET} PUBLIC LLAMA_USE_HTTPLIB)
    set(LLAMA_COMMON_EXTRA_LIBS ${LLAMA_COMMON_EXTRA_LIBS} cpp-httplib)
endif()

if (LLAMA_LLGUIDANCE)
    include(ExternalProject)
    set(LLGUIDANCE_SRC ${CMAKE_BINARY_DIR}/llguidance/source)
    set(LLGUIDANCE_PATH ${LLGUIDANCE_SRC}/target/release)

    # Set the correct library file extension based on platform
    if (WIN32)
        set(LLGUIDANCE_LIB_NAME "llguidance.lib")
        # Add Windows-specific libraries
        set(LLGUIDANCE_PLATFORM_LIBS
            ws2_32    # Windows Sockets API
            userenv   # For GetUserProfileDirectoryW
            ntdll     # For NT functions
            bcrypt    # For BCryptGenRandom
        )
    else()
        set(LLGUIDANCE_LIB_NAME "libllguidance.a")
        set(LLGUIDANCE_PLATFORM_LIBS "")
    endif()

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
    # Add platform libraries to the main target
    set(LLAMA_COMMON_EXTRA_LIBS ${LLAMA_COMMON_EXTRA_LIBS} llguidance ${LLGUIDANCE_PLATFORM_LIBS})
endif ()

target_link_libraries(${TARGET} PRIVATE ${LLAMA_COMMON_EXTRA_LIBS} PUBLIC llama Threads::Threads)
