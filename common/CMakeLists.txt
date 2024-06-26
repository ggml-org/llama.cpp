# common

find_package(Threads REQUIRED)

# Build info header
#

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../.git")
    set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.git")

    # Is git submodule
    if(NOT IS_DIRECTORY "${GIT_DIR}")
        file(READ ${GIT_DIR} REAL_GIT_DIR_LINK)
        string(REGEX REPLACE "gitdir: (.*)\n$" "\\1" REAL_GIT_DIR ${REAL_GIT_DIR_LINK})
        string(FIND "${REAL_GIT_DIR}" "/" SLASH_POS)
        if (SLASH_POS EQUAL 0)
            set(GIT_DIR "${REAL_GIT_DIR}")
        else()
            set(GIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../${REAL_GIT_DIR}")
        endif()
    endif()

    if(EXISTS "${GIT_DIR}/index")
        set(GIT_INDEX "${GIT_DIR}/index")
    else()
        message(WARNING "Git index not found in git repository.")
        set(GIT_INDEX "")
    endif()
else()
    message(WARNING "Git repository not found; to enable automatic generation of build info, make sure Git is installed and the project is a Git repository.")
    set(GIT_INDEX "")
endif()

# Add a custom command to rebuild build-info.cpp when .git/index changes
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/build-info.cpp"
    COMMENT "Generating build details from Git"
    COMMAND ${CMAKE_COMMAND} -DMSVC=${MSVC} -DCMAKE_C_COMPILER_VERSION=${CMAKE_C_COMPILER_VERSION}
            -DCMAKE_C_COMPILER_ID=${CMAKE_C_COMPILER_ID} -DCMAKE_VS_PLATFORM_NAME=${CMAKE_VS_PLATFORM_NAME}
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info-gen-cpp.cmake"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/build-info.cpp.in" ${GIT_INDEX}
    VERBATIM
)
set(TARGET build_info)
add_library(${TARGET} OBJECT build-info.cpp)
if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

set(TARGET common)

add_library(${TARGET} STATIC
    base64.hpp
    common.h
    common.cpp
    sampling.h
    sampling.cpp
    console.h
    console.cpp
    grammar-parser.h
    grammar-parser.cpp
    json.hpp
    json-schema-to-grammar.cpp
    train.h
    train.cpp
    ngram-cache.h
    ngram-cache.cpp
    )

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

set(LLAMA_COMMON_EXTRA_LIBS build_info)

# Use curl to download model url
if (LLAMA_CURL)
    find_package(CURL REQUIRED)
    add_definitions(-DLLAMA_USE_CURL)
    include_directories(${CURL_INCLUDE_DIRS})
    find_library(CURL_LIBRARY curl REQUIRED)
    set(LLAMA_COMMON_EXTRA_LIBS ${LLAMA_COMMON_EXTRA_LIBS} ${CURL_LIBRARY})
endif ()

target_include_directories(${TARGET} PUBLIC .)
target_compile_features   (${TARGET} PUBLIC cxx_std_11)
target_link_libraries     (${TARGET} PRIVATE ${LLAMA_COMMON_EXTRA_LIBS} PUBLIC llama Threads::Threads)
