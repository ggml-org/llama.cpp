# common

find_package(Threads REQUIRED)

llama_add_compile_flags()

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
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME} -DCMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/build-info-gen-cpp.cmake"
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
    arg.cpp
    arg.h
    base64.hpp
    chat-parser.cpp
    chat-parser.h
    chat.cpp
    chat.h
    common.cpp
    common.h
    console.cpp
    console.h
    json-partial.cpp
    json-partial.h
    json-schema-to-grammar.cpp
    llguidance.cpp
    log.cpp
    log.h
    ngram-cache.cpp
    ngram-cache.h
    regex-partial.cpp
    regex-partial.h
    sampling.cpp
    sampling.h
    speculative.cpp
    speculative.h
    )

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

set(LLAMA_COMMON_EXTRA_LIBS build_info)

# Use curl to download model url
if (LLAMA_CURL)
    find_package(CURL)
    if (NOT CURL_FOUND)
        message(FATAL_ERROR "Could NOT find CURL. Hint: to disable this feature, set -DLLAMA_CURL=OFF")
    endif()
    target_compile_definitions(${TARGET} PUBLIC LLAMA_USE_CURL)
    include_directories(${CURL_INCLUDE_DIRS})
    find_library(CURL_LIBRARY curl REQUIRED)
    set(LLAMA_COMMON_EXTRA_LIBS ${LLAMA_COMMON_EXTRA_LIBS} ${CURL_LIBRARY})
endif ()

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
        # v0.7.20 (+ fix to build on GCC 15):
        GIT_TAG b5b8b64dba11c4e4ee6b1d1450d3a3ae279891e8
        PREFIX ${CMAKE_BINARY_DIR}/llguidance
        SOURCE_DIR ${LLGUIDANCE_SRC}
        BUILD_IN_SOURCE TRUE
        CONFIGURE_COMMAND ""
        BUILD_COMMAND cargo build --release
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

target_include_directories(${TARGET} PUBLIC . ../vendor)
target_compile_features   (${TARGET} PUBLIC cxx_std_17)
target_link_libraries     (${TARGET} PRIVATE ${LLAMA_COMMON_EXTRA_LIBS} PUBLIC llama Threads::Threads)


#
# copy the license files
#

# Check if running in GitHub Actions
if (DEFINED ENV{GITHUB_ACTIONS} AND "$ENV{GITHUB_ACTIONS}" STREQUAL "true")
    message(STATUS "Running inside GitHub Actions - copying license files")

    # Copy all files from licenses/ to build/bin/
    file(GLOB LICENSE_FILES "${CMAKE_SOURCE_DIR}/licenses/*")
    foreach(LICENSE_FILE ${LICENSE_FILES})
        get_filename_component(FILENAME ${LICENSE_FILE} NAME)
        add_custom_command(
            POST_BUILD
            TARGET ${TARGET}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${LICENSE_FILE}"
                "$<TARGET_FILE_DIR:llama>/${FILENAME}"
            COMMENT "Copying ${FILENAME} to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
        message(STATUS "Copying ${LICENSE_FILE} to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${FILENAME}")
    endforeach()
endif()
