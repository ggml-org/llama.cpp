set(BUILD_NUMBER 0)
set(BUILD_COMMIT "unknown")
set(BUILD_COMPILER "unknown")
set(BUILD_TARGET "unknown")

# The source tree this file belongs to. Derived from the file's own location rather than
# from CMAKE_CURRENT_SOURCE_DIR, which is the invoking working directory when this file is
# reached through a `cmake -P` script (scripts/ui-assets.cmake does exactly that).
get_filename_component(BUILD_INFO_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." REALPATH)

# Look for git
find_package(Git)
if(NOT Git_FOUND)
    find_program(GIT_EXECUTABLE NAMES git git.exe)
    if(GIT_EXECUTABLE)
        set(Git_FOUND TRUE)
        message(STATUS "Found Git: ${GIT_EXECUTABLE}")
    else()
        message(WARNING "Git not found. Build info will not be accurate.")
    endif()
endif()

# git searches parent directories for a repository, so a source tree with none of its own
# (a release tarball, or vendored sources) picks up whatever repository sits above it and
# reports that one's HEAD as llama.cpp's. Only trust git when the repository it finds is
# this source tree.
set(BUILD_INFO_GIT_USABLE FALSE)
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --show-toplevel
        WORKING_DIRECTORY "${BUILD_INFO_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_TOPLEVEL
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE RES
    )
    if (RES EQUAL 0 AND NOT GIT_TOPLEVEL STREQUAL "")
        get_filename_component(GIT_TOPLEVEL "${GIT_TOPLEVEL}" REALPATH)
        if (GIT_TOPLEVEL STREQUAL BUILD_INFO_SOURCE_DIR)
            set(BUILD_INFO_GIT_USABLE TRUE)
        else()
            message(STATUS "Git repository at ${GIT_TOPLEVEL} is not the llama.cpp source tree "
                           "(${BUILD_INFO_SOURCE_DIR}). Build info will not be accurate.")
        endif()
    endif()
endif()

# Get the commit count and hash
if(BUILD_INFO_GIT_USABLE)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY "${BUILD_INFO_SOURCE_DIR}"
        OUTPUT_VARIABLE HEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(BUILD_COMMIT ${HEAD})
    endif()
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
        WORKING_DIRECTORY "${BUILD_INFO_SOURCE_DIR}"
        OUTPUT_VARIABLE COUNT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE RES
    )
    if (RES EQUAL 0)
        set(BUILD_NUMBER ${COUNT})
    endif()
endif()

set(BUILD_COMPILER "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

if(CMAKE_VS_PLATFORM_NAME)
    set(BUILD_TARGET ${CMAKE_VS_PLATFORM_NAME})
else()
    set(BUILD_TARGET "${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
endif()
