# CMakeLists.txt for llama.cpp C++ Learning Examples
#
# This CMakeLists.txt builds the learning examples for llama.cpp.
# It can be used in two ways:
#
# 1. As part of the main llama.cpp build:
#    - The examples will link against the main llama library
#    - Build from the main llama.cpp directory
#
# 2. As a standalone build (assumes llama.cpp is already built):
#    - Set LLAMA_CPP_DIR to point to your llama.cpp installation
#    - mkdir build && cd build
#    - cmake .. -DLLAMA_CPP_DIR=/path/to/llama.cpp
#    - cmake --build .

cmake_minimum_required(VERSION 3.14)
project(llama_cpp_learning_examples CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# =============================================================================
# Find llama.cpp library
# =============================================================================

# Check if we're building as part of llama.cpp
if(TARGET llama)
    message(STATUS "Building as part of llama.cpp project")
    set(LLAMA_LIBRARY llama)
    set(LLAMA_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
else()
    # Standalone build - need to find llama.cpp
    message(STATUS "Building as standalone project")

    # Allow user to specify llama.cpp directory
    set(LLAMA_CPP_DIR "" CACHE PATH "Path to llama.cpp installation")

    if(LLAMA_CPP_DIR)
        message(STATUS "Using LLAMA_CPP_DIR: ${LLAMA_CPP_DIR}")
        set(LLAMA_INCLUDE_DIR ${LLAMA_CPP_DIR}/include)

        # Try to find the library
        find_library(LLAMA_LIBRARY
            NAMES llama libllama
            PATHS ${LLAMA_CPP_DIR}/build ${LLAMA_CPP_DIR}/build/src ${LLAMA_CPP_DIR}
            PATH_SUFFIXES lib lib64
            NO_DEFAULT_PATH
        )

        if(NOT LLAMA_LIBRARY)
            message(FATAL_ERROR "Could not find llama library in ${LLAMA_CPP_DIR}. "
                                "Please build llama.cpp first or set LLAMA_CPP_DIR correctly.")
        endif()
    else()
        message(FATAL_ERROR "Please set LLAMA_CPP_DIR to point to your llama.cpp installation. "
                            "Example: cmake .. -DLLAMA_CPP_DIR=/path/to/llama.cpp")
    endif()

    message(STATUS "Found llama library: ${LLAMA_LIBRARY}")
    message(STATUS "Using include directory: ${LLAMA_INCLUDE_DIR}")
endif()

# =============================================================================
# Find dependencies
# =============================================================================

find_package(Threads REQUIRED)

# =============================================================================
# Helper function to add examples
# =============================================================================

function(add_learning_example target source)
    add_executable(${target} ${source})

    target_include_directories(${target} PRIVATE ${LLAMA_INCLUDE_DIR})

    target_link_libraries(${target} PRIVATE
        ${LLAMA_LIBRARY}
        ${CMAKE_THREAD_LIBS_INIT}
    )

    # Enable C++17 features
    target_compile_features(${target} PRIVATE cxx_std_17)

    # Add compiler warnings
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic
            -Wno-unused-parameter
        )
    elseif(MSVC)
        target_compile_options(${target} PRIVATE
            /W4
        )
    endif()

    # Installation
    install(TARGETS ${target}
            RUNTIME DESTINATION bin
            COMPONENT examples)

    message(STATUS "Added learning example: ${target}")
endfunction()

# =============================================================================
# Add examples
# =============================================================================

add_learning_example(01-simple-inference 01-simple-inference.cpp)
add_learning_example(02-embeddings 02-embeddings.cpp)
add_learning_example(03-custom-sampling 03-custom-sampling.cpp)

# =============================================================================
# Print build information
# =============================================================================

message(STATUS "")
message(STATUS "=== llama.cpp Learning Examples Build Configuration ===")
message(STATUS "  CMAKE_BUILD_TYPE:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_COMPILER:  ${CMAKE_CXX_COMPILER}")
message(STATUS "  CMAKE_CXX_STANDARD:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  LLAMA_LIBRARY:       ${LLAMA_LIBRARY}")
message(STATUS "  LLAMA_INCLUDE_DIR:   ${LLAMA_INCLUDE_DIR}")
message(STATUS "")
message(STATUS "Examples built:")
message(STATUS "  - 01-simple-inference: Basic text generation")
message(STATUS "  - 02-embeddings:       Embedding generation for RAG")
message(STATUS "  - 03-custom-sampling:  Custom sampling strategies")
message(STATUS "")
message(STATUS "After building, run examples from the build directory:")
message(STATUS "  ./01-simple-inference -m /path/to/model.gguf -p \"Hello\"")
message(STATUS "  ./02-embeddings -m /path/to/model.gguf -t \"Text 1\" \"Text 2\"")
message(STATUS "  ./03-custom-sampling -m /path/to/model.gguf -s creative")
message(STATUS "======================================================")
message(STATUS "")
