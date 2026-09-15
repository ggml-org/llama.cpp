cmake_minimum_required(VERSION 3.14)
project(llama-simple)

set(CMAKE_CXX_STANDARD 17)

option(LLAMA_TEST_USE_SUBDIR "Use add_subdirectory instead of find_package" OFF)

if(LLAMA_TEST_USE_SUBDIR)
    add_subdirectory(../../ llama.cpp)
else()
    find_package(llama 0.1.0 REQUIRED)
endif()

add_executable(test-cmake test-cmake.cpp)
target_link_libraries(test-cmake PRIVATE llama)

if(DEFINED LLAMA_BUILD_NUMBER)
    target_compile_definitions(test-cmake PRIVATE
        LLAMA_BUILD_NUMBER=${LLAMA_BUILD_NUMBER}
        LLAMA_BUILD_COMMIT="${LLAMA_BUILD_COMMIT}"
    )
endif()
