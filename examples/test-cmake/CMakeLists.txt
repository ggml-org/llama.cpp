cmake_minimum_required(VERSION 3.14)
project(llama-simple)

set(CMAKE_CXX_STANDARD 17)

find_package(llama 0.1.0 REQUIRED)

add_executable(test-cmake test-cmake.cpp)
target_link_libraries(test-cmake PRIVATE llama)
target_compile_definitions(test-cmake PRIVATE
    LLAMA_BUILD_NUMBER=${LLAMA_BUILD_NUMBER}
    LLAMA_BUILD_COMMIT="${LLAMA_BUILD_COMMIT}"
)
