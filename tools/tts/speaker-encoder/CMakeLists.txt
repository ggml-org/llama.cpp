cmake_minimum_required(VERSION 3.14)

# When built standalone (outside llama.cpp tree), find ggml as an installed package.
# When built as part of llama.cpp, the targets already exist.
if(NOT TARGET ggml)
    project(speaker-encoder LANGUAGES C CXX)
    find_package(ggml REQUIRED)
endif()

add_executable(speaker-encoder speaker_encoder.cpp)
target_link_libraries(speaker-encoder PRIVATE ggml)
target_include_directories(speaker-encoder PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/include
)
install(TARGETS speaker-encoder RUNTIME)
