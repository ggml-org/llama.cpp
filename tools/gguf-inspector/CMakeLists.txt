# gguf-inspector — annotated GGUF hex viewer (Qt6)
#
# This tool is self-contained: it parses GGUF bytes directly and does not
# link against llama/common/ggml. It can be built standalone
# (cmake this directory directly) or as part of the llama.cpp tree by
# uncommenting the add_subdirectory in tools/CMakeLists.txt.

cmake_minimum_required(VERSION 3.16)
project(gguf-inspector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 6.2 COMPONENTS Widgets REQUIRED)

add_executable(gguf-inspector gguf_inspector.cpp)
target_link_libraries(gguf-inspector PRIVATE Qt6::Widgets)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS gguf-inspector RUNTIME)
endif()
