message(STATUS "Using ET backend")

find_package(runtime REQUIRED)

# Build ET kernels
add_subdirectory(et-kernels)

ggml_add_backend_library(ggml-et
                         ggml-et.cpp
                         ggml-et-kernels.cpp
                         ggml-et-memops.cpp
                         ggml-et-ops.cpp
                         ggml-et-cpu-compare.cpp
                        )

# Mark generated files as such
set_source_files_properties(
    ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-kernels-embed.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-kernels-embed.hpp
    PROPERTIES GENERATED TRUE
)

# Add embedded kernel sources
target_sources(ggml-et PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-kernels-embed.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/et-kernels/ggml-et-kernels-embed.hpp
)

# Include directory for embedded headers
target_include_directories(ggml-et PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/et-kernels)

target_link_libraries(ggml-et PRIVATE runtime::etrt_static deviceLayer::deviceLayer)

# Ensure kernels are built and embedded before the backend library
add_dependencies(ggml-et et-kernels-embed)
