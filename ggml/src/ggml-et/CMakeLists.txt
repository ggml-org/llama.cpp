message(STATUS "Using ET backend")

# Configure ET platform path
if(DEFINED ENV{ET_PLATFORM})
    set(ET_PLATFORM_ROOT $ENV{ET_PLATFORM})
else()
    set(ET_PLATFORM_ROOT /opt/et)
endif()

# Add ET platform CMake modules and config files to search paths
list(APPEND CMAKE_MODULE_PATH "${ET_PLATFORM_ROOT}/share/cmake/Modules" "${ET_PLATFORM_ROOT}/lib/cmake")
list(APPEND CMAKE_PREFIX_PATH "${ET_PLATFORM_ROOT}")

message(STATUS "ET platform root: ${ET_PLATFORM_ROOT}")

# Find all dependencies with custom Find modules before runtime
# (runtime's find_dependency doesn't inherit CMAKE_MODULE_PATH)
find_package(linuxDriver REQUIRED)
find_package(libcap REQUIRED)
find_package(easy_profiler REQUIRED)

# Set libcap_FOUND so find_dependency can find it
set(libcap_FOUND TRUE)

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
