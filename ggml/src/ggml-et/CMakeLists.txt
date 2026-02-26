
message(STATUS "Using ET backend")

# Configure ET platform path
if(DEFINED ENV{ET_PLATFORM})
    set(ET_PLATFORM_PATH $ENV{ET_PLATFORM})
else()
    set(ET_PLATFORM_PATH "/opt/et")
endif()

# Use sysemu for ET backend if compiled with `-DGGML_ET_SYSEMU=ON`
if(GGML_ET_SYSEMU)
    message(STATUS "Using ET backend with sysemu instead of hardware")
    add_compile_definitions(GGML_ET_SYSEMU=1)
else()
    message(STATUS "Using ET backend with hardware device")
endif()

# Add ET platform CMake modules and config files to search paths
list(APPEND CMAKE_PREFIX_PATH ${ET_PLATFORM_PATH}/lib/cmake)
list(APPEND CMAKE_MODULE_PATH ${ET_PLATFORM_PATH}/lib/cmake)
include(aifoundry-utils/ProjectFunctions)

message("Using ET Platform at ${ET_PLATFORM_PATH}")

find_package(runtime REQUIRED)

# Build ET kernels (cross-compiled in subdirectory scope)
add_subdirectory(et-kernels)

# --- Host backend library ---
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
