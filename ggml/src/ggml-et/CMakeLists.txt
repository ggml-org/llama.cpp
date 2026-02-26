
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

# --- Device kernels (cross-compiled via ExternalProject) ---
include(ExternalProject)

set(KERNEL_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/et-kernels)
set(KERNEL_CMAKE_ARGS "")
if(DEFINED ADDRESS)
    list(APPEND KERNEL_CMAKE_ARGS -DADDRESS=${ADDRESS})
endif()


ExternalProject_Add(et-kernels-build
    SOURCE_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/et-kernels
    BINARY_DIR  ${KERNEL_BINARY_DIR}
    CMAKE_ARGS  -DCMAKE_TOOLCHAIN_FILE=${ET_PLATFORM_PATH}/lib/cmake/riscv64-ec-toolchain.cmake
                -DCMAKE_PREFIX_PATH=${ET_PLATFORM_PATH}
                -DCMAKE_MODULE_PATH=${ET_PLATFORM_PATH}/lib/cmake/cmake-modules
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_INSTALL_LIBDIR=lib
                -DTOOLCHAIN_DIR=${ET_PLATFORM_PATH}
                ${KERNEL_CMAKE_ARGS}
    INSTALL_COMMAND ""
    BUILD_ALWAYS TRUE
)

set(EMBED_HPP ${KERNEL_BINARY_DIR}/ggml-et-kernels-embed.hpp)
set(EMBED_CPP ${KERNEL_BINARY_DIR}/ggml-et-kernels-embed.cpp)

ggml_add_backend_library(ggml-et
                         ggml-et.cpp
                         ggml-et-kernels.cpp
                         ggml-et-memops.cpp
                         ggml-et-ops.cpp
                         ggml-et-cpu-compare.cpp
                        )

# Mark generated files as such
set_source_files_properties(
    ${EMBED_CPP}
    ${EMBED_HPP}
    PROPERTIES GENERATED TRUE
)

# Add embedded kernel sources
target_sources(ggml-et PRIVATE
    ${EMBED_CPP}
    ${EMBED_HPP}
)

# Include directory for embedded headers
target_include_directories(ggml-et PRIVATE ${KERNEL_BINARY_DIR})

target_link_libraries(ggml-et PRIVATE runtime::etrt_static deviceLayer::deviceLayer)

# Ensure kernels are built and embedded before the backend library
add_dependencies(ggml-et et-kernels-build)
