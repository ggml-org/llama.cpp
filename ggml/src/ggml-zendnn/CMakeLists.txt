ggml_add_backend_library(ggml-zendnn
                         ggml-zendnn.cpp)

# Get ZenDNN path
if (NOT DEFINED ZENDNN_ROOT OR ZENDNN_ROOT STREQUAL "")
    set(ZENDNN_ROOT "$ENV{ZENDNN_ROOT}")
endif()

# Check if path is still empty or OFF
if (NOT ZENDNN_ROOT OR ZENDNN_ROOT STREQUAL "" OR ZENDNN_ROOT STREQUAL "OFF")
    message(STATUS "ZENDNN_ROOT not set. Automatically downloading and building ZenDNN...")
    message(STATUS "This will take several minutes on first build...")

    include(ExternalProject)

    set(ZENDNN_PREFIX      ${CMAKE_BINARY_DIR}/_deps/zendnn-prefix)
    set(ZENDNN_SOURCE_DIR  ${ZENDNN_PREFIX}/src/zendnn)
    set(ZENDNN_BUILD_DIR   ${ZENDNN_PREFIX}/build)
    set(ZENDNN_INSTALL_DIR ${ZENDNN_BUILD_DIR}/install)

    ExternalProject_Add(
        zendnn
        GIT_REPOSITORY https://github.com/amd/ZenDNN.git
        GIT_TAG zendnnl
        PREFIX      ${ZENDNN_PREFIX}
        SOURCE_DIR  ${ZENDNN_SOURCE_DIR}
        BINARY_DIR  ${ZENDNN_BUILD_DIR}
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_PREFIX=${ZENDNN_INSTALL_DIR}
            -DZENDNNL_BUILD_EXAMPLES=OFF
            -DZENDNNL_BUILD_DOXYGEN=OFF
            -DZENDNNL_BUILD_GTEST=OFF
            -DZENDNNL_BUILD_BENCHDNN=OFF
            # Enable ALL matmul algorithm backends
            -DZENDNNL_DEPENDS_AOCLDLP=ON
            -DZENDNNL_DEPENDS_ONEDNN=ON
            -DZENDNNL_DEPENDS_LIBXSMM=ON
        BUILD_COMMAND   ${CMAKE_COMMAND} --build ${ZENDNN_BUILD_DIR} --target zendnnl
        INSTALL_COMMAND ${CMAKE_COMMAND} --build ${ZENDNN_BUILD_DIR} --target install
        BUILD_ALWAYS OFF
        LOG_DOWNLOAD ON
        LOG_CONFIGURE ON
        LOG_BUILD ON
        LOG_INSTALL ON
    )

    # Add dependency so ZenDNN builds before our library
    add_dependencies(ggml-zendnn zendnn)

    # Set ZENDNN_ROOT to the installation directory
    set(ZENDNN_ROOT ${ZENDNN_INSTALL_DIR})

    message(STATUS "ZenDNN will be built to: ${ZENDNN_ROOT}")
else()
    message(STATUS "Using custom ZenDNN installation at: ${ZENDNN_ROOT}")
endif()

# ZenDNN headers + libs
target_include_directories(ggml-zendnn PRIVATE
    ${ZENDNN_ROOT}/zendnnl/include
    ${ZENDNN_ROOT}/deps/aocldlp/include
    ${ZENDNN_ROOT}/deps/aoclutils/include
    ${ZENDNN_ROOT}/deps/json/include
    ${ZENDNN_ROOT}/deps/libxsmm/include
    ${ZENDNN_ROOT}/deps/onednn/include
)

target_link_directories(ggml-zendnn PRIVATE
    ${ZENDNN_ROOT}/zendnnl/lib
    ${ZENDNN_ROOT}/deps/aocldlp/lib
    ${ZENDNN_ROOT}/deps/aoclutils/lib
    ${ZENDNN_ROOT}/deps/libxsmm/lib
    ${ZENDNN_ROOT}/deps/onednn/lib
)

target_link_libraries(ggml-zendnn PRIVATE
    zendnnl_archive    # ZenDNN main
    aocl-dlp           # AOCL libraries
    aoclutils
    au_cpuid
    dnnl               # OneDNN
    xsmm               # libxsmm small matrix math
    xsmmext
    xsmmnoblas
    m
    pthread
)

if (GGML_OPENMP)
    target_link_libraries(ggml-zendnn PRIVATE OpenMP::OpenMP_CXX)
endif()
