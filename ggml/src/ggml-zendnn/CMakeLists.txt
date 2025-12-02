ggml_add_backend_library(ggml-zendnn
                         ggml-zendnn.cpp
                        )

# Get ZenDNN path
if (NOT DEFINED GGML_ZENDNN_PATH OR GGML_ZENDNN_PATH STREQUAL "")
    set(GGML_ZENDNN_PATH "$ENV{GGML_ZENDNN_PATH}")
endif()

# Check if path is still empty or OFF
if (NOT GGML_ZENDNN_PATH OR GGML_ZENDNN_PATH STREQUAL "" OR GGML_ZENDNN_PATH STREQUAL "OFF")
    message(STATUS "GGML_ZENDNN_PATH not set. Automatically downloading and building ZenDNN...")
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

    # Set GGML_ZENDNN_PATH to the installation directory
    set(GGML_ZENDNN_PATH ${ZENDNN_INSTALL_DIR})

    message(STATUS "ZenDNN will be built to: ${GGML_ZENDNN_PATH}")
else()
    message(STATUS "Using custom ZenDNN installation at: ${GGML_ZENDNN_PATH}")
endif()

# ZenDNN headers + libs
target_include_directories(ggml-zendnn PRIVATE
    ${GGML_ZENDNN_PATH}/zendnnl/include
    ${GGML_ZENDNN_PATH}/deps/aocldlp/include
    ${GGML_ZENDNN_PATH}/deps/aoclutils/include
    ${GGML_ZENDNN_PATH}/deps/json/include
    ${GGML_ZENDNN_PATH}/deps/libxsmm/include
)

target_link_directories(ggml-zendnn PRIVATE
    ${GGML_ZENDNN_PATH}/zendnnl/lib
    ${GGML_ZENDNN_PATH}/deps/aocldlp/lib
    ${GGML_ZENDNN_PATH}/deps/aoclutils/lib
    ${GGML_ZENDNN_PATH}/deps/libxsmm/lib
)

target_link_libraries(ggml-zendnn PRIVATE
    zendnnl_archive    # ZenDNN main
    aocl-dlp           # AOCL libraries
    aoclutils
    au_cpuid
    xsmm               # libxsmm small matrix math
    xsmmext
    xsmmnoblas
    xsmmf
    m                  # System math
    pthread            # Threading support
)

if (GGML_OPENMP)
    target_link_libraries(ggml-zendnn PRIVATE OpenMP::OpenMP_CXX)
endif()
