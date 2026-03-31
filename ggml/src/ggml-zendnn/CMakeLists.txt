ggml_add_backend_library(ggml-zendnn
                         ggml-zendnn.cpp)

if (NOT DEFINED ZENDNN_ROOT OR ZENDNN_ROOT STREQUAL "")
    set(ZENDNN_ROOT "$ENV{ZENDNN_ROOT}")
endif()

if (BUILD_SHARED_LIBS)
    set(ZENDNN_SHARED_LIB ON)
    set(ZENDNN_ARCHIVE_LIB OFF)
else()
    set(ZENDNN_SHARED_LIB OFF)
    set(ZENDNN_ARCHIVE_LIB ON)
endif()

# Download and build ZenDNN if not provided
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
        GIT_TAG a18adf8c605fb5f5e52cefd7eda08a7b18febbaf    # ZenDNN-2026-WW08
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
            -DZENDNNL_DEPENDS_FBGEMM=OFF
            -DZENDNNL_LIB_BUILD_ARCHIVE=${ZENDNN_ARCHIVE_LIB}
            -DZENDNNL_LIB_BUILD_SHARED=${ZENDNN_SHARED_LIB}
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

    add_dependencies(ggml-zendnn zendnn)
    set(ZENDNN_ROOT ${ZENDNN_INSTALL_DIR})
    message(STATUS "ZenDNN will be built to: ${ZENDNN_ROOT}")
else()
    message(STATUS "Using custom ZenDNN installation at: ${ZENDNN_ROOT}")
endif()

target_include_directories(ggml-zendnn PRIVATE
    ${ZENDNN_ROOT}/zendnnl/include
    ${ZENDNN_ROOT}/deps/json/include
    ${ZENDNN_ROOT}/deps/aoclutils/include
    ${ZENDNN_ROOT}/deps/aocldlp/include
    ${ZENDNN_ROOT}/deps/onednn/include
    ${ZENDNN_ROOT}/deps/libxsmm/include)

if (ZENDNN_SHARED_LIB)
    target_link_directories(ggml-zendnn PRIVATE ${ZENDNN_ROOT}/zendnnl/lib)
    target_link_libraries(ggml-zendnn PRIVATE zendnnl)
elseif (ZENDNN_ARCHIVE_LIB)
    target_link_libraries(ggml-zendnn PRIVATE
        ${ZENDNN_ROOT}/zendnnl/lib/libzendnnl_archive.a
        ${ZENDNN_ROOT}/deps/aoclutils/${CMAKE_INSTALL_LIBDIR}/libaoclutils.a
        ${ZENDNN_ROOT}/deps/aoclutils/${CMAKE_INSTALL_LIBDIR}/libau_cpuid.a
        ${ZENDNN_ROOT}/deps/aocldlp/lib/libaocl-dlp.a
        ${ZENDNN_ROOT}/deps/onednn/${CMAKE_INSTALL_LIBDIR}/libdnnl.a
        ${ZENDNN_ROOT}/deps/libxsmm/lib/libxsmm.a
        ${ZENDNN_ROOT}/deps/libxsmm/lib/libxsmmext.a
        ${ZENDNN_ROOT}/deps/libxsmm/lib/libxsmmnoblas.a)
endif()

target_link_libraries(ggml-zendnn PRIVATE m pthread)

if (GGML_OPENMP)
    target_link_libraries(ggml-zendnn PRIVATE OpenMP::OpenMP_CXX)
endif()
