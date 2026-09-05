set(HRX_SOURCE_DIR "" CACHE PATH "Optional HRX source tree to build instead of using installed hrx and loomc packages")

if(HRX_SOURCE_DIR)
    include(ExternalProject)
    include(GNUInstallDirs)

    get_filename_component(HRX_SOURCE_DIR "${HRX_SOURCE_DIR}" ABSOLUTE)
    if(NOT EXISTS "${HRX_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "HRX_SOURCE_DIR does not contain a CMakeLists.txt: ${HRX_SOURCE_DIR}")
    endif()

    set(GGML_HRX_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/hrx")
    set(GGML_HRX_BUILD_DIR "${GGML_HRX_PREFIX}/src/ggml-hrx-deps-build")
    set(GGML_HRX_LIB "${GGML_HRX_BUILD_DIR}/libhrx/src/libhrx/${CMAKE_SHARED_LIBRARY_PREFIX}hrx${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(GGML_LOOMC_LIB "${GGML_HRX_BUILD_DIR}/loom/binding/c/${CMAKE_SHARED_LIBRARY_PREFIX}loomc${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(GGML_HRX_LOOM_LINK "${GGML_HRX_BUILD_DIR}/loom/src/loom/tools/loom-link/loom-link${CMAKE_EXECUTABLE_SUFFIX}")
    set(GGML_HRX_LOOM_FORMAT "${GGML_HRX_BUILD_DIR}/loom/src/loom/tools/loom-format/loom-format${CMAKE_EXECUTABLE_SUFFIX}")
    set(GGML_HRX_DEPS_TARGET ggml-hrx-deps)

    set(GGML_HRX_CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DIREE_BUILD_TESTS=OFF
        -DIREE_BUILD_BENCHMARKS=OFF
        -DIREE_HAL_DRIVER_DEFAULTS=OFF
        -DIREE_HAL_DRIVER_AMDGPU=ON
        -DIREE_HAL_DRIVER_LOCAL_SYNC=ON
        -DIREE_HAL_DRIVER_LOCAL_TASK=ON
        -DIREE_HAL_DRIVER_NULL=ON
        -DLIBHRX_BUILD_CTS=OFF
    )
    if(IREE_ROCM_PATH)
        list(APPEND GGML_HRX_CMAKE_ARGS -DIREE_ROCM_PATH=${IREE_ROCM_PATH})
    endif()
    if(FETCHCONTENT_BASE_DIR)
        list(APPEND GGML_HRX_CMAKE_ARGS -DFETCHCONTENT_BASE_DIR=${FETCHCONTENT_BASE_DIR})
    endif()

    ExternalProject_Add(ggml-hrx-deps
        SOURCE_DIR "${HRX_SOURCE_DIR}"
        PREFIX "${GGML_HRX_PREFIX}"
        CMAKE_ARGS ${GGML_HRX_CMAKE_ARGS}
        BUILD_COMMAND ${CMAKE_COMMAND} --build . --target hrx loomc_shared loom_tools_loom-link_loom-link loom_tools_loom-format_loom-format --config ${CMAKE_BUILD_TYPE}
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS "${GGML_HRX_LIB}" "${GGML_LOOMC_LIB}" "${GGML_HRX_LOOM_LINK}" "${GGML_HRX_LOOM_FORMAT}"
        UPDATE_COMMAND ""
    )

    add_library(hrx::hrx SHARED IMPORTED GLOBAL)
    set_target_properties(hrx::hrx PROPERTIES
        IMPORTED_LOCATION "${GGML_HRX_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${HRX_SOURCE_DIR}/libhrx/include")
    add_dependencies(hrx::hrx ggml-hrx-deps)

    add_library(loomc::loomc SHARED IMPORTED GLOBAL)
    set_target_properties(loomc::loomc PROPERTIES
        IMPORTED_LOCATION "${GGML_LOOMC_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${HRX_SOURCE_DIR}/loom/binding/c/include"
        INTERFACE_COMPILE_DEFINITIONS LOOMC_USING_SHARED_LIBRARY)
    add_dependencies(loomc::loomc ggml-hrx-deps)
else()
    find_package(hrx CONFIG REQUIRED)
    find_package(loomc CONFIG REQUIRED)
    find_program(GGML_HRX_LOOM_LINK NAMES loom-link)
    find_program(GGML_HRX_LOOM_FORMAT NAMES loom-format)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(GGML_HRX_KERNEL_CORPUS_SOURCE_FORMAT "binary" CACHE STRING "Embedded Loom corpus source format: text or binary")
set_property(CACHE GGML_HRX_KERNEL_CORPUS_SOURCE_FORMAT PROPERTY STRINGS text binary)
if(GGML_HRX_KERNEL_CORPUS_SOURCE_FORMAT STREQUAL "binary")
    if(NOT GGML_HRX_LOOM_LINK OR NOT GGML_HRX_LOOM_FORMAT)
        message(FATAL_ERROR "GGML_HRX_KERNEL_CORPUS_SOURCE_FORMAT=binary requires loom-link and loom-format")
    endif()
    set(GGML_HRX_KERNEL_CORPUS_TOOL_ARGS
        --loom-link "${GGML_HRX_LOOM_LINK}"
        --loom-format "${GGML_HRX_LOOM_FORMAT}"
    )
    set(GGML_HRX_KERNEL_CORPUS_TOOL_DEPENDS
        "${GGML_HRX_LOOM_LINK}"
        "${GGML_HRX_LOOM_FORMAT}"
    )
elseif(NOT GGML_HRX_KERNEL_CORPUS_SOURCE_FORMAT STREQUAL "text")
    message(FATAL_ERROR "Unsupported GGML_HRX_KERNEL_CORPUS_SOURCE_FORMAT: ${GGML_HRX_KERNEL_CORPUS_SOURCE_FORMAT}")
endif()

set(GGML_HRX_QWEN_KERNEL_CORPUS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernel-corpus/kernels/qwen_moe")
set(GGML_HRX_QWEN_KERNEL_CORPUS_MANIFEST "${GGML_HRX_QWEN_KERNEL_CORPUS_DIR}/manifest.json")
set(GGML_HRX_KERNEL_CORPUS_SOURCES_INC "${CMAKE_CURRENT_BINARY_DIR}/kernel-corpus-sources.inc")
set(GGML_HRX_KERNEL_CORPUS_QWEN_INC "${CMAKE_CURRENT_BINARY_DIR}/kernel-corpus-qwen.inc")
set(GGML_HRX_KERNEL_CORPUS_CATALOG_INC "${CMAKE_CURRENT_BINARY_DIR}/kernel-corpus-catalog.inc")
set(GGML_HRX_KERNEL_CORPUS_DEPFILE "${CMAKE_CURRENT_BINARY_DIR}/kernel-corpus.d")

add_custom_command(
    OUTPUT
        "${GGML_HRX_KERNEL_CORPUS_SOURCES_INC}"
        "${GGML_HRX_KERNEL_CORPUS_QWEN_INC}"
        "${GGML_HRX_KERNEL_CORPUS_CATALOG_INC}"
    COMMAND ${Python3_EXECUTABLE}
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_kernel_corpus.py"
        --source-output "${GGML_HRX_KERNEL_CORPUS_SOURCES_INC}"
        --corpus-output "${GGML_HRX_KERNEL_CORPUS_QWEN_INC}"
        --catalog-output "${GGML_HRX_KERNEL_CORPUS_CATALOG_INC}"
        --manifest "${GGML_HRX_QWEN_KERNEL_CORPUS_MANIFEST}"
        --corpus-dir "${GGML_HRX_QWEN_KERNEL_CORPUS_DIR}"
        --source-format "${GGML_HRX_KERNEL_CORPUS_SOURCE_FORMAT}"
        ${GGML_HRX_KERNEL_CORPUS_TOOL_ARGS}
        --depfile "${GGML_HRX_KERNEL_CORPUS_DEPFILE}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_kernel_corpus.py"
        "${GGML_HRX_QWEN_KERNEL_CORPUS_MANIFEST}"
        ${GGML_HRX_KERNEL_CORPUS_TOOL_DEPENDS}
    DEPFILE "${GGML_HRX_KERNEL_CORPUS_DEPFILE}"
    VERBATIM
)

option(GGML_HRX_BUNDLE_RUNTIME_LIBS "Bundle HRX/ROCm runtime libraries next to the HRX backend" OFF)
set(GGML_HRX_BUNDLE_LIBRARY_DIRS "" CACHE STRING "Library directories to scan when GGML_HRX_BUNDLE_RUNTIME_LIBS=ON")

add_library(ggml-hrx-kernel-corpus STATIC
    status.h
    kernel-corpus/kernel-corpus-json.cpp
    kernel-corpus/kernel-corpus-json.h
    kernel-corpus/kernel-corpus-catalog-verify.h
    kernel-corpus/kernel-corpus-catalog.h
    kernel-corpus/kernel-corpus.cpp
    kernel-corpus/kernel-corpus.h
    kernel-corpus/kernel-types.h
    "${GGML_HRX_KERNEL_CORPUS_SOURCES_INC}"
    "${GGML_HRX_KERNEL_CORPUS_QWEN_INC}"
    "${GGML_HRX_KERNEL_CORPUS_CATALOG_INC}"
)
if(GGML_HRX_DEPS_TARGET)
    add_dependencies(ggml-hrx-kernel-corpus ${GGML_HRX_DEPS_TARGET})
endif()
target_include_directories(ggml-hrx-kernel-corpus PUBLIC . PRIVATE "${CMAKE_CURRENT_BINARY_DIR}" ../../../vendor)
target_compile_features(ggml-hrx-kernel-corpus PRIVATE cxx_std_17)
set_target_properties(ggml-hrx-kernel-corpus PROPERTIES POSITION_INDEPENDENT_CODE ON)

ggml_add_backend_library(ggml-hrx
    backend-buffer-binding.cpp
    backend-buffer-binding.h
    backend-context.h
    dispatch/command-plan-metadata.cpp
    dispatch/command-plan-metadata.h
    dispatch/command-plan.h
    dispatch/command-program-bindings.cpp
    dispatch/command-program-bindings.h
    dispatch/command-program-diagnostics.cpp
    dispatch/command-program-diagnostics.h
    dispatch/command-program.cpp
    dispatch/command-program.h
    dispatch/command-program-resolver.cpp
    dispatch/command-program-resolver.h
    dispatch/dispatch-scheduler.cpp
    dispatch/dispatch-scheduler.h
    dispatch/dispatch.h
    dispatch/transient-allocator.cpp
    dispatch/transient-allocator.h
    dispatch_registration/dispatch-add.cpp
    dispatch_registration/dispatch-add.h
    dispatch_registration/dispatch-gather-add.cpp
    dispatch_registration/dispatch-gather-add.h
    dispatch_registration/dispatch-llm-matmul.cpp
    dispatch_registration/dispatch-llm-matmul.h
    dispatch_registration/dispatch-llm-profiles.h
    dispatch_registration/dispatch-llm-shapes.h
    dispatch_registration/dispatch-qwen-attention-postprocess.cpp
    dispatch_registration/dispatch-qwen-attention-postprocess.h
    dispatch_registration/dispatch-qwen-flash-attention.cpp
    dispatch_registration/dispatch-qwen-flash-attention.h
    dispatch_registration/dispatch-qwen-matmul.cpp
    dispatch_registration/dispatch-qwen-matmul.h
    dispatch_registration/dispatch-moe-router.cpp
    dispatch_registration/dispatch-moe-router.h
    dispatch_registration/dispatch-qwen-preamble.cpp
    dispatch_registration/dispatch-qwen-preamble.h
    dispatch_registration/dispatch-registry.cpp
    dispatch_registration/dispatch-registry.h
    dispatch_registration/dispatch-rmsnorm.cpp
    dispatch_registration/dispatch-rmsnorm.h
    dispatch_registration/dispatch-routed-ffn.cpp
    dispatch_registration/dispatch-routed-ffn.h
    status.h
    graph/graph.cpp
    graph/graph-diagnostics.cpp
    graph/graph-diagnostics.h
    graph/graph.h
    graph/graph-matcher.cpp
    graph/graph-matcher.h
    graph/graph-traversal.cpp
    graph/graph-traversal.h
    graph/op-params.cpp
    graph/op-params.h
    ggml-hrx.cpp
    loom-jit.cpp
    graph/value-map.cpp
    graph/value-map.h
    runtime/command-program-executor.cpp
    runtime/command-program-executor.h
    runtime/graph-executor.cpp
    runtime/graph-executor.h
    runtime/graph-program-cache.cpp
    runtime/graph-program-cache.h
    runtime/host-memory.cpp
    runtime/host-memory.h
    runtime/kernel-executable-cache.cpp
    runtime/kernel-executable-cache.h
    runtime/loom-kernel-jit.cpp
    runtime/loom-kernel-jit.h
    runtime/prepared-command-program-cache.cpp
    runtime/prepared-command-program-cache.h
    runtime/transient-arena.cpp
    runtime/transient-arena.h
    runtime/host-buffer-registry.cpp
    runtime/host-buffer-registry.h
)
target_link_libraries(ggml-hrx PRIVATE ggml-hrx-kernel-corpus hrx::hrx loomc::loomc)
target_include_directories(ggml-hrx PRIVATE . "${CMAKE_CURRENT_BINARY_DIR}" ../../../vendor)
target_compile_definitions(ggml-hrx PRIVATE GGML_USE_HRX)

if (GGML_HRX_BUNDLE_RUNTIME_LIBS)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/BundleRuntime.cmake")
    ggml_hrx_bundle_runtime(ggml-hrx)
endif()

add_executable(ggml-hrx-compile-kernel
    tools/compile-kernel.cpp
    tools/tool-utils.h
    loom-jit.cpp
)
target_link_libraries(ggml-hrx-compile-kernel PRIVATE hrx::hrx loomc::loomc)
target_include_directories(ggml-hrx-compile-kernel PRIVATE .)
target_compile_features(ggml-hrx-compile-kernel PRIVATE cxx_std_17)

add_executable(ggml-hrx-analyze-graph
    tools/analyze-graph.cpp
)
target_link_libraries(ggml-hrx-analyze-graph PRIVATE ggml-hrx ggml ggml-hrx-kernel-corpus)
target_include_directories(ggml-hrx-analyze-graph PRIVATE . ../../../vendor)
target_compile_features(ggml-hrx-analyze-graph PRIVATE cxx_std_17)
