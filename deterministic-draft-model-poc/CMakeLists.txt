# deterministic-draft-model-poc/CMakeLists.txt
#
# Standalone proof-of-concept plugin project. Simulates an external consumer
# (e.g., a law firm, government agency, or commercial vendor) building a
# deterministic draft plugin against the distributed SDK artifacts.
#
# This project has NO dependency on the llama.cpp build tree. It expects the
# SDK artifacts to be available in lib/ (copied or symlinked from external/):
#   lib/deterministic_draft_plugin.h   -- the C API contract
#   lib/libdeterministic_draft_spec.so -- the spec shared library (for testing)
#
# Build steps:
#   1. Build llama.cpp with -DDETERMINISTIC_SPEC_ENABLED=ON
#      This produces external/lib/libdeterministic_draft_spec.so and
#      external/include/deterministic_draft_plugin.h
#
#   2. Copy/symlink the SDK artifacts into this project's lib/ directory:
#      ln -sf ../external/lib/libdeterministic_draft_spec.so deterministic-draft-model-poc/lib/
#      ln -sf ../external/include/deterministic_draft_plugin.h deterministic-draft-model-poc/lib/
#
#   3. Build this project:
#      cmake -B build -S deterministic-draft-model-poc
#      cmake --build build -j$(nproc)
#
#   4. Use the resulting plugin:
#      ./build/bin/benchmark-deterministic-draft \
#          --deterministic-draft-model ./deterministic-draft-model-poc/build/deterministic-draft.so

cmake_minimum_required(VERSION 3.14)
project(deterministic-draft-poc LANGUAGES C CXX)

# ── SDK artifacts ───────────────────────────────────────────────────
# The plugin API header must be available in lib/
set(PLUGIN_API_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/lib/deterministic_draft_plugin.h"
    CACHE FILEPATH "Path to deterministic_draft_plugin.h")

if(NOT EXISTS "${PLUGIN_API_HEADER}")
    message(FATAL_ERROR
        "deterministic_draft_plugin.h not found in lib/.\n"
        "Build llama.cpp with -DDETERMINISTIC_SPEC_ENABLED=ON and run link-sdk.sh:\n"
        "  cmake -B build -DGGML_CUDA=OFF -DGGML_METAL=OFF -DDETERMINISTIC_SPEC_ENABLED=ON\n"
        "  cmake --build build --target deterministic_draft_spec -j\$(nproc)\n"
        "  ./deterministic-draft-model-poc/link-sdk.sh")
endif()

message(STATUS "PoC plugin API header: ${PLUGIN_API_HEADER}")

# ── Find XGrammar ───────────────────────────────────────────────────
# XGrammar is fetched and built as part of this project
include(FetchContent)

# Disable Python bindings and tests before fetching
set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Build Python bindings" FORCE)
set(XGRAMMAR_BUILD_TVM_FFI OFF CACHE BOOL "Build TVM FFI" FORCE)
set(XGRAMMAR_BUILD_TESTS OFF CACHE BOOL "Build C++ tests" FORCE)
set(XGRAMMAR_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(XGRAMMAR_BUILD_TVM_FFI OFF CACHE BOOL "Build TVM FFI bindings" FORCE)

FetchContent_Declare(
    xgrammar
    GIT_REPOSITORY https://github.com/mlc-ai/xgrammar.git
    GIT_TAG        v0.2.3
    GIT_SHALLOW    TRUE
    CMAKE_ARGS
        -DXGRAMMAR_BUILD_PYTHON_BINDINGS=OFF
        -DXGRAMMAR_BUILD_TVM_FFI=OFF
        -DXGRAMMAR_BUILD_TESTS=OFF
        -DXGRAMMAR_BUILD_EXAMPLES=OFF
)

FetchContent_Populate(xgrammar)

# Patch XGrammar to disable Python bindings BEFORE configuring
file(READ "${xgrammar_SOURCE_DIR}/CMakeLists.txt" XGRAMMAR_CMAKE_CONTENT)
string(REPLACE "option(XGRAMMAR_BUILD_PYTHON_BINDINGS \"Build Python bindings\" ON)"
               "option(XGRAMMAR_BUILD_PYTHON_BINDINGS \"Build Python bindings\" OFF)"
               XGRAMMAR_CMAKE_CONTENT "${XGRAMMAR_CMAKE_CONTENT}")
file(WRITE "${xgrammar_SOURCE_DIR}/CMakeLists.txt" "${XGRAMMAR_CMAKE_CONTENT}")

# Also patch config.cmake
file(READ "${xgrammar_SOURCE_DIR}/cmake/config.cmake" XGRAMMAR_CONFIG_CONTENT)
string(REPLACE "set(XGRAMMAR_BUILD_PYTHON_BINDINGS ON)"
               "set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF)"
               XGRAMMAR_CONFIG_CONTENT "${XGRAMMAR_CONFIG_CONTENT}")
file(WRITE "${xgrammar_SOURCE_DIR}/cmake/config.cmake" "${XGRAMMAR_CONFIG_CONTENT}")

add_subdirectory(${xgrammar_SOURCE_DIR} ${xgrammar_BINARY_DIR})

message(STATUS "XGrammar configured from git")

# ── Build the plugin .so ────────────────────────────────────────────
add_library(deterministic-draft MODULE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/plugin.cpp"
)

target_include_directories(deterministic-draft PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/lib"
    "${xgrammar_SOURCE_DIR}/include"
)

target_link_libraries(deterministic-draft PRIVATE xgrammar ${CMAKE_DL_LIBS})

target_compile_features(deterministic-draft PRIVATE cxx_std_17)
set_target_properties(deterministic-draft PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
)
target_compile_definitions(deterministic-draft PRIVATE DETERMINISTIC_DRAFT_VERSION="3.0.0")

# Bundle the grammar files alongside the built plugin, so deterministic_draft_set_language()
# can find them at "<plugin_dir>/grammars/<lang>.gbnf" without any host configuration.
# Grammars remain plain data files here - editable/addable without rebuilding the plugin.
file(GLOB GRAMMAR_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/grammars/*.gbnf")
set(GRAMMAR_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/grammars")
set(GRAMMAR_OUTPUTS "")
foreach(grammar ${GRAMMAR_FILES})
    get_filename_component(grammar_name ${grammar} NAME)
    set(grammar_output "${GRAMMAR_OUTPUT_DIR}/${grammar_name}")
    list(APPEND GRAMMAR_OUTPUTS ${grammar_output})
    add_custom_command(OUTPUT ${grammar_output}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${GRAMMAR_OUTPUT_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${grammar}" "${grammar_output}"
        DEPENDS "${grammar}"
        COMMENT "Copying grammar ${grammar_name}"
    )
endforeach()
add_custom_target(deterministic-draft-grammars ALL DEPENDS ${GRAMMAR_OUTPUTS})
add_dependencies(deterministic-draft deterministic-draft-grammars)

message(STATUS "PoC plugin -> ${CMAKE_BINARY_DIR}/deterministic-draft.so")

# Grammar acceptance tests
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    enable_testing()
    add_test(
        NAME grammar-acceptance-tests
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/test_file_based.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests"
            "$<TARGET_FILE:deterministic-draft>"
    )
    set_tests_properties(grammar-acceptance-tests PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()

