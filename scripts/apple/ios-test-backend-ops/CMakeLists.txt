# Minimal iOS app that runs tests/test-backend-ops.cpp on a device (iOS apps cannot run command-line tools).
# Used by run.sh; builds only ggml (no llama), with the Metal and ANE backends.
cmake_minimum_required(VERSION 3.21)
project(ggml-ios-test-backend-ops LANGUAGES C CXX OBJC)

set(LLAMA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

set(BUILD_SHARED_LIBS        OFF CACHE BOOL "" FORCE)
set(GGML_METAL               ON  CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON  CACHE BOOL "" FORCE)
set(GGML_BLAS                OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP              OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE              OFF CACHE BOOL "" FORCE)
add_subdirectory(${LLAMA_ROOT}/ggml ggml)

set(BUNDLE_ID "com.example.ggml-test-backend-ops" CACHE STRING "bundle identifier of the test app")

add_executable(test-backend-ops-app MACOSX_BUNDLE main.m shim.cpp ${LLAMA_ROOT}/tests/test-backend-ops.cpp)
# the test's main() is called from the app
set_source_files_properties(${LLAMA_ROOT}/tests/test-backend-ops.cpp PROPERTIES COMPILE_DEFINITIONS main=test_backend_ops_main)
target_compile_features(test-backend-ops-app PRIVATE cxx_std_17)
target_link_libraries(test-backend-ops-app PRIVATE ggml "-framework UIKit")
set_target_properties(test-backend-ops-app PROPERTIES
    OUTPUT_NAME                                 ggml-test-backend-ops
    MACOSX_BUNDLE_GUI_IDENTIFIER                ${BUNDLE_ID}
    MACOSX_BUNDLE_BUNDLE_NAME                   ggml-test-backend-ops
    MACOSX_BUNDLE_INFO_PLIST                    ${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.in
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER   ${BUNDLE_ID}
    XCODE_ATTRIBUTE_CODE_SIGN_STYLE             Automatic
    XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY      "1,2"
)
