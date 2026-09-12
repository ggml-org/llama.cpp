# Model-free checks are opt-in with the experimental backend.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND "${Python3_EXECUTABLE}" -B -c "import numpy"
    RESULT_VARIABLE Q4KP_NUMPY_RESULT OUTPUT_QUIET ERROR_QUIET)
if (NOT Q4KP_NUMPY_RESULT EQUAL 0)
    message(FATAL_ERROR "Q4KP tests require NumPy in Python3_EXECUTABLE")
endif()

set(Q4KP_DIR "${PROJECT_SOURCE_DIR}/ggml/src/ggml-cpu/q4kp")
set(Q4KP_FLAGS -mavx2 -mbmi2 -mfma -mf16c)
# The ctypes test library links the static backend on ELF platforms.
set_target_properties(ggml-cpu ggml-base PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(test-q4kp-kernels SHARED oracle.cpp
    ${Q4KP_DIR}/q4kp_kernel.cpp ${Q4KP_DIR}/q4kp_vnni_kernel.cpp ${Q4KP_DIR}/q4kp_wide_kernel.cpp)
set_target_properties(test-q4kp-kernels PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_executable(test-q4kp-runtime test-runtime.cpp)
foreach(target test-q4kp-kernels test-q4kp-runtime)
    target_include_directories(${target} PRIVATE "${PROJECT_SOURCE_DIR}/ggml/src"
        "${PROJECT_SOURCE_DIR}/ggml/include"
        "${PROJECT_SOURCE_DIR}/ggml/src/ggml-cpu" "${Q4KP_DIR}")
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE ${Q4KP_FLAGS})
    target_compile_definitions(${target} PRIVATE GGML_CPU_Q4KP)
    target_link_libraries(${target} PRIVATE ggml-cpu)
endforeach()

get_filename_component(Q4KP_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
add_test(NAME q4kp-kernels COMMAND "${CMAKE_COMMAND}" -E env
    "Q4KP_TEST_LIBRARY=$<TARGET_FILE:test-q4kp-kernels>"
    "Q4KP_TEST_RUNTIME_DIR=${Q4KP_COMPILER_DIR}"
    "${Python3_EXECUTABLE}" -B -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}" -p "test_q4kp*.py" -v)
foreach(mode off p6 vnni wide invalid)
    add_test(NAME q4kp-runtime-${mode} COMMAND "${CMAKE_COMMAND}" -E env
        "GGML_Q4KP=${mode}" $<TARGET_FILE:test-q4kp-runtime>)
endforeach()
