# llama-bench-impl: benchmark logic, reusable by app

set(TARGET llama-bench-impl)

add_library(${TARGET} llama-bench.cpp)
set_target_properties(${TARGET} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC llama-common llama ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

# llama-bench executable

set(TARGET llama-bench)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-bench-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

if(LLAMA_BUILD_TESTS)
    set(LLAMA_BENCH_TEST_MODEL "${CMAKE_BINARY_DIR}/tests/test-models/qwen35-dense.gguf")

    add_test(
        NAME test-llama-bench-continue-on-error
        COMMAND ${CMAKE_COMMAND}
            -DLLAMA_BENCH=$<TARGET_FILE:llama-bench>
            -DGOOD_MODEL=${LLAMA_BENCH_TEST_MODEL}
            -DMISSING_MODEL=${CMAKE_CURRENT_BINARY_DIR}/missing-model.gguf
            -P ${CMAKE_CURRENT_SOURCE_DIR}/test-continue-on-error.cmake
    )
    set_tests_properties(test-llama-bench-continue-on-error PROPERTIES
        LABELS main
        FIXTURES_REQUIRED generate-models
    )
endif()
