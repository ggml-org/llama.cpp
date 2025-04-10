set(TARGET llama-eval-callback)
add_executable(${TARGET} eval-callback.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TEST_TARGET test-eval-callback)
add_test(NAME ${TEST_TARGET}
        COMMAND llama-eval-callback --hf-repo ggml-org/models --hf-file tinyllamas/stories260K.gguf --model stories260K.gguf --prompt hello --seed 42 -ngl 0)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS eval-callback curl)
