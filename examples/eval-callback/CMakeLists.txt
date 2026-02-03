set(TARGET llama-eval-callback)
add_executable(${TARGET} eval-callback.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_BUILD_TESTS)
    if(NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
        set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
        set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
    else()
        set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
        set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
    endif()
    set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")
    set(TEST_TARGET test-eval-callback)
    add_test(NAME ${TEST_TARGET}-download-model COMMAND ${CMAKE_COMMAND}
        -DDEST=${MODEL_DEST}
        -DNAME=${MODEL_NAME}
        -DHASH=${MODEL_HASH}
        -P ${CMAKE_SOURCE_DIR}/cmake/download-models.cmake
    )
    set_tests_properties(${TEST_TARGET}-download-model PROPERTIES FIXTURES_SETUP ${TEST_TARGET}-download-model)
    add_test(NAME ${TEST_TARGET} COMMAND llama-eval-callback -m "${MODEL_DEST}" --prompt hello --seed 42 -ngl 0)
    set_tests_properties(${TEST_TARGET} PROPERTIES FIXTURES_REQUIRED ${TEST_TARGET}-download-model)
endif()
