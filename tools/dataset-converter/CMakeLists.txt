include_directories(. ../../common)

if(LLAMA_PARQUET)
    find_package(Arrow REQUIRED)
    find_package(Parquet REQUIRED)
endif()

add_library(dataset-to-gguf-lib STATIC
        dataset-to-gguf/llama-gguf-writer.cpp
        dataset-to-gguf/llama-gguf-file.cpp
        dataset-to-gguf/llama-dataset-reader/llama-text-data-reader.cpp
        dataset-to-gguf/llama-gguf-converter.cpp
        dataset-to-gguf/llama-gguf-reader.cpp
        dataset-to-gguf/llama-dataset-reader/llama-parquet-data-reader.cpp
)

target_compile_features(dataset-to-gguf-lib PRIVATE cxx_std_17)

target_link_libraries(dataset-to-gguf-lib common llama ${CMAKE_THREAD_LIBS_INIT})
if(LLAMA_PARQUET)
    target_link_libraries(dataset-to-gguf-lib Arrow::arrow_shared Parquet::parquet_shared)
endif()

add_executable(convert-to-train-gguf convert-to-train-gguf.cpp)
add_dependencies(convert-to-train-gguf dataset-to-gguf-lib)
target_link_libraries(convert-to-train-gguf PRIVATE dataset-to-gguf-lib)

set(TEST_TARGET_GGUF_DATASET_CONVERTER_TESTS dataset-to-gguf-unit-tests)
add_executable(${TEST_TARGET_GGUF_DATASET_CONVERTER_TESTS} dataset-to-gguf/tests/dataset-to-gguf-tests.cpp)
add_dependencies(${TEST_TARGET_GGUF_DATASET_CONVERTER_TESTS} dataset-to-gguf-lib)
target_link_libraries(${TEST_TARGET_GGUF_DATASET_CONVERTER_TESTS} PRIVATE common llama dataset-to-gguf-lib)

add_test(
        NAME ${TEST_TARGET_GGUF_DATASET_CONVERTER_TESTS} #
        COMMAND $<TARGET_FILE:${TEST_TARGET_GGUF_DATASET_CONVERTER_TESTS}>
        LABEL "training"
)
