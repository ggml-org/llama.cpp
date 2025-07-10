include_directories(.)

if(LLAMA_PARQUET)
    find_package(Arrow REQUIRED)
    find_package(Parquet REQUIRED)
endif()

add_library(dataset-to-gguf-lib
        dataset-to-gguf/llama-gguf-writer.cpp
        dataset-to-gguf/llama-gguf-file.cpp
        dataset-to-gguf/llama-dataset-reader/llama-text-data-reader.cpp
        dataset-to-gguf/llama-gguf-converter.cpp
        dataset-to-gguf/llama-gguf-reader.cpp
        dataset-to-gguf/llama-dataset-reader/llama-parquet-data-reader.cpp
)

# Link libraries for dataset-to-gguf-lib
target_link_libraries(dataset-to-gguf-lib common llama ${CMAKE_THREAD_LIBS_INIT})
if(LLAMA_PARQUET)
    target_link_libraries(dataset-to-gguf-lib Arrow::arrow_shared Parquet::parquet_shared)
endif()
target_compile_features(dataset-to-gguf-lib PRIVATE cxx_std_11)


add_executable(convert-to-train-gguf convert-to-train-gguf.cpp)
target_link_libraries(convert-to-train-gguf PRIVATE dataset-to-gguf-lib) # Link to the new library
target_compile_features(convert-to-train-gguf PRIVATE cxx_std_11) # Apply C++ standard to the executable

# Define the executable for the unit tests
set(TEST_TARGET_GGUF_DATASET_CONVERTER_TESTS dataset-to-gguf-unit-tests)
add_executable(${TEST_TARGET_GGUF_DATASET_CONVERTER_TESTS} dataset-to-gguf/tests/dataset-to-gguf-tests.cpp)

# Link necessary libraries for the test executable
target_link_libraries(${TEST_TARGET_GGUF_DATASET_CONVERTER_TESTS} PRIVATE common llama dataset-to-gguf-lib)

# Ensure C++17 for filesystem usage for the test executable
target_compile_features(${TEST_TARGET_GGUF_DATASET_CONVERTER_TESTS} PRIVATE cxx_std_17)

add_test(
        NAME ${TEST_TARGET_GGUF_DATASET_CONVERTER_TESTS} #
        COMMAND $<TARGET_FILE:${TEST_TARGET_GGUF_DATASET_CONVERTER_TESTS}>
        LABEL "training"
)
