#include <cassert>     // For assert
#include <filesystem>  // For working with the file system (creating/deleting temporary files)
#include <fstream>
#include <iostream>    // For std::cerr
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../llama-dataset-reader/llama-text-data-reader.h"
#include "common.h"
#include "dataset-to-gguf/llama-gguf-converter.h"
#include "dataset-to-gguf/llama-gguf-reader.h"
#include "dataset-to-gguf/llama-gguf-writer.h"
#include "llama.h"  // For llama_backend_init, llama_backend_free, llama_model_load_from_file, llama_model_free

namespace fs = std::filesystem;

// Global variables for tests requiring llama_model
static llama_model * g_llama_model     = nullptr;
static std::string   g_test_model_path = "../../gte-small.Q2_K.gguf";  // Specify the actual path to your model

// Helper for assertions
#define TEST_ASSERT(condition, message)                                                                              \
    do {                                                                                                             \
        if (!(condition)) {                                                                                          \
            std::cerr << "Assertion failed: " << message << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            return false;                                                                                            \
        }                                                                                                            \
    } while (0)
bool SetUpLlamaBackend();
bool Testllama_gguf_file_DefaultConstructorInitializesContext();
bool Testllama_gguf_file_ConstructorFromFileThrowsOnError();
bool Testllama_gguf_file_SetAndGetMetadataString();
bool Testllama_gguf_file_SetAndGetMetadataU64();
bool Testllama_gguf_file_SetAndGetMetadataStringArray();
bool CreateTestllama_gguf_file(const std::string & path, llama_model * model_ptr);
bool Testllama_gguf_reader_ConstructorInitializesFromFile();
bool Testllama_gguf_reader_GetMetadata();
bool Testllama_gguf_reader_GetTensorCount();
bool Testllama_gguf_reader_GetTensorNameAndTypeAndSize();
bool Testllama_gguf_reader_ReadTensorData();
bool Testllama_gguf_reader_ReadTensorDataInvalidIndex();
bool TestTextDataReader_OpenFile();
bool TestTextDataReader_ReadNextSequenceTextMode();
bool TestTextDataReader_ReadNextSequencePreTokenizedMode();
bool TestTextDataReader_ResetFunctionality();
bool TestTextDataReader_GetTotalSequences();
bool Testllama_gguf_converter_ConvertTextFileSuccess();
void TearDownLlamaBackend();


// Global setup for llama.cpp backend
bool SetUpLlamaBackend() {
    llama_backend_init();
    // Load the model for the tokenizer
    llama_model_params model_params = llama_model_default_params();
    g_llama_model = llama_model_load_from_file(g_test_model_path.c_str(), model_params);
    if (g_llama_model == nullptr) {
        std::cerr << "WARNING: Failed to load llama model for tests from " << g_test_model_path
                  << ". Some tests may be skipped or fail." << std::endl;
        // It's okay to continue if model loading fails, but tests relying on it will skip.
    }
    return true;
}

// Global teardown for llama.cpp backend
void TearDownLlamaBackend() {
    if (g_llama_model) {
        llama_model_free(g_llama_model);
        g_llama_model = nullptr;
    }
    llama_backend_free();
}

// =============================================================================
// Tests for llama_gguf_file
// =============================================================================

bool Testllama_gguf_file_DefaultConstructorInitializesContext() {
    printf("  Testllama_gguf_file_DefaultConstructorInitializesContext\n");
    llama_gguf_file gguf_file;
    TEST_ASSERT(gguf_file.llama_gguf_file_is_initialized(),
                "llama_gguf_file should be initialized by default constructor");
    return true;
}

bool Testllama_gguf_file_ConstructorFromFileThrowsOnError() {
    printf("  Testllama_gguf_file_ConstructorFromFileThrowsOnError\n");
    bool threw_exception = false;
    try {
        llama_gguf_file("non_existent_file.gguf");
    } catch (const std::runtime_error & e) {
        threw_exception = true;
    }
    TEST_ASSERT(threw_exception, "Constructor should throw for non-existent file");
    return true;
}

bool Testllama_gguf_file_SetAndGetMetadataString() {
    printf("  Testllama_gguf_file_SetAndGetMetadataString\n");
    llama_gguf_file gguf_file;
    gguf_file.llama_gguf_file_set_val_str("test.key.string", "test_value");
    TEST_ASSERT(gguf_file.llama_gguf_file_get_val_str("test.key.string") == "test_value",
                "Failed to get correct string value");
    TEST_ASSERT(gguf_file.llama_gguf_file_get_val_str("non.existent.key", "default_value") == "default_value",
                "Failed to get default string value");
    return true;
}

bool Testllama_gguf_file_SetAndGetMetadataU64() {
    printf("  Testllama_gguf_file_SetAndGetMetadataU64\n");
    llama_gguf_file gguf_file;
    gguf_file.llama_gguf_file_set_val_u64("test.key.u64", 12345ULL);
    TEST_ASSERT(gguf_file.llama_gguf_file_get_val_u64("test.key.u64") == 12345ULL, "Failed to get correct u64 value");
    TEST_ASSERT(gguf_file.llama_gguf_file_get_val_u64("non.existent.key.u64", 99ULL) == 99ULL,
                "Failed to get default u64 value");
    return true;
}

bool Testllama_gguf_file_SetAndGetMetadataStringArray() {
    printf("  Testllama_gguf_file_SetAndGetMetadataStringArray\n");
    llama_gguf_file           gguf_file;
    std::vector<const char *> arr = { "val1", "val2", "val3" };
    gguf_file.llama_gguf_file_set_arr_str("test.key.array_str", arr);
    // As noted before, verifying array content requires more complex logic to read the GGUF file.
    // For now, we assert that the operation doesn't crash.
    return true;
}

// =============================================================================
// Tests for llama_gguf_reader
// =============================================================================

// Helper to create a temporary GGUF file for llama_gguf_reader tests
bool CreateTestllama_gguf_file(const std::string & path, llama_model * model_ptr) {
    llama_gguf_file   writer_file;
    llama_gguf_writer writer(&writer_file);

    writer.llama_gguf_writer_init_metadata(model_ptr, "dummy_input.txt", 2);  // 2 sequences

    std::vector<llama_token> seq1 = { 1, 2, 3, 4, 5 };
    std::vector<llama_token> seq2 = { 10, 20, 30 };
    writer.llama_gguf_writer_add_sequence_tensor(0, seq1);
    writer.llama_gguf_writer_add_sequence_tensor(1, seq2);

    return writer.llama_gguf_writer_write_to_file(path);
}

bool Testllama_gguf_reader_ConstructorInitializesFromFile() {
    printf("  Testllama_gguf_reader_ConstructorInitializesFromFile\n");
    std::string test_gguf_path = "test_output_reader.gguf";
    TEST_ASSERT(CreateTestllama_gguf_file(test_gguf_path, g_llama_model),
                "Failed to create test GGUF file for reader test");

    llama_gguf_reader reader(test_gguf_path);
    TEST_ASSERT(reader.llama_gguf_reader_is_initialized(), "llama_gguf_reader should be initialized from file");
    fs::remove(test_gguf_path);
    return true;
}

bool Testllama_gguf_reader_GetMetadata() {
    printf("  Testllama_gguf_reader_GetMetadata\n");
    std::string test_gguf_path = "test_output_reader_meta.gguf";
    TEST_ASSERT(CreateTestllama_gguf_file(test_gguf_path, g_llama_model),
                "Failed to create test GGUF file for reader meta test");

    llama_gguf_reader reader(test_gguf_path);
    TEST_ASSERT(reader.llama_gguf_reader_get_metadata_str("training.dataset.name") == "dummy_input.txt",
                "Incorrect dataset name");
    TEST_ASSERT(reader.llama_gguf_reader_get_metadata_u64("training.sequence.count") == 2ULL,
                "Incorrect sequence count");
    // The tokenizer model name might vary, so just check it's not empty/default if model was loaded
    if (g_llama_model) {
        TEST_ASSERT(reader.llama_gguf_reader_get_metadata_str("training.tokenizer.gguf.model", "default") != "default",
                    "Tokenizer model name should not be default");
    }
    fs::remove(test_gguf_path);
    return true;
}

bool Testllama_gguf_reader_GetTensorCount() {
    printf("  Testllama_gguf_reader_GetTensorCount\n");
    std::string test_gguf_path = "test_output_reader_count.gguf";
    TEST_ASSERT(CreateTestllama_gguf_file(test_gguf_path, g_llama_model),
                "Failed to create test GGUF file for reader count test");

    llama_gguf_reader reader(test_gguf_path);
    TEST_ASSERT(reader.llama_gguf_reader_get_tensor_count() == 2, "Incorrect tensor count");
    fs::remove(test_gguf_path);
    return true;
}

bool Testllama_gguf_reader_GetTensorNameAndTypeAndSize() {
    printf("  Testllama_gguf_reader_GetTensorNameAndTypeAndSize\n");
    std::string test_gguf_path = "test_output_reader_tensor_info.gguf";
    TEST_ASSERT(CreateTestllama_gguf_file(test_gguf_path, g_llama_model),
                "Failed to create test GGUF file for reader tensor info test");

    llama_gguf_reader reader(test_gguf_path);
    TEST_ASSERT(reader.llama_gguf_reader_get_tensor_name(0) == "training.tensor.0",
                "Incorrect tensor name for index 0");
    TEST_ASSERT(reader.llama_gguf_reader_get_tensor_type(0) == GGML_TYPE_I32, "Incorrect tensor type for index 0");
    TEST_ASSERT(reader.llama_gguf_reader_get_tensor_size(0) == 5 * sizeof(llama_token),
                "Incorrect tensor size for index 0");

    TEST_ASSERT(reader.llama_gguf_reader_get_tensor_name(1) == "training.tensor.1",
                "Incorrect tensor name for index 1");
    TEST_ASSERT(reader.llama_gguf_reader_get_tensor_type(1) == GGML_TYPE_I32, "Incorrect tensor type for index 1");
    TEST_ASSERT(reader.llama_gguf_reader_get_tensor_size(1) == 3 * sizeof(llama_token),
                "Incorrect tensor size for index 1");
    fs::remove(test_gguf_path);
    return true;
}

bool Testllama_gguf_reader_ReadTensorData() {
    printf("  Testllama_gguf_reader_ReadTensorData\n");
    std::string test_gguf_path = "test_output_reader_data.gguf";
    TEST_ASSERT(CreateTestllama_gguf_file(test_gguf_path, g_llama_model),
                "Failed to create test GGUF file for reader data test");

    llama_gguf_reader        reader(test_gguf_path);
    std::vector<llama_token> tokens;

    // Read first sequence
    TEST_ASSERT(reader.llama_gguf_reader_read_tensor_data(0, tokens), "Failed to read tensor data for index 0");
    TEST_ASSERT(tokens.size() == 5, "Incorrect token count for index 0");
    TEST_ASSERT(tokens[0] == 1, "Incorrect token value at index 0, pos 0");
    TEST_ASSERT(tokens[4] == 5, "Incorrect token value at index 0, pos 4");

    // Read second sequence
    TEST_ASSERT(reader.llama_gguf_reader_read_tensor_data(1, tokens), "Failed to read tensor data for index 1");
    TEST_ASSERT(tokens.size() == 3, "Incorrect token count for index 1");
    TEST_ASSERT(tokens[0] == 10, "Incorrect token value at index 1, pos 0");
    TEST_ASSERT(tokens[2] == 30, "Incorrect token value at index 1, pos 2");
    fs::remove(test_gguf_path);
    return true;
}

bool Testllama_gguf_reader_ReadTensorDataInvalidIndex() {
    printf("  Testllama_gguf_reader_ReadTensorDataInvalidIndex\n");
    std::string test_gguf_path = "test_output_reader_invalid_idx.gguf";
    TEST_ASSERT(CreateTestllama_gguf_file(test_gguf_path, g_llama_model),
                "Failed to create test GGUF file for reader invalid index test");

    llama_gguf_reader        reader(test_gguf_path);
    std::vector<llama_token> tokens;
    TEST_ASSERT(!reader.llama_gguf_reader_read_tensor_data(99, tokens), "Reading invalid index should fail");
    fs::remove(test_gguf_path);
    return true;
}

// =============================================================================
// Tests for TextDataReader
// =============================================================================

// Helper to set up TextDataReader test files
struct TextDataReaderTestFixture {
    std::string   test_text_file         = "test_input.txt";
    std::string   test_pretokenized_file = "test_pretokenized.txt";
    llama_model * model_for_reader_test  = nullptr;

    TextDataReaderTestFixture(llama_model * model) : model_for_reader_test(model) {
        // Create test text file
        std::ofstream ofs(test_text_file);
        ofs << "Hello world\n";
        ofs << "This is a test line.\n";
        ofs << "\n";  // Empty line
        ofs << "Another line";
        ofs.close();

        // Create test pre-tokenized file
        std::ofstream ofs_pretokenized(test_pretokenized_file);
        ofs_pretokenized << "101 200 300 102\n";
        ofs_pretokenized << "500 600\n";
        ofs_pretokenized << "\n";  // Empty line
        ofs_pretokenized << "700";
        ofs_pretokenized.close();
    }

    ~TextDataReaderTestFixture() {
        fs::remove(test_text_file);
        fs::remove(test_pretokenized_file);
    }
};

bool TestTextDataReader_OpenFile() {
    printf("  TestTextDataReader_OpenFile\n");
    TextDataReaderTestFixture fixture(g_llama_model);
    llama_text_dataset_reader reader(fixture.model_for_reader_test, 128, false);
    TEST_ASSERT(reader.open(fixture.test_text_file), "Failed to open valid text file");
    reader.close();
    TEST_ASSERT(!reader.open("non_existent.txt"), "Opened non-existent file unexpectedly");
    return true;
}

bool TestTextDataReader_ReadNextSequenceTextMode() {
    printf("  TestTextDataReader_ReadNextSequenceTextMode\n");
    if (g_llama_model == nullptr) {
        printf("    Skipping: Llama model not loaded.\n");
        return true;  // Skip test gracefully
    }

    TextDataReaderTestFixture fixture(g_llama_model);
    llama_text_dataset_reader reader(fixture.model_for_reader_test, 128, false);
    TEST_ASSERT(reader.open(fixture.test_text_file), "Failed to open text file for read test");

    std::vector<llama_token> tokens;

    // Read "Hello world"
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read first sequence");
    TEST_ASSERT(!tokens.empty(), "First sequence should not be empty");

    // Read "This is a test line."
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read second sequence");
    TEST_ASSERT(!tokens.empty(), "Second sequence should not be empty");

    // Read empty line
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read empty line");
    TEST_ASSERT(tokens.empty(), "Empty line should result in 0 tokens");

    // Read "Another line"
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read third sequence");
    TEST_ASSERT(!tokens.empty(), "Third sequence should not be empty");

    // End of file
    TEST_ASSERT(!reader.read_next_sequence(tokens), "Should be end of file");
    reader.close();
    return true;
}

bool TestTextDataReader_ReadNextSequencePreTokenizedMode() {
    printf("  TestTextDataReader_ReadNextSequencePreTokenizedMode\n");
    TextDataReaderTestFixture fixture(g_llama_model);
    llama_text_dataset_reader reader(fixture.model_for_reader_test, 128, true);
    TEST_ASSERT(reader.open(fixture.test_pretokenized_file), "Failed to open pre-tokenized file for read test");

    std::vector<llama_token> tokens;

    // Read "101 200 300 102"
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read first pre-tokenized sequence");
    TEST_ASSERT(tokens.size() == 4, "Incorrect token count for first pre-tokenized sequence");
    TEST_ASSERT(tokens[0] == 101, "Incorrect token value for first pre-tokenized sequence");
    TEST_ASSERT(tokens[1] == 200, "Incorrect token value for first pre-tokenized sequence");

    // Read "500 600"
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read second pre-tokenized sequence");
    TEST_ASSERT(tokens.size() == 2, "Incorrect token count for second pre-tokenized sequence");
    TEST_ASSERT(tokens[0] == 500, "Incorrect token value for second pre-tokenized sequence");

    // Read empty line
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read empty pre-tokenized line");
    TEST_ASSERT(tokens.empty(), "Empty pre-tokenized line should result in 0 tokens");

    // Read "700"
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read third pre-tokenized sequence");
    TEST_ASSERT(tokens.size() == 1, "Incorrect token count for third pre-tokenized sequence");
    TEST_ASSERT(tokens[0] == 700, "Incorrect token value for third pre-tokenized sequence");

    // End of file
    TEST_ASSERT(!reader.read_next_sequence(tokens), "Should be end of pre-tokenized file");
    reader.close();
    return true;
}

bool TestTextDataReader_ResetFunctionality() {
    printf("  TestTextDataReader_ResetFunctionality\n");
    TextDataReaderTestFixture fixture(g_llama_model);
    llama_text_dataset_reader reader(fixture.model_for_reader_test, 128, false);
    TEST_ASSERT(reader.open(fixture.test_text_file), "Failed to open text file for reset test");

    std::vector<llama_token> tokens;
    reader.read_next_sequence(tokens);                      // Read one line
    reader.read_next_sequence(tokens);                      // Read another line

    TEST_ASSERT(reader.reset(), "Failed to reset reader");  // Reset to beginning

    // Should read the first line again
    TEST_ASSERT(reader.read_next_sequence(tokens), "Failed to read first sequence after reset");
    // (Add specific token check if you know the expected tokens for "Hello world")
    reader.close();
    return true;
}

bool TestTextDataReader_GetTotalSequences() {
    printf("  TestTextDataReader_GetTotalSequences\n");
    TextDataReaderTestFixture fixture(g_llama_model);

    llama_text_dataset_reader reader_text(fixture.model_for_reader_test, 128, false);
    TEST_ASSERT(reader_text.open(fixture.test_text_file), "Failed to open text file for total sequences test");
    TEST_ASSERT(reader_text.total_sequences() == 4,
                "Incorrect total sequence count for text file");  // 4 lines in test_input.txt
    reader_text.close();

    llama_text_dataset_reader reader_pretokenized(fixture.model_for_reader_test, 128, true);
    TEST_ASSERT(reader_pretokenized.open(fixture.test_pretokenized_file),
                "Failed to open pre-tokenized file for total sequences test");
    TEST_ASSERT(reader_pretokenized.total_sequences() == 4,
                "Incorrect total sequence count for pre-tokenized file");  // 4 lines in test_pretokenized.txt
    reader_pretokenized.close();
    return true;
}

// =============================================================================
// Tests for llama_gguf_converter (integration)
// =============================================================================

// Helper to set up llama_gguf_converter test files
struct llama_gguf_converterTestFixture {
    std::string   input_text_file          = "converter_input.txt";
    std::string   output_gguf_file         = "converter_output.gguf";
    llama_model * model_for_converter_test = nullptr;

    llama_gguf_converterTestFixture(llama_model * model) : model_for_converter_test(model) {
        // Create test text file
        std::ofstream ofs(input_text_file);
        ofs << "The quick brown fox jumps over the lazy dog.\n";
        ofs << "Hello, GGUF conversion!\n";
        ofs.close();
    }

    ~llama_gguf_converterTestFixture() {
        fs::remove(input_text_file);
        fs::remove(output_gguf_file);
    }
};

bool Testllama_gguf_converter_ConvertTextFileSuccess() {
    printf("  Testllama_gguf_converter_ConvertTextFileSuccess\n");
    if (g_llama_model == nullptr) {
        printf("    Skipping: Llama model not loaded.\n");
        return true;  // Skip test gracefully
    }

    llama_gguf_converterTestFixture fixture(g_llama_model);

    common_params params;
    params.in_files.push_back(fixture.input_text_file);
    params.out_file              = fixture.output_gguf_file;
    params.max_seq_len           = 128;
    params.pre_tokenized         = false;
    params.dataset_format        = "text";
#ifdef LLAMA_PARQUET
    params.parquet_text_column   = "text";    // Not used for text, but for completeness
    params.parquet_tokens_column = "tokens";  // Not used for text, but for completeness
#endif
    llama_gguf_converter converter;
    TEST_ASSERT(converter.llama_gguf_converter_convert(params, g_llama_model), "GGUF conversion failed");

    // Verify file was created
    TEST_ASSERT(fs::exists(fixture.output_gguf_file), "Output GGUF file was not created");

    // Verify GGUF file content using llama_gguf_reader
    llama_gguf_reader reader(fixture.output_gguf_file);
    TEST_ASSERT(reader.llama_gguf_reader_is_initialized(), "llama_gguf_reader failed to initialize for verification");
    TEST_ASSERT(reader.llama_gguf_reader_get_metadata_u64("training.sequence.count") == 2ULL,
                "Incorrect sequence count in GGUF metadata");
    TEST_ASSERT(reader.llama_gguf_reader_get_tensor_count() == 2, "Incorrect tensor count in GGUF file");

    std::vector<llama_token> tokens;
    TEST_ASSERT(reader.llama_gguf_reader_read_tensor_data(0, tokens), "Failed to read first tensor data");
    TEST_ASSERT(!tokens.empty(), "First sequence should not be empty");

    TEST_ASSERT(reader.llama_gguf_reader_read_tensor_data(1, tokens), "Failed to read second tensor data");
    TEST_ASSERT(!tokens.empty(), "Second sequence should not be empty");
    return true;
}

// =============================================================================
// Main function to run all tests
// =============================================================================

int main() {
    printf("Running dataset-to-gguf tests...\n\n");

    // Global setup for llama.cpp backend
    if (!SetUpLlamaBackend()) {
        printf("Global setup failed. Exiting tests.\n");
        return 1;
    }

    int failed_tests = 0;

    // Run llama_gguf_file tests
    printf("--- llama_gguf_file Tests ---\n");
    if (!Testllama_gguf_file_DefaultConstructorInitializesContext()) {
        failed_tests++;
    }
    if (!Testllama_gguf_file_ConstructorFromFileThrowsOnError()) {
        failed_tests++;
    }
    if (!Testllama_gguf_file_SetAndGetMetadataString()) {
        failed_tests++;
    }
    if (!Testllama_gguf_file_SetAndGetMetadataU64()) {
        failed_tests++;
    }
    if (!Testllama_gguf_file_SetAndGetMetadataStringArray()) {
        failed_tests++;
    }
    printf("\n");

    // Run llama_gguf_reader tests
    printf("--- llama_gguf_reader Tests ---\n");
    if (!Testllama_gguf_reader_ConstructorInitializesFromFile()) {
        failed_tests++;
    }
    if (!Testllama_gguf_reader_GetMetadata()) {
        failed_tests++;
    }
    if (!Testllama_gguf_reader_GetTensorCount()) {
        failed_tests++;
    }
    if (!Testllama_gguf_reader_GetTensorNameAndTypeAndSize()) {
        failed_tests++;
    }
    if (!Testllama_gguf_reader_ReadTensorData()) {
        failed_tests++;
    }
    if (!Testllama_gguf_reader_ReadTensorDataInvalidIndex()) {
        failed_tests++;
    }
    printf("\n");

    // Run TextDataReader tests
    printf("--- TextDataReader Tests ---\n");
    if (!TestTextDataReader_OpenFile()) {
        failed_tests++;
    }
    if (!TestTextDataReader_ReadNextSequenceTextMode()) {
        failed_tests++;
    }
    if (!TestTextDataReader_ReadNextSequencePreTokenizedMode()) {
        failed_tests++;
    }
    if (!TestTextDataReader_ResetFunctionality()) {
        failed_tests++;
    }
    if (!TestTextDataReader_GetTotalSequences()) {
        failed_tests++;
    }
    printf("\n");

    // Run llama_gguf_converter integration tests
    printf("--- llama_gguf_converter Tests ---\n");
    if (!Testllama_gguf_converter_ConvertTextFileSuccess()) {
        failed_tests++;
    }
    printf("\n");

    // Add ParquetDataReader tests here when you have test files and logic
    // printf("--- ParquetDataReader Tests ---\n");
    // if (!TestParquetDataReader_OpenFile()) failed_tests++;
    // ...

    // Global teardown for llama.cpp backend
    TearDownLlamaBackend();

    if (failed_tests == 0) {
        printf("All tests passed!\n");
        return 0;
    } else {
        printf("%d tests failed.\n", failed_tests);
        return 1;
    }
}
