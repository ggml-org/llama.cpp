// Tests for Multi-Modal LoRA (MMLoRA) functionality
#include "mtmd.h"
#include "mtmd-helper.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cstring>

#undef NDEBUG
#include <cassert>

#define LOG_INF(...) printf(__VA_ARGS__)

#define TEST(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "TEST FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        return 1; \
    } \
} while(0)

// Test string↔enum conversion functions
static int test_mmlora_modality_type_conversion() {
LOG_INF("Testing modality type string conversions...\n");

    // Test enum to string
    TEST(strcmp(mtmd_modality_type_to_str(MTMD_INPUT_CHUNK_TYPE_IMAGE), "image") == 0);
    TEST(strcmp(mtmd_modality_type_to_str(MTMD_INPUT_CHUNK_TYPE_AUDIO), "audio") == 0);
    TEST(strcmp(mtmd_modality_type_to_str(MTMD_INPUT_CHUNK_TYPE_TEXT), "text") == 0);
    TEST(strcmp(mtmd_modality_type_to_str(MTMD_INPUT_CHUNK_TYPE_UNKNOWN), "unknown") == 0);
    TEST(strcmp(mtmd_modality_type_to_str((enum mtmd_input_chunk_type)999), "unknown") == 0);

    // Test string to enum
    TEST(mtmd_modality_type_from_str("image") == MTMD_INPUT_CHUNK_TYPE_IMAGE);
    TEST(mtmd_modality_type_from_str("audio") == MTMD_INPUT_CHUNK_TYPE_AUDIO);
    TEST(mtmd_modality_type_from_str("text") == MTMD_INPUT_CHUNK_TYPE_TEXT);
    TEST(mtmd_modality_type_from_str("unknown") == MTMD_INPUT_CHUNK_TYPE_UNKNOWN);
    TEST(mtmd_modality_type_from_str("invalid") == MTMD_INPUT_CHUNK_TYPE_UNKNOWN);

    // Test validation
    TEST(mtmd_is_valid_modality_str("image") == true);
    TEST(mtmd_is_valid_modality_str("audio") == true);
    TEST(mtmd_is_valid_modality_str("text") == false);
    TEST(mtmd_is_valid_modality_str("unknown") == false);
    TEST(mtmd_is_valid_modality_str("invalid") == false);
    TEST(mtmd_is_valid_modality_str(NULL) == false);

    LOG_INF("  Passed modality type conversions\n");
    return 0;
}

// Test CLI argument parsing (simulated)
static int test_mmlora_arg_parsing() {LOG_INF("Testing MMLoRA argument parsing logic...\n");

    // Simulate parsing "0:image,audio"
    std::string value = "0:image,audio";
    size_t colon_pos = value.find(':');
    TEST(colon_pos != std::string::npos);

    std::string index_str = value.substr(0, colon_pos);
    std::string modalities_str = value.substr(colon_pos + 1);

    TEST(index_str == "0");
    std::vector<std::string> modality_strs = string_split<std::string>(modalities_str, ',');
    TEST(modality_strs.size() == 2);
    TEST(modality_strs[0] == "image");
    TEST(modality_strs[1] == "audio");

    // Validate modalities
    for (const auto & mod : modality_strs) {
        TEST(mod == "image" || mod == "audio");
    }

    LOG_INF("  Passed argument parsing logic\n");
    return 0;
}

int main(int argc, char ** argv) {
    (void)argc;
    (void)argv;

    LOG_INF("MMLoRA unit tests starting...\n");

    int result = 0;
    result += test_mmlora_modality_type_conversion();
    result += test_mmlora_arg_parsing();

    LOG_INF("MMLoRA unit tests %s!\n", result == 0 ? "passed" : "failed");
    return result;
}
