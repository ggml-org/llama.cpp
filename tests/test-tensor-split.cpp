#include "llama-model.h"

#include <cstdio>

static int failures = 0;

static void expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        ++failures;
    }
}

int main() {
    const float equal4[] = {1, 1, 1, 1};
    expect(llama_tensor_split_is_valid(2048, 256, equal4, 4, 0), "equal split rotation 0");
    expect(llama_tensor_split_is_valid(2048, 256, equal4, 4, 1), "equal split rotation 1");
    expect(llama_tensor_split_is_valid(2048, 256, equal4, 4, 2), "equal split rotation 2");
    expect(llama_tensor_split_is_valid(2048, 256, equal4, 4, 3), "equal split rotation 3");

    // Qwen3.6 27B vocabulary-axis output split: 248320 / 4 = 62080.
    expect(llama_tensor_split_is_valid(248320, 128, equal4, 4, 0), "vocabulary split rotation 0");
    expect(llama_tensor_split_is_valid(248320, 128, equal4, 4, 1), "vocabulary split rotation 1");
    expect(llama_tensor_split_has_min_width(248320, 128, 256, equal4, 4, 0), "vocabulary shards meet top-k minimum");
    expect(!llama_tensor_split_is_valid(248319, 128, equal4, 4, 0), "unaligned vocabulary rejected");

    // Rotation can turn a valid unrotated ratio into a zero-width physical rank.
    const float skewed[] = {0.45f, 0.45f, 0.05f, 0.05f};
    expect(llama_tensor_split_is_valid(2048, 128, skewed, 4, 0), "skewed split rotation 0");
    expect(llama_tensor_split_has_min_width(2048, 128, 128, skewed, 4, 0), "skewed split minimum 128");
    expect(!llama_tensor_split_has_min_width(2048, 128, 256, skewed, 4, 0), "skewed split minimum 256 rejected");
    expect(!llama_tensor_split_is_valid(2048, 128, skewed, 4, 2), "skewed split rotation 2 rejected");

    const float zero_ranks[] = {1, 0, 0, 0};
    expect(!llama_tensor_split_is_valid(2048, 128, zero_ranks, 4, 0), "zero-width ranks rejected");
    expect(!llama_tensor_split_is_valid(2000, 128, equal4, 4, 0), "non-divisible width rejected");
    expect(!llama_tensor_split_is_valid(2048, 128, equal4, 1, 0), "single device rejected");
    expect(!llama_tensor_split_is_valid(0, 128, equal4, 4, 0), "zero tensor width rejected");

    if (failures != 0) {
        std::fprintf(stderr, "%d tensor split validation test(s) failed\n", failures);
        return 1;
    }
    std::puts("tensor split validation tests passed");
    return 0;
}