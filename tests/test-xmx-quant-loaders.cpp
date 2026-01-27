// Unit tests for XMX quantization loader infrastructure
// Tests QuantTraits compile-time constants and XMXTileLoader dequantization

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

// Include GGML common definitions for block types
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml.h"

// Include the header under test
#include "ggml-sycl/xmx-quant-loaders.hpp"

// Test utilities
static const char* RESULT_STR[] = {"ok", "FAILED"};
static int num_failed = 0;

// Helper to compare floats with tolerance
static bool float_eq(float a, float b, float tolerance = 1e-4f) {
    return std::fabs(a - b) < tolerance;
}

// =============================================================================
// QuantTraits Tests
// =============================================================================

void test_quant_traits_q4_0() {
    printf("Testing QuantTraits<GGML_TYPE_Q4_0>... ");

    using Traits = xmx::QuantTraits<GGML_TYPE_Q4_0>;

    bool failed = false;

    // Block size should be 32
    if (Traits::block_size != 32) {
        printf("block_size=%d (expected 32) ", Traits::block_size);
        failed = true;
    }

    // Bytes per block should be 18 (2 for d + 16 for qs)
    if (Traits::bytes_per_block != sizeof(block_q4_0)) {
        printf("bytes_per_block=%d (expected %zu) ", Traits::bytes_per_block, sizeof(block_q4_0));
        failed = true;
    }

    // Bits per weight should be 4
    if (Traits::bits_per_weight != 4) {
        printf("bits_per_weight=%d (expected 4) ", Traits::bits_per_weight);
        failed = true;
    }

    // Has super block should be false
    if (Traits::has_super_block != false) {
        printf("has_super_block=%d (expected false) ", Traits::has_super_block);
        failed = true;
    }

    // Check block_type is correct
    static_assert(std::is_same<Traits::block_type, block_q4_0>::value,
                  "block_type should be block_q4_0");

    printf("%s\n", RESULT_STR[failed]);
    if (failed) num_failed++;
}

void test_quant_traits_q8_0() {
    printf("Testing QuantTraits<GGML_TYPE_Q8_0>... ");

    using Traits = xmx::QuantTraits<GGML_TYPE_Q8_0>;

    bool failed = false;

    // Block size should be 32
    if (Traits::block_size != 32) {
        printf("block_size=%d (expected 32) ", Traits::block_size);
        failed = true;
    }

    // Bytes per block should be 34 (2 for d + 32 for qs)
    if (Traits::bytes_per_block != sizeof(block_q8_0)) {
        printf("bytes_per_block=%d (expected %zu) ", Traits::bytes_per_block, sizeof(block_q8_0));
        failed = true;
    }

    // Bits per weight should be 8
    if (Traits::bits_per_weight != 8) {
        printf("bits_per_weight=%d (expected 8) ", Traits::bits_per_weight);
        failed = true;
    }

    // Has super block should be false
    if (Traits::has_super_block != false) {
        printf("has_super_block=%d (expected false) ", Traits::has_super_block);
        failed = true;
    }

    // Check block_type is correct
    static_assert(std::is_same<Traits::block_type, block_q8_0>::value,
                  "block_type should be block_q8_0");

    printf("%s\n", RESULT_STR[failed]);
    if (failed) num_failed++;
}

void test_quant_traits_q6_k() {
    printf("Testing QuantTraits<GGML_TYPE_Q6_K>... ");

    using Traits = xmx::QuantTraits<GGML_TYPE_Q6_K>;

    bool failed = false;

    // Block size should be 256 (QK_K)
    if (Traits::block_size != 256) {
        printf("block_size=%d (expected 256) ", Traits::block_size);
        failed = true;
    }

    // Bytes per block should match sizeof(block_q6_K)
    if (Traits::bytes_per_block != sizeof(block_q6_K)) {
        printf("bytes_per_block=%d (expected %zu) ", Traits::bytes_per_block, sizeof(block_q6_K));
        failed = true;
    }

    // Bits per weight should be 6
    if (Traits::bits_per_weight != 6) {
        printf("bits_per_weight=%d (expected 6) ", Traits::bits_per_weight);
        failed = true;
    }

    // Has super block should be true (K-quants have super blocks)
    if (Traits::has_super_block != true) {
        printf("has_super_block=%d (expected true) ", Traits::has_super_block);
        failed = true;
    }

    // Check block_type is correct
    static_assert(std::is_same<Traits::block_type, block_q6_K>::value,
                  "block_type should be block_q6_K");

    printf("%s\n", RESULT_STR[failed]);
    if (failed) num_failed++;
}

void test_quant_traits_q4_k() {
    printf("Testing QuantTraits<GGML_TYPE_Q4_K>... ");

    using Traits = xmx::QuantTraits<GGML_TYPE_Q4_K>;

    bool failed = false;

    // Block size should be 256 (QK_K)
    if (Traits::block_size != 256) {
        printf("block_size=%d (expected 256) ", Traits::block_size);
        failed = true;
    }

    // Bytes per block should match sizeof(block_q4_K)
    if (Traits::bytes_per_block != sizeof(block_q4_K)) {
        printf("bytes_per_block=%d (expected %zu) ", Traits::bytes_per_block, sizeof(block_q4_K));
        failed = true;
    }

    // Bits per weight should be 4
    if (Traits::bits_per_weight != 4) {
        printf("bits_per_weight=%d (expected 4) ", Traits::bits_per_weight);
        failed = true;
    }

    // Has super block should be true (K-quants have super blocks)
    if (Traits::has_super_block != true) {
        printf("has_super_block=%d (expected true) ", Traits::has_super_block);
        failed = true;
    }

    // Check block_type is correct
    static_assert(std::is_same<Traits::block_type, block_q4_K>::value,
                  "block_type should be block_q4_K");

    printf("%s\n", RESULT_STR[failed]);
    if (failed) num_failed++;
}

// =============================================================================
// Dequantization Tests
// =============================================================================

void test_dequant_q4_0() {
    printf("Testing Q4_0 dequantization... ");

    // Create a test Q4_0 block with known values
    block_q4_0 block;

    // Set scale (d) = 0.5 as fp16
    // ggml_half is uint16_t, we need to convert float to fp16
    float d_val = 0.5f;
    memcpy(&block.d, &d_val, sizeof(ggml_half)); // Note: This is actually wrong for fp16
    // For proper fp16: use a union or conversion function
    // Let's use a simpler approach: set qs to known pattern

    // Actually, let's manually set up a simple test case:
    // d = 1.0 (scale factor)
    // qs[i] contains two 4-bit values: low nibble and high nibble
    // After dequant: value = (nibble - 8) * d

    // For proper testing, we'll use raw bytes:
    // fp16 1.0 = 0x3C00
    uint16_t d_fp16 = 0x3C00; // 1.0 in fp16
    memcpy(&block.d, &d_fp16, sizeof(uint16_t));

    // Set qs to pattern: qs[0] = 0x98 means low=8 (val=0), high=9 (val=1)
    for (int i = 0; i < 16; i++) {
        // Each byte: low nibble = i, high nibble = i+1
        // So values after dequant: (i-8)*1.0 and (i+1-8)*1.0
        block.qs[i] = static_cast<uint8_t>((((i + 1) & 0xF) << 4) | (i & 0xF));
    }

    // Dequantize
    std::vector<float> output(32);
    xmx::dequant_block_q4_0(&block, output.data());

    bool failed = false;

    // Check first 16 values (low nibbles)
    for (int i = 0; i < 16; i++) {
        float expected = static_cast<float>((i & 0xF) - 8) * 1.0f;
        if (!float_eq(output[i], expected, 0.01f)) {
            printf("output[%d]=%.4f (expected %.4f) ", i, output[i], expected);
            failed = true;
            break;
        }
    }

    // Check last 16 values (high nibbles)
    for (int i = 0; i < 16 && !failed; i++) {
        float expected = static_cast<float>(((i + 1) & 0xF) - 8) * 1.0f;
        if (!float_eq(output[i + 16], expected, 0.01f)) {
            printf("output[%d]=%.4f (expected %.4f) ", i + 16, output[i + 16], expected);
            failed = true;
            break;
        }
    }

    printf("%s\n", RESULT_STR[failed]);
    if (failed) num_failed++;
}

void test_dequant_q8_0() {
    printf("Testing Q8_0 dequantization... ");

    // Create a test Q8_0 block with known values
    block_q8_0 block;

    // d = 0.5 in fp16 (0x3800)
    uint16_t d_fp16 = 0x3800; // 0.5 in fp16
    memcpy(&block.d, &d_fp16, sizeof(uint16_t));

    // Set qs to sequential signed values: -16 to 15
    for (int i = 0; i < 32; i++) {
        block.qs[i] = static_cast<int8_t>(i - 16);
    }

    // Dequantize
    std::vector<float> output(32);
    xmx::dequant_block_q8_0(&block, output.data());

    bool failed = false;

    // Check all 32 values
    for (int i = 0; i < 32; i++) {
        float expected = static_cast<float>(i - 16) * 0.5f;
        if (!float_eq(output[i], expected, 0.01f)) {
            printf("output[%d]=%.4f (expected %.4f) ", i, output[i], expected);
            failed = true;
            break;
        }
    }

    printf("%s\n", RESULT_STR[failed]);
    if (failed) num_failed++;
}

// =============================================================================
// XMXTileLoader Template Tests
// =============================================================================

void test_tile_loader_q4_0_traits() {
    printf("Testing XMXTileLoader<Q4_0, 16, 32> traits... ");

    using Loader = xmx::XMXTileLoader<GGML_TYPE_Q4_0, 16, 32>;

    bool failed = false;

    // Check that Traits are accessible
    if (Loader::Traits::block_size != 32) {
        printf("Traits::block_size=%d (expected 32) ", Loader::Traits::block_size);
        failed = true;
    }

    // Check tile dimensions are stored correctly
    if (Loader::TileN != 16) {
        printf("TileN=%d (expected 16) ", Loader::TileN);
        failed = true;
    }

    if (Loader::TileK != 32) {
        printf("TileK=%d (expected 32) ", Loader::TileK);
        failed = true;
    }

    printf("%s\n", RESULT_STR[failed]);
    if (failed) num_failed++;
}

void test_tile_loader_q8_0_traits() {
    printf("Testing XMXTileLoader<Q8_0, 8, 64> traits... ");

    using Loader = xmx::XMXTileLoader<GGML_TYPE_Q8_0, 8, 64>;

    bool failed = false;

    // Check that Traits are accessible
    if (Loader::Traits::block_size != 32) {
        printf("Traits::block_size=%d (expected 32) ", Loader::Traits::block_size);
        failed = true;
    }

    // Check tile dimensions are stored correctly
    if (Loader::TileN != 8) {
        printf("TileN=%d (expected 8) ", Loader::TileN);
        failed = true;
    }

    if (Loader::TileK != 64) {
        printf("TileK=%d (expected 64) ", Loader::TileK);
        failed = true;
    }

    printf("%s\n", RESULT_STR[failed]);
    if (failed) num_failed++;
}

// =============================================================================
// Compile-time validation tests
// =============================================================================

void test_static_assertions() {
    printf("Testing compile-time validations... ");

    // These should compile successfully - they verify static_asserts work
    // The header should have static_asserts for tile alignment

    // Test that we can instantiate with valid tile sizes
    using ValidLoader1 = xmx::XMXTileLoader<GGML_TYPE_Q4_0, 16, 32>;  // K=32 aligns with block_size=32
    using ValidLoader2 = xmx::XMXTileLoader<GGML_TYPE_Q4_0, 8, 64>;   // K=64 is multiple of 32
    using ValidLoader3 = xmx::XMXTileLoader<GGML_TYPE_Q8_0, 16, 32>;  // K=32 aligns with block_size=32

    // Verify types exist
    (void)sizeof(ValidLoader1);
    (void)sizeof(ValidLoader2);
    (void)sizeof(ValidLoader3);

    printf("%s\n", RESULT_STR[0]); // Always ok if compilation succeeds
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("=== XMX Quant Loaders Unit Tests ===\n\n");

    printf("--- QuantTraits Tests ---\n");
    test_quant_traits_q4_0();
    test_quant_traits_q8_0();
    test_quant_traits_q6_k();
    test_quant_traits_q4_k();

    printf("\n--- Dequantization Tests ---\n");
    test_dequant_q4_0();
    test_dequant_q8_0();

    printf("\n--- XMXTileLoader Tests ---\n");
    test_tile_loader_q4_0_traits();
    test_tile_loader_q8_0_traits();

    printf("\n--- Compile-time Validation Tests ---\n");
    test_static_assertions();

    printf("\n=== Summary ===\n");
    printf("%d tests failed\n", num_failed);

    return num_failed > 0 ? 1 : 0;
}
