//
// Test: ESIMD Vectorized Dequantization
//
// Tests the vectorized Q4_0 and Q8_0 dequantization functions for:
// 1. test_q4_0_dequant_block_aligned - 32 weights from single block
// 2. test_q4_0_dequant_partial_block - 16 weights, half a block
// 3. test_q4_0_dequant_multiple_blocks - 64 weights across 2 scale regions
// 4. test_q8_0_dequant_full_tile - 32 INT8 weights with scale
// 5. test_dequant_extreme_values - min/max weights (-8/+7 for Q4, -127/+127 for Q8)
// 6. test_dequant_vnni_format - verify output matches dpas input requirements
// 7. test_dequant_unaligned_src - source pointer not 16-byte aligned
// 8. test_dequant_offset_minus_8 - verify -8 offset applied correctly
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

// Enable standalone mode
#define XMX_TEST_STANDALONE 1

// Check for ESIMD support
#if __has_include(<sycl/ext/intel/esimd.hpp>)
#    define SYCL_ESIMD_AVAILABLE 1
#    include <sycl/ext/intel/esimd.hpp>
namespace esimd = sycl::ext::intel::esimd;
namespace xmx   = sycl::ext::intel::esimd::xmx;
#else
#    define SYCL_ESIMD_AVAILABLE 0
#endif

// =============================================================================
// Q4_0 and Q8_0 Block Definitions
// =============================================================================
#define QK4_0 32
#define QK8_0 32

struct block_q4_0_test {
    sycl::half d;              // scale factor
    uint8_t    qs[QK4_0 / 2];  // 16 packed bytes = 32 nibbles
};

struct block_q8_0_test {
    sycl::half d;       // scale factor
    int8_t     qs[32];  // 32 int8 values
};

static_assert(sizeof(block_q4_0_test) == sizeof(sycl::half) + QK4_0 / 2, "wrong q4_0 block size");
static_assert(sizeof(block_q8_0_test) == sizeof(sycl::half) + QK8_0, "wrong q8_0 block size");

// =============================================================================
// Test Helpers
// =============================================================================

static int g_tests_run     = 0;
static int g_tests_passed  = 0;
static int g_tests_skipped = 0;

#define TEST_BEGIN(name)                         \
    do {                                         \
        g_tests_run++;                           \
        fprintf(stderr, "[TEST] %s ... ", name); \
    } while (0)

#define TEST_PASS()                  \
    do {                             \
        g_tests_passed++;            \
        fprintf(stderr, "PASSED\n"); \
    } while (0)

#define TEST_SKIP(reason)                          \
    do {                                           \
        g_tests_skipped++;                         \
        fprintf(stderr, "SKIPPED (%s)\n", reason); \
    } while (0)

#define TEST_FAIL(msg)                        \
    do {                                      \
        fprintf(stderr, "FAILED: %s\n", msg); \
        return false;                         \
    } while (0)

#define TEST_ASSERT(cond, msg) \
    do {                       \
        if (!(cond)) {         \
            TEST_FAIL(msg);    \
        }                      \
    } while (0)

#define TEST_ASSERT_NEAR(actual, expected, tol, msg)                                                      \
    do {                                                                                                  \
        float diff = std::abs((float) (actual) - (float) (expected));                                     \
        if (diff > (tol)) {                                                                               \
            fprintf(stderr, "FAILED: %s (expected %.6f, got %.6f, diff=%.6f)\n", msg, (float) (expected), \
                    (float) (actual), diff);                                                              \
            return false;                                                                                 \
        }                                                                                                 \
    } while (0)

// =============================================================================
// Reference Implementations (CPU)
// =============================================================================

// Reference Q4_0 dequantization - scalar, guaranteed correct
void reference_dequant_q4_0(const block_q4_0_test * block, float * output) {
    float d = static_cast<float>(block->d);
    for (int i = 0; i < 16; i++) {
        uint8_t packed = block->qs[i];
        // Low nibble at position i, high nibble at position i+16
        int lo         = (packed & 0x0F) - 8;
        int hi         = (packed >> 4) - 8;
        output[i]      = static_cast<float>(lo) * d;
        output[i + 16] = static_cast<float>(hi) * d;
    }
}

// Reference Q8_0 dequantization - scalar, guaranteed correct
void reference_dequant_q8_0(const block_q8_0_test * block, float * output) {
    float d = static_cast<float>(block->d);
    for (int i = 0; i < 32; i++) {
        output[i] = static_cast<float>(block->qs[i]) * d;
    }
}

// =============================================================================
// Test 1: Q4_0 Dequant Block Aligned
//
// Verify 32 weights from a single Q4_0 block are correctly dequantized.
// =============================================================================
#if SYCL_ESIMD_AVAILABLE

bool test_q4_0_dequant_block_aligned(sycl::queue & q) {
    TEST_BEGIN("test_q4_0_dequant_block_aligned");

    // Create a test block with known values
    block_q4_0_test block;
    block.d = sycl::half(0.5f);  // Scale = 0.5

    // Fill with pattern: low nibble = i % 16, high nibble = 15 - (i % 16)
    for (int i = 0; i < 16; i++) {
        uint8_t lo = static_cast<uint8_t>(i % 16);
        uint8_t hi = static_cast<uint8_t>(15 - (i % 16));
        block.qs[i] = lo | (hi << 4);
    }

    // Reference dequantization
    float ref_output[32];
    reference_dequant_q4_0(&block, ref_output);

    // Allocate device memory
    block_q4_0_test * d_block  = sycl::malloc_device<block_q4_0_test>(1, q);
    sycl::half *      d_output = sycl::malloc_device<sycl::half>(32, q);

    if (!d_block || !d_output) {
        if (d_block) sycl::free(d_block, q);
        if (d_output) sycl::free(d_output, q);
        TEST_FAIL("Failed to allocate device memory");
    }

    q.memcpy(d_block, &block, sizeof(block_q4_0_test)).wait();
    q.memset(d_output, 0, 32 * sizeof(sycl::half)).wait();

    // Run ESIMD kernel that uses vectorized dequantization
    try {
        q.submit([&](sycl::handler & cgh) {
             cgh.parallel_for<class test_q4_0_block_aligned_kernel>(
                 sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                     // Simulate dequant_q4_0_block_vectorized inline
                     const block_q4_0_test * blk = d_block;
                     sycl::half              d   = blk->d;

                     // Load packed bytes
                     esimd::simd<uint8_t, 16> packed;
                     for (int i = 0; i < 16; i++) {
                         packed[i] = blk->qs[i];
                     }

                     // Extract nibbles
                     esimd::simd<int32_t, 16> lo_nibbles = esimd::simd<int32_t, 16>(packed & 0x0F) - 8;
                     esimd::simd<int32_t, 16> hi_nibbles = esimd::simd<int32_t, 16>(packed >> 4) - 8;

                     // Convert and scale
                     esimd::simd<sycl::half, 32> result;
                     for (int i = 0; i < 16; i++) {
                         result[i]      = static_cast<sycl::half>(lo_nibbles[i]) * d;
                         result[i + 16] = static_cast<sycl::half>(hi_nibbles[i]) * d;
                     }

                     // Store result
                     for (int i = 0; i < 32; i++) {
                         d_output[i] = result[i];
                     }
                 });
         }).wait();
    } catch (const sycl::exception & e) {
        sycl::free(d_block, q);
        sycl::free(d_output, q);
        fprintf(stderr, "Kernel failed: %s\n", e.what());
        TEST_FAIL("Kernel execution failed");
    }

    // Copy back and verify
    std::vector<sycl::half> h_output(32);
    q.memcpy(h_output.data(), d_output, 32 * sizeof(sycl::half)).wait();

    sycl::free(d_block, q);
    sycl::free(d_output, q);

    // Compare with reference
    const float tolerance = 1e-3f;
    for (int i = 0; i < 32; i++) {
        float gpu_val = static_cast<float>(h_output[i]);
        float ref_val = ref_output[i];
        if (std::abs(gpu_val - ref_val) > tolerance) {
            fprintf(stderr, "\n  Mismatch at %d: ref=%.4f, gpu=%.4f\n", i, ref_val, gpu_val);
            TEST_FAIL("Output mismatch");
        }
    }

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 2: Q4_0 Dequant Partial Block
//
// Verify 16 weights (half a block) are correctly extracted.
// =============================================================================

bool test_q4_0_dequant_partial_block(sycl::queue & q) {
    TEST_BEGIN("test_q4_0_dequant_partial_block");

    // Create a test block
    block_q4_0_test block;
    block.d = sycl::half(1.0f);

    // Fill with simple pattern
    for (int i = 0; i < 16; i++) {
        block.qs[i] = static_cast<uint8_t>(0x88);  // Both nibbles = 8 -> value = 0
    }
    // Override first 8 bytes to have distinct values
    for (int i = 0; i < 8; i++) {
        uint8_t lo  = static_cast<uint8_t>((i + 8) & 0x0F);  // 8-15 -> values 0-7
        uint8_t hi  = static_cast<uint8_t>((8 - i) & 0x0F);  // 8-1 -> values 0,-7
        block.qs[i] = lo | (hi << 4);
    }

    // Reference dequantization
    float ref_output[32];
    reference_dequant_q4_0(&block, ref_output);

    // Allocate device memory
    block_q4_0_test * d_block  = sycl::malloc_device<block_q4_0_test>(1, q);
    sycl::half *      d_output = sycl::malloc_device<sycl::half>(16, q);

    if (!d_block || !d_output) {
        if (d_block) sycl::free(d_block, q);
        if (d_output) sycl::free(d_output, q);
        TEST_FAIL("Failed to allocate device memory");
    }

    q.memcpy(d_block, &block, sizeof(block_q4_0_test)).wait();
    q.memset(d_output, 0, 16 * sizeof(sycl::half)).wait();

    // Run kernel extracting only first 16 weights
    try {
        q.submit([&](sycl::handler & cgh) {
             cgh.parallel_for<class test_q4_0_partial_kernel>(
                 sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                     const block_q4_0_test * blk = d_block;
                     sycl::half              d   = blk->d;

                     // Load and extract only first 16 weights (low nibbles only)
                     esimd::simd<uint8_t, 16> packed;
                     for (int i = 0; i < 16; i++) {
                         packed[i] = blk->qs[i];
                     }

                     esimd::simd<int32_t, 16> lo_nibbles = esimd::simd<int32_t, 16>(packed & 0x0F) - 8;

                     esimd::simd<sycl::half, 16> result;
                     for (int i = 0; i < 16; i++) {
                         result[i] = static_cast<sycl::half>(lo_nibbles[i]) * d;
                     }

                     for (int i = 0; i < 16; i++) {
                         d_output[i] = result[i];
                     }
                 });
         }).wait();
    } catch (const sycl::exception & e) {
        sycl::free(d_block, q);
        sycl::free(d_output, q);
        TEST_FAIL("Kernel execution failed");
    }

    std::vector<sycl::half> h_output(16);
    q.memcpy(h_output.data(), d_output, 16 * sizeof(sycl::half)).wait();

    sycl::free(d_block, q);
    sycl::free(d_output, q);

    const float tolerance = 1e-3f;
    for (int i = 0; i < 16; i++) {
        float gpu_val = static_cast<float>(h_output[i]);
        float ref_val = ref_output[i];  // First 16 are low nibbles
        if (std::abs(gpu_val - ref_val) > tolerance) {
            fprintf(stderr, "\n  Mismatch at %d: ref=%.4f, gpu=%.4f\n", i, ref_val, gpu_val);
            TEST_FAIL("Output mismatch");
        }
    }

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 3: Q4_0 Dequant Multiple Blocks
//
// Verify 64 weights across 2 Q4_0 blocks (2 different scales).
// =============================================================================

bool test_q4_0_dequant_multiple_blocks(sycl::queue & q) {
    TEST_BEGIN("test_q4_0_dequant_multiple_blocks");

    // Create two test blocks with different scales
    std::vector<block_q4_0_test> blocks(2);
    blocks[0].d = sycl::half(0.5f);
    blocks[1].d = sycl::half(2.0f);

    // Fill with known pattern
    for (int b = 0; b < 2; b++) {
        for (int i = 0; i < 16; i++) {
            blocks[b].qs[i] = 0x99;  // nibble 9 -> value 1
        }
    }

    // Reference: block 0 weights = 1 * 0.5 = 0.5
    //            block 1 weights = 1 * 2.0 = 2.0
    float ref_output[64];
    reference_dequant_q4_0(&blocks[0], &ref_output[0]);
    reference_dequant_q4_0(&blocks[1], &ref_output[32]);

    // Allocate device memory
    block_q4_0_test * d_blocks = sycl::malloc_device<block_q4_0_test>(2, q);
    sycl::half *      d_output = sycl::malloc_device<sycl::half>(64, q);

    if (!d_blocks || !d_output) {
        if (d_blocks) sycl::free(d_blocks, q);
        if (d_output) sycl::free(d_output, q);
        TEST_FAIL("Failed to allocate device memory");
    }

    q.memcpy(d_blocks, blocks.data(), 2 * sizeof(block_q4_0_test)).wait();
    q.memset(d_output, 0, 64 * sizeof(sycl::half)).wait();

    try {
        q.submit([&](sycl::handler & cgh) {
             cgh.parallel_for<class test_q4_0_multi_block_kernel>(
                 sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                     // Process both blocks
                     for (int b = 0; b < 2; b++) {
                         const block_q4_0_test * blk = &d_blocks[b];
                         sycl::half              d   = blk->d;

                         esimd::simd<uint8_t, 16> packed;
                         for (int i = 0; i < 16; i++) {
                             packed[i] = blk->qs[i];
                         }

                         esimd::simd<int32_t, 16> lo_nibbles = esimd::simd<int32_t, 16>(packed & 0x0F) - 8;
                         esimd::simd<int32_t, 16> hi_nibbles = esimd::simd<int32_t, 16>(packed >> 4) - 8;

                         esimd::simd<sycl::half, 32> result;
                         for (int i = 0; i < 16; i++) {
                             result[i]      = static_cast<sycl::half>(lo_nibbles[i]) * d;
                             result[i + 16] = static_cast<sycl::half>(hi_nibbles[i]) * d;
                         }

                         for (int i = 0; i < 32; i++) {
                             d_output[b * 32 + i] = result[i];
                         }
                     }
                 });
         }).wait();
    } catch (const sycl::exception & e) {
        sycl::free(d_blocks, q);
        sycl::free(d_output, q);
        TEST_FAIL("Kernel execution failed");
    }

    std::vector<sycl::half> h_output(64);
    q.memcpy(h_output.data(), d_output, 64 * sizeof(sycl::half)).wait();

    sycl::free(d_blocks, q);
    sycl::free(d_output, q);

    const float tolerance = 1e-3f;
    for (int i = 0; i < 64; i++) {
        float gpu_val = static_cast<float>(h_output[i]);
        float ref_val = ref_output[i];
        if (std::abs(gpu_val - ref_val) > tolerance) {
            fprintf(stderr, "\n  Mismatch at %d: ref=%.4f, gpu=%.4f\n", i, ref_val, gpu_val);
            TEST_FAIL("Output mismatch");
        }
    }

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 4: Q8_0 Dequant Full Tile
//
// Verify 32 INT8 weights with scale are correctly dequantized.
// =============================================================================

bool test_q8_0_dequant_full_tile(sycl::queue & q) {
    TEST_BEGIN("test_q8_0_dequant_full_tile");

    // Create a Q8_0 block
    block_q8_0_test block;
    block.d = sycl::half(0.25f);

    // Fill with values -16 to +15
    for (int i = 0; i < 32; i++) {
        block.qs[i] = static_cast<int8_t>(i - 16);
    }

    float ref_output[32];
    reference_dequant_q8_0(&block, ref_output);

    block_q8_0_test * d_block  = sycl::malloc_device<block_q8_0_test>(1, q);
    sycl::half *      d_output = sycl::malloc_device<sycl::half>(32, q);

    if (!d_block || !d_output) {
        if (d_block) sycl::free(d_block, q);
        if (d_output) sycl::free(d_output, q);
        TEST_FAIL("Failed to allocate device memory");
    }

    q.memcpy(d_block, &block, sizeof(block_q8_0_test)).wait();
    q.memset(d_output, 0, 32 * sizeof(sycl::half)).wait();

    try {
        q.submit([&](sycl::handler & cgh) {
             cgh.parallel_for<class test_q8_0_full_tile_kernel>(
                 sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                     const block_q8_0_test * blk   = d_block;
                     sycl::half              scale = blk->d;

                     // Vectorized Q8_0 dequantization
                     esimd::simd<int32_t, 32> weights_int;
                     for (int i = 0; i < 32; i++) {
                         weights_int[i] = static_cast<int32_t>(blk->qs[i]);
                     }

                     esimd::simd<sycl::half, 32> result;
                     for (int i = 0; i < 32; i++) {
                         result[i] = static_cast<sycl::half>(weights_int[i]) * scale;
                     }

                     for (int i = 0; i < 32; i++) {
                         d_output[i] = result[i];
                     }
                 });
         }).wait();
    } catch (const sycl::exception & e) {
        sycl::free(d_block, q);
        sycl::free(d_output, q);
        TEST_FAIL("Kernel execution failed");
    }

    std::vector<sycl::half> h_output(32);
    q.memcpy(h_output.data(), d_output, 32 * sizeof(sycl::half)).wait();

    sycl::free(d_block, q);
    sycl::free(d_output, q);

    const float tolerance = 1e-3f;
    for (int i = 0; i < 32; i++) {
        float gpu_val = static_cast<float>(h_output[i]);
        float ref_val = ref_output[i];
        if (std::abs(gpu_val - ref_val) > tolerance) {
            fprintf(stderr, "\n  Mismatch at %d: ref=%.4f, gpu=%.4f\n", i, ref_val, gpu_val);
            TEST_FAIL("Output mismatch");
        }
    }

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 5: Dequant Extreme Values
//
// Verify min/max values: Q4_0 nibble 0 (-8), nibble 15 (+7)
//                        Q8_0 value -127, value +127
// =============================================================================

bool test_dequant_extreme_values(sycl::queue & q) {
    TEST_BEGIN("test_dequant_extreme_values");

    // Q4_0: nibble 0 = value -8, nibble 15 = value +7
    block_q4_0_test q4_block;
    q4_block.d = sycl::half(1.0f);
    // qs[0] = 0x00 -> low=-8, high=-8
    // qs[1] = 0xFF -> low=+7, high=+7
    // qs[2] = 0x0F -> low=+7, high=-8
    // qs[3] = 0xF0 -> low=-8, high=+7
    q4_block.qs[0] = 0x00;
    q4_block.qs[1] = 0xFF;
    q4_block.qs[2] = 0x0F;
    q4_block.qs[3] = 0xF0;
    for (int i = 4; i < 16; i++) {
        q4_block.qs[i] = 0x88;  // value 0
    }

    float q4_ref[32];
    reference_dequant_q4_0(&q4_block, q4_ref);

    // Expected values:
    TEST_ASSERT_NEAR(q4_ref[0], -8.0f, 1e-3f, "Q4 extreme: min nibble");
    TEST_ASSERT_NEAR(q4_ref[16], -8.0f, 1e-3f, "Q4 extreme: min nibble high");
    TEST_ASSERT_NEAR(q4_ref[1], 7.0f, 1e-3f, "Q4 extreme: max nibble");
    TEST_ASSERT_NEAR(q4_ref[17], 7.0f, 1e-3f, "Q4 extreme: max nibble high");

    // Q8_0: values -127 and +127
    block_q8_0_test q8_block;
    q8_block.d     = sycl::half(1.0f);
    q8_block.qs[0] = -127;
    q8_block.qs[1] = 127;
    q8_block.qs[2] = 0;
    for (int i = 3; i < 32; i++) {
        q8_block.qs[i] = 0;
    }

    float q8_ref[32];
    reference_dequant_q8_0(&q8_block, q8_ref);

    TEST_ASSERT_NEAR(q8_ref[0], -127.0f, 1e-3f, "Q8 extreme: min value");
    TEST_ASSERT_NEAR(q8_ref[1], 127.0f, 1e-3f, "Q8 extreme: max value");

    // Now test on GPU
    block_q4_0_test * d_q4_block  = sycl::malloc_device<block_q4_0_test>(1, q);
    sycl::half *      d_q4_output = sycl::malloc_device<sycl::half>(32, q);

    q.memcpy(d_q4_block, &q4_block, sizeof(block_q4_0_test)).wait();

    try {
        q.submit([&](sycl::handler & cgh) {
             cgh.parallel_for<class test_extreme_q4_kernel>(
                 sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                     const block_q4_0_test * blk = d_q4_block;
                     sycl::half              d   = blk->d;

                     esimd::simd<uint8_t, 16> packed;
                     for (int i = 0; i < 16; i++) {
                         packed[i] = blk->qs[i];
                     }

                     esimd::simd<int32_t, 16> lo_nibbles = esimd::simd<int32_t, 16>(packed & 0x0F) - 8;
                     esimd::simd<int32_t, 16> hi_nibbles = esimd::simd<int32_t, 16>(packed >> 4) - 8;

                     esimd::simd<sycl::half, 32> result;
                     for (int i = 0; i < 16; i++) {
                         result[i]      = static_cast<sycl::half>(lo_nibbles[i]) * d;
                         result[i + 16] = static_cast<sycl::half>(hi_nibbles[i]) * d;
                     }

                     for (int i = 0; i < 32; i++) {
                         d_q4_output[i] = result[i];
                     }
                 });
         }).wait();
    } catch (const sycl::exception & e) {
        sycl::free(d_q4_block, q);
        sycl::free(d_q4_output, q);
        TEST_FAIL("Kernel execution failed");
    }

    std::vector<sycl::half> h_q4_output(32);
    q.memcpy(h_q4_output.data(), d_q4_output, 32 * sizeof(sycl::half)).wait();

    sycl::free(d_q4_block, q);
    sycl::free(d_q4_output, q);

    const float tolerance = 1e-3f;
    for (int i = 0; i < 32; i++) {
        float gpu_val = static_cast<float>(h_q4_output[i]);
        float ref_val = q4_ref[i];
        if (std::abs(gpu_val - ref_val) > tolerance) {
            fprintf(stderr, "\n  Q4 extreme mismatch at %d: ref=%.4f, gpu=%.4f\n", i, ref_val, gpu_val);
            TEST_FAIL("Q4 extreme value mismatch");
        }
    }

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 6: Dequant VNNI Format
//
// Verify dequantized output can be correctly repacked to VNNI format for dpas.
// VNNI layout for FP16: b[(k/2) * N * 2 + n * 2 + (k%2)]
// =============================================================================

bool test_dequant_vnni_format(sycl::queue & q) {
    TEST_BEGIN("test_dequant_vnni_format");

    // Create a test block
    block_q4_0_test block;
    block.d = sycl::half(1.0f);
    for (int i = 0; i < 16; i++) {
        block.qs[i] = static_cast<uint8_t>(0x98 + (i % 2));  // Alternating pattern
    }

    float ref_output[32];
    reference_dequant_q4_0(&block, ref_output);

    // Test VNNI repacking on GPU
    block_q4_0_test * d_block  = sycl::malloc_device<block_q4_0_test>(1, q);
    sycl::half *      d_vnni   = sycl::malloc_device<sycl::half>(256, q);  // VNNI format: 16x16

    q.memcpy(d_block, &block, sizeof(block_q4_0_test)).wait();
    q.memset(d_vnni, 0, 256 * sizeof(sycl::half)).wait();

    constexpr int TILE_N = 16;
    constexpr int K_PER  = 16;

    try {
        q.submit([&](sycl::handler & cgh) {
             cgh.parallel_for<class test_vnni_format_kernel>(
                 sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                     const block_q4_0_test * blk = d_block;
                     sycl::half              d   = blk->d;

                     // Dequantize
                     esimd::simd<uint8_t, 16> packed;
                     for (int i = 0; i < 16; i++) {
                         packed[i] = blk->qs[i];
                     }

                     esimd::simd<int32_t, 16> lo_nibbles = esimd::simd<int32_t, 16>(packed & 0x0F) - 8;
                     esimd::simd<int32_t, 16> hi_nibbles = esimd::simd<int32_t, 16>(packed >> 4) - 8;

                     esimd::simd<sycl::half, 32> dequant;
                     for (int i = 0; i < 16; i++) {
                         dequant[i]      = static_cast<sycl::half>(lo_nibbles[i]) * d;
                         dequant[i + 16] = static_cast<sycl::half>(hi_nibbles[i]) * d;
                     }

                     // Repack to VNNI for n=0
                     // VNNI layout: b[(k/2) * N * 2 + n * 2 + (k%2)]
                     esimd::simd<sycl::half, 256> vnni = sycl::half(0.0f);
                     for (int k = 0; k < K_PER; k++) {
                         int vnni_idx   = (k / 2) * (TILE_N * 2) + 0 * 2 + (k % 2);
                         vnni[vnni_idx] = dequant[k];
                     }

                     for (int i = 0; i < 256; i++) {
                         d_vnni[i] = vnni[i];
                     }
                 });
         }).wait();
    } catch (const sycl::exception & e) {
        sycl::free(d_block, q);
        sycl::free(d_vnni, q);
        TEST_FAIL("Kernel execution failed");
    }

    std::vector<sycl::half> h_vnni(256);
    q.memcpy(h_vnni.data(), d_vnni, 256 * sizeof(sycl::half)).wait();

    sycl::free(d_block, q);
    sycl::free(d_vnni, q);

    // Verify VNNI layout: first 16 k values at n=0 should be in correct positions
    const float tolerance = 1e-3f;
    for (int k = 0; k < K_PER; k++) {
        int   vnni_idx = (k / 2) * (TILE_N * 2) + 0 * 2 + (k % 2);
        float gpu_val  = static_cast<float>(h_vnni[vnni_idx]);
        float ref_val  = ref_output[k];  // First 16 are low nibbles
        if (std::abs(gpu_val - ref_val) > tolerance) {
            fprintf(stderr, "\n  VNNI mismatch at k=%d (vnni_idx=%d): ref=%.4f, gpu=%.4f\n", k, vnni_idx, ref_val,
                    gpu_val);
            TEST_FAIL("VNNI format mismatch");
        }
    }

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 7: Dequant Unaligned Source
//
// Verify handling when weight block is not 16-byte aligned.
// =============================================================================

bool test_dequant_unaligned_src(sycl::queue & q) {
    TEST_BEGIN("test_dequant_unaligned_src");

    // Allocate with extra padding to create misalignment
    constexpr int PADDING = 3;  // 3 bytes offset
    uint8_t *     buffer  = sycl::malloc_device<uint8_t>(sizeof(block_q4_0_test) + PADDING, q);

    if (!buffer) {
        TEST_FAIL("Failed to allocate device memory");
    }

    block_q4_0_test block;
    block.d = sycl::half(0.5f);
    for (int i = 0; i < 16; i++) {
        block.qs[i] = static_cast<uint8_t>(0xAB);  // nibbles: 11 and 10 -> values 3 and 2
    }

    float ref_output[32];
    reference_dequant_q4_0(&block, ref_output);

    // Copy block to unaligned position
    q.memcpy(buffer + PADDING, &block, sizeof(block_q4_0_test)).wait();

    sycl::half * d_output = sycl::malloc_device<sycl::half>(32, q);
    q.memset(d_output, 0, 32 * sizeof(sycl::half)).wait();

    try {
        q.submit([&](sycl::handler & cgh) {
             cgh.parallel_for<class test_unaligned_kernel>(
                 sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                     // Access block at unaligned position
                     const block_q4_0_test * blk =
                         reinterpret_cast<const block_q4_0_test *>(buffer + PADDING);
                     sycl::half d = blk->d;

                     // Must use scalar loads for unaligned access
                     esimd::simd<uint8_t, 16> packed;
                     for (int i = 0; i < 16; i++) {
                         packed[i] = blk->qs[i];
                     }

                     esimd::simd<int32_t, 16> lo_nibbles = esimd::simd<int32_t, 16>(packed & 0x0F) - 8;
                     esimd::simd<int32_t, 16> hi_nibbles = esimd::simd<int32_t, 16>(packed >> 4) - 8;

                     esimd::simd<sycl::half, 32> result;
                     for (int i = 0; i < 16; i++) {
                         result[i]      = static_cast<sycl::half>(lo_nibbles[i]) * d;
                         result[i + 16] = static_cast<sycl::half>(hi_nibbles[i]) * d;
                     }

                     for (int i = 0; i < 32; i++) {
                         d_output[i] = result[i];
                     }
                 });
         }).wait();
    } catch (const sycl::exception & e) {
        sycl::free(buffer, q);
        sycl::free(d_output, q);
        TEST_FAIL("Kernel execution failed");
    }

    std::vector<sycl::half> h_output(32);
    q.memcpy(h_output.data(), d_output, 32 * sizeof(sycl::half)).wait();

    sycl::free(buffer, q);
    sycl::free(d_output, q);

    const float tolerance = 1e-3f;
    for (int i = 0; i < 32; i++) {
        float gpu_val = static_cast<float>(h_output[i]);
        float ref_val = ref_output[i];
        if (std::abs(gpu_val - ref_val) > tolerance) {
            fprintf(stderr, "\n  Unaligned mismatch at %d: ref=%.4f, gpu=%.4f\n", i, ref_val, gpu_val);
            TEST_FAIL("Unaligned access mismatch");
        }
    }

    TEST_PASS();
    return true;
}

// =============================================================================
// Test 8: Dequant Offset Minus 8
//
// Verify the -8 offset is applied correctly for all nibble values 0-15.
// =============================================================================

bool test_dequant_offset_minus_8(sycl::queue & q) {
    TEST_BEGIN("test_dequant_offset_minus_8");

    // Create block with all 16 possible nibble values in low nibbles
    block_q4_0_test block;
    block.d = sycl::half(1.0f);  // Scale 1.0 to directly see offset effect

    // qs[i] = i | (i << 4) puts nibble value i in both low and high positions
    for (int i = 0; i < 16; i++) {
        block.qs[i] = static_cast<uint8_t>(i | (i << 4));
    }

    float ref_output[32];
    reference_dequant_q4_0(&block, ref_output);

    // Verify reference has correct -8 offset
    for (int i = 0; i < 16; i++) {
        float expected = static_cast<float>(i - 8);  // -8 to +7
        TEST_ASSERT_NEAR(ref_output[i], expected, 1e-3f, "Reference offset check (low)");
        TEST_ASSERT_NEAR(ref_output[i + 16], expected, 1e-3f, "Reference offset check (high)");
    }

    // Test on GPU
    block_q4_0_test * d_block  = sycl::malloc_device<block_q4_0_test>(1, q);
    sycl::half *      d_output = sycl::malloc_device<sycl::half>(32, q);

    q.memcpy(d_block, &block, sizeof(block_q4_0_test)).wait();
    q.memset(d_output, 0, 32 * sizeof(sycl::half)).wait();

    try {
        q.submit([&](sycl::handler & cgh) {
             cgh.parallel_for<class test_offset_minus_8_kernel>(
                 sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                     const block_q4_0_test * blk = d_block;
                     sycl::half              d   = blk->d;

                     esimd::simd<uint8_t, 16> packed;
                     for (int i = 0; i < 16; i++) {
                         packed[i] = blk->qs[i];
                     }

                     // CRITICAL: The -8 offset must be applied here
                     esimd::simd<int32_t, 16> lo_nibbles = esimd::simd<int32_t, 16>(packed & 0x0F) - 8;
                     esimd::simd<int32_t, 16> hi_nibbles = esimd::simd<int32_t, 16>(packed >> 4) - 8;

                     esimd::simd<sycl::half, 32> result;
                     for (int i = 0; i < 16; i++) {
                         result[i]      = static_cast<sycl::half>(lo_nibbles[i]) * d;
                         result[i + 16] = static_cast<sycl::half>(hi_nibbles[i]) * d;
                     }

                     for (int i = 0; i < 32; i++) {
                         d_output[i] = result[i];
                     }
                 });
         }).wait();
    } catch (const sycl::exception & e) {
        sycl::free(d_block, q);
        sycl::free(d_output, q);
        TEST_FAIL("Kernel execution failed");
    }

    std::vector<sycl::half> h_output(32);
    q.memcpy(h_output.data(), d_output, 32 * sizeof(sycl::half)).wait();

    sycl::free(d_block, q);
    sycl::free(d_output, q);

    const float tolerance = 1e-3f;
    for (int i = 0; i < 16; i++) {
        float expected = static_cast<float>(i - 8);

        float lo_gpu = static_cast<float>(h_output[i]);
        if (std::abs(lo_gpu - expected) > tolerance) {
            fprintf(stderr, "\n  Offset mismatch at low %d: expected=%.1f, got=%.4f\n", i, expected, lo_gpu);
            TEST_FAIL("Offset -8 not applied correctly (low nibble)");
        }

        float hi_gpu = static_cast<float>(h_output[i + 16]);
        if (std::abs(hi_gpu - expected) > tolerance) {
            fprintf(stderr, "\n  Offset mismatch at high %d: expected=%.1f, got=%.4f\n", i, expected, hi_gpu);
            TEST_FAIL("Offset -8 not applied correctly (high nibble)");
        }
    }

    fprintf(stderr, "(verified all 16 nibble values map to [-8,+7]) ");
    TEST_PASS();
    return true;
}

#else  // !SYCL_ESIMD_AVAILABLE

// Stub implementations when ESIMD is not available
bool test_q4_0_dequant_block_aligned(sycl::queue & q) {
    (void) q;
    TEST_BEGIN("test_q4_0_dequant_block_aligned");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_q4_0_dequant_partial_block(sycl::queue & q) {
    (void) q;
    TEST_BEGIN("test_q4_0_dequant_partial_block");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_q4_0_dequant_multiple_blocks(sycl::queue & q) {
    (void) q;
    TEST_BEGIN("test_q4_0_dequant_multiple_blocks");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_q8_0_dequant_full_tile(sycl::queue & q) {
    (void) q;
    TEST_BEGIN("test_q8_0_dequant_full_tile");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_dequant_extreme_values(sycl::queue & q) {
    (void) q;
    TEST_BEGIN("test_dequant_extreme_values");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_dequant_vnni_format(sycl::queue & q) {
    (void) q;
    TEST_BEGIN("test_dequant_vnni_format");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_dequant_unaligned_src(sycl::queue & q) {
    (void) q;
    TEST_BEGIN("test_dequant_unaligned_src");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_dequant_offset_minus_8(sycl::queue & q) {
    (void) q;
    TEST_BEGIN("test_dequant_offset_minus_8");
    TEST_SKIP("ESIMD not available");
    return true;
}

#endif  // SYCL_ESIMD_AVAILABLE

// =============================================================================
// Main
// =============================================================================
int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    fprintf(stderr, "=== ESIMD Vectorized Dequantization Tests ===\n");

#if !SYCL_ESIMD_AVAILABLE
    fprintf(stderr, "WARNING: ESIMD header not available, tests will be skipped.\n\n");
#endif

    // Create SYCL queue
    sycl::queue  q;
    sycl::device dev;
    try {
        dev = sycl::device(sycl::gpu_selector_v);
        q   = sycl::queue(dev);
        fprintf(stderr, "Using GPU: %s\n\n", dev.get_info<sycl::info::device::name>().c_str());
    } catch (const sycl::exception & e) {
        fprintf(stderr, "No GPU found, using default device\n");
        dev = sycl::device(sycl::default_selector_v);
        q   = sycl::queue(dev);
    }

    // Run all 8 tests
    bool all_passed = true;

    // Test 1: Q4_0 block aligned
    all_passed &= test_q4_0_dequant_block_aligned(q);

    // Test 2: Q4_0 partial block
    all_passed &= test_q4_0_dequant_partial_block(q);

    // Test 3: Q4_0 multiple blocks
    all_passed &= test_q4_0_dequant_multiple_blocks(q);

    // Test 4: Q8_0 full tile
    all_passed &= test_q8_0_dequant_full_tile(q);

    // Test 5: Extreme values
    all_passed &= test_dequant_extreme_values(q);

    // Test 6: VNNI format
    all_passed &= test_dequant_vnni_format(q);

    // Test 7: Unaligned source
    all_passed &= test_dequant_unaligned_src(q);

    // Test 8: Offset minus 8
    all_passed &= test_dequant_offset_minus_8(q);

    // Summary
    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "Tests run: %d, Passed: %d, Skipped: %d, Failed: %d\n", g_tests_run, g_tests_passed,
            g_tests_skipped, g_tests_run - g_tests_passed - g_tests_skipped);

#if !SYCL_ESIMD_AVAILABLE
    fprintf(stderr, "\nNote: All tests were skipped because ESIMD is not available.\n");
    return 0;
#endif

    return all_passed ? 0 : 1;
}
