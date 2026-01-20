//
// Test: ESIMD Basic Operations for XMX
//
// Tests basic ESIMD operations that are prerequisites for XMX kernels:
// - simd vector operations
// - block loads/stores
// - MXFP4 nibble unpacking
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sycl/sycl.hpp>
#include <vector>

// Check for ESIMD support
#if __has_include(<sycl/ext/intel/esimd.hpp>)
#    define SYCL_ESIMD_AVAILABLE 1
#    include <sycl/ext/intel/esimd.hpp>
namespace esimd = sycl::ext::intel::esimd;
#else
#    define SYCL_ESIMD_AVAILABLE 0
#endif

// Include XMX ESIMD common header (will be created)
#include "../xmx-esimd-common.hpp"

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

#define TEST_ASSERT_EQ(actual, expected, msg)                                                             \
    do {                                                                                                  \
        if ((actual) != (expected)) {                                                                     \
            fprintf(stderr, "FAILED: %s (expected %d, got %d)\n", msg, (int) (expected), (int) (actual)); \
            return false;                                                                                 \
        }                                                                                                 \
    } while (0)

#define TEST_ASSERT_NEAR(actual, expected, tol, msg)                                                      \
    do {                                                                                                  \
        float diff = std::abs((float) (actual) - (float) (expected));                                     \
        if (diff > (tol)) {                                                                               \
            fprintf(stderr, "FAILED: %s (expected %.4f, got %.4f, diff=%.4f)\n", msg, (float) (expected), \
                    (float) (actual), diff);                                                              \
            return false;                                                                                 \
        }                                                                                                 \
    } while (0)

// =============================================================================
// Test 1: Basic simd<int, 16> vector operations compile and run
// =============================================================================
#if SYCL_ESIMD_AVAILABLE

// ESIMD kernel for basic vector add
class esimd_vector_add_kernel;

bool test_esimd_simd_vector_ops(sycl::queue & q) {
    TEST_BEGIN("ESIMD simd<int, 16> vector operations");

    constexpr int N = 16;

    // Allocate USM memory
    int * a   = sycl::malloc_shared<int>(N, q);
    int * b   = sycl::malloc_shared<int>(N, q);
    int * out = sycl::malloc_shared<int>(N, q);

    if (!a || !b || !out) {
        if (a) {
            sycl::free(a, q);
        }
        if (b) {
            sycl::free(b, q);
        }
        if (out) {
            sycl::free(out, q);
        }
        TEST_FAIL("Failed to allocate USM memory");
    }

    // Initialize
    for (int i = 0; i < N; i++) {
        a[i]   = i;
        b[i]   = i * 2;
        out[i] = 0;
    }

    // Submit ESIMD kernel
    try {
        q.submit([&](sycl::handler & cgh) {
             cgh.single_task<esimd_vector_add_kernel>([=]() SYCL_ESIMD_KERNEL {
                 esimd::simd<int, N> va;
                 esimd::simd<int, N> vb;

                 // Load vectors
                 va.copy_from(a);
                 vb.copy_from(b);

                 // Add
                 esimd::simd<int, N> result = va + vb;

                 // Store
                 result.copy_to(out);
             });
         }).wait();
    } catch (const sycl::exception & e) {
        sycl::free(a, q);
        sycl::free(b, q);
        sycl::free(out, q);
        fprintf(stderr, "ESIMD kernel failed: %s\n", e.what());
        TEST_FAIL("ESIMD kernel execution failed");
    }

    // Verify
    bool correct = true;
    for (int i = 0; i < N && correct; i++) {
        if (out[i] != a[i] + b[i]) {
            fprintf(stderr, "Mismatch at %d: expected %d, got %d\n", i, a[i] + b[i], out[i]);
            correct = false;
        }
    }

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(out, q);

    TEST_ASSERT(correct, "Vector addition results incorrect");
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 2: ESIMD simd<float, 32> operations (typical XMX accumulator size)
// =============================================================================
class esimd_float32_kernel;

bool test_esimd_simd_float32(sycl::queue & q) {
    TEST_BEGIN("ESIMD simd<float, 32> accumulator operations");

    constexpr int N = 32;

    float * a   = sycl::malloc_shared<float>(N, q);
    float * b   = sycl::malloc_shared<float>(N, q);
    float * out = sycl::malloc_shared<float>(N, q);

    if (!a || !b || !out) {
        if (a) {
            sycl::free(a, q);
        }
        if (b) {
            sycl::free(b, q);
        }
        if (out) {
            sycl::free(out, q);
        }
        TEST_FAIL("Failed to allocate USM memory");
    }

    for (int i = 0; i < N; i++) {
        a[i]   = static_cast<float>(i) * 0.5f;
        b[i]   = static_cast<float>(i) * 0.25f;
        out[i] = 0.0f;
    }

    try {
        q.submit([&](sycl::handler & cgh) {
             cgh.single_task<esimd_float32_kernel>([=]() SYCL_ESIMD_KERNEL {
                 esimd::simd<float, N> va;
                 esimd::simd<float, N> vb;

                 va.copy_from(a);
                 vb.copy_from(b);

                 // Fused multiply-add pattern (common in GEMM)
                 esimd::simd<float, N> result = va * vb + va;

                 result.copy_to(out);
             });
         }).wait();
    } catch (const sycl::exception & e) {
        sycl::free(a, q);
        sycl::free(b, q);
        sycl::free(out, q);
        fprintf(stderr, "ESIMD kernel failed: %s\n", e.what());
        TEST_FAIL("ESIMD kernel execution failed");
    }

    bool correct = true;
    for (int i = 0; i < N && correct; i++) {
        float expected = a[i] * b[i] + a[i];
        float diff     = std::abs(out[i] - expected);
        if (diff > 1e-5f) {
            fprintf(stderr, "Mismatch at %d: expected %.6f, got %.6f\n", i, expected, out[i]);
            correct = false;
        }
    }

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(out, q);

    TEST_ASSERT(correct, "Float32 FMA results incorrect");
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 3: MXFP4 nibble unpacking (critical for MoE path)
// =============================================================================
class esimd_nibble_unpack_kernel;

bool test_esimd_nibble_unpack(sycl::queue & q) {
    TEST_BEGIN("ESIMD nibble unpacking for MXFP4");

    // MXFP4: 16 packed bytes -> 32 4-bit values
    constexpr int PACKED_SIZE   = 16;
    constexpr int UNPACKED_SIZE = 32;

    uint8_t * packed   = sycl::malloc_shared<uint8_t>(PACKED_SIZE, q);
    int8_t *  unpacked = sycl::malloc_shared<int8_t>(UNPACKED_SIZE, q);

    if (!packed || !unpacked) {
        if (packed) {
            sycl::free(packed, q);
        }
        if (unpacked) {
            sycl::free(unpacked, q);
        }
        TEST_FAIL("Failed to allocate USM memory");
    }

    // Initialize with known pattern
    // Each byte: low nibble = i, high nibble = 15 - i
    for (int i = 0; i < PACKED_SIZE; i++) {
        uint8_t lo = i & 0x0F;
        uint8_t hi = (15 - i) & 0x0F;
        packed[i]  = (hi << 4) | lo;
    }
    memset(unpacked, 0, UNPACKED_SIZE);

    try {
        q.submit([&](sycl::handler & cgh) {
             cgh.single_task<esimd_nibble_unpack_kernel>([=]() SYCL_ESIMD_KERNEL {
                 // Use the helper from xmx-esimd-common.hpp
                 esimd::simd<uint8_t, PACKED_SIZE> packed_data;
                 packed_data.copy_from(packed);

                 // Unpack nibbles using the XMX ESIMD helper
                 esimd::simd<int8_t, UNPACKED_SIZE> result = ggml_sycl_xmx::unpack_nibbles_mxfp4(packed_data);

                 result.copy_to(unpacked);
             });
         }).wait();
    } catch (const sycl::exception & e) {
        sycl::free(packed, q);
        sycl::free(unpacked, q);
        fprintf(stderr, "ESIMD kernel failed: %s\n", e.what());
        TEST_FAIL("ESIMD kernel execution failed");
    }

    // Verify unpacking
    // Expected: unpacked[0..15] = low nibbles, unpacked[16..31] = high nibbles
    bool correct = true;
    for (int i = 0; i < PACKED_SIZE && correct; i++) {
        int8_t expected_lo = i & 0x0F;
        int8_t expected_hi = (15 - i) & 0x0F;

        if (unpacked[i] != expected_lo) {
            fprintf(stderr, "Low nibble mismatch at %d: expected %d, got %d\n", i, expected_lo, unpacked[i]);
            correct = false;
        }
        if (unpacked[i + PACKED_SIZE] != expected_hi) {
            fprintf(stderr, "High nibble mismatch at %d: expected %d, got %d\n", i + PACKED_SIZE, expected_hi,
                    unpacked[i + PACKED_SIZE]);
            correct = false;
        }
    }

    sycl::free(packed, q);
    sycl::free(unpacked, q);

    TEST_ASSERT(correct, "Nibble unpacking results incorrect");
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 4: ESIMD horizontal sum reduction (used in dot products)
// =============================================================================
class esimd_hsum_kernel;

bool test_esimd_horizontal_sum(sycl::queue & q) {
    TEST_BEGIN("ESIMD horizontal sum reduction");

    constexpr int N = 32;

    float * input  = sycl::malloc_shared<float>(N, q);
    float * output = sycl::malloc_shared<float>(1, q);

    if (!input || !output) {
        if (input) {
            sycl::free(input, q);
        }
        if (output) {
            sycl::free(output, q);
        }
        TEST_FAIL("Failed to allocate USM memory");
    }

    float expected_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        input[i] = static_cast<float>(i + 1);
        expected_sum += input[i];
    }
    output[0] = 0.0f;

    try {
        q.submit([&](sycl::handler & cgh) {
             cgh.single_task<esimd_hsum_kernel>([=]() SYCL_ESIMD_KERNEL {
                 esimd::simd<float, N> v;
                 v.copy_from(input);

                 // Use the horizontal sum helper from xmx-esimd-common.hpp
                 float sum = ggml_sycl_xmx::esimd_hsum<N>(v);

                 *output = sum;
             });
         }).wait();
    } catch (const sycl::exception & e) {
        sycl::free(input, q);
        sycl::free(output, q);
        fprintf(stderr, "ESIMD kernel failed: %s\n", e.what());
        TEST_FAIL("ESIMD kernel execution failed");
    }

    float diff    = std::abs(output[0] - expected_sum);
    bool  correct = (diff < 1e-3f);  // Allow small floating point error

    fprintf(stderr, "(expected=%.1f, got=%.1f) ", expected_sum, output[0]);

    sycl::free(input, q);
    sycl::free(output, q);

    TEST_ASSERT(correct, "Horizontal sum result incorrect");
    TEST_PASS();
    return true;
}

// =============================================================================
// Test 5: ESIMD block load/store operations
// =============================================================================
class esimd_block_load_store_kernel;

bool test_esimd_block_load_store(sycl::queue & q) {
    TEST_BEGIN("ESIMD block load/store operations");

    constexpr int N = 64;

    float * src = sycl::malloc_shared<float>(N, q);
    float * dst = sycl::malloc_shared<float>(N, q);

    if (!src || !dst) {
        if (src) {
            sycl::free(src, q);
        }
        if (dst) {
            sycl::free(dst, q);
        }
        TEST_FAIL("Failed to allocate USM memory");
    }

    for (int i = 0; i < N; i++) {
        src[i] = static_cast<float>(i);
        dst[i] = 0.0f;
    }

    try {
        q.submit([&](sycl::handler & cgh) {
             cgh.single_task<esimd_block_load_store_kernel>([=]() SYCL_ESIMD_KERNEL {
                 // Block load 32 elements at a time
                 esimd::simd<float, 32> block0;
                 esimd::simd<float, 32> block1;

                 block0.copy_from(src);
                 block1.copy_from(src + 32);

                 // Apply some operation (multiply by 2)
                 block0 = block0 * 2.0f;
                 block1 = block1 * 2.0f;

                 // Block store
                 block0.copy_to(dst);
                 block1.copy_to(dst + 32);
             });
         }).wait();
    } catch (const sycl::exception & e) {
        sycl::free(src, q);
        sycl::free(dst, q);
        fprintf(stderr, "ESIMD kernel failed: %s\n", e.what());
        TEST_FAIL("ESIMD kernel execution failed");
    }

    bool correct = true;
    for (int i = 0; i < N && correct; i++) {
        // Note: src was not modified, so we compare against original value * 2
        float orig_expected = static_cast<float>(i) * 2.0f;
        if (std::abs(dst[i] - orig_expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at %d: expected %.1f, got %.1f\n", i, orig_expected, dst[i]);
            correct = false;
        }
    }

    sycl::free(src, q);
    sycl::free(dst, q);

    TEST_ASSERT(correct, "Block load/store results incorrect");
    TEST_PASS();
    return true;
}

#else   // !SYCL_ESIMD_AVAILABLE

bool test_esimd_simd_vector_ops(sycl::queue & q) {
    (void) q;
    TEST_BEGIN("ESIMD simd<int, 16> vector operations");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_esimd_simd_float32(sycl::queue & q) {
    (void) q;
    TEST_BEGIN("ESIMD simd<float, 32> accumulator operations");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_esimd_nibble_unpack(sycl::queue & q) {
    (void) q;
    TEST_BEGIN("ESIMD nibble unpacking for MXFP4");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_esimd_horizontal_sum(sycl::queue & q) {
    (void) q;
    TEST_BEGIN("ESIMD horizontal sum reduction");
    TEST_SKIP("ESIMD not available");
    return true;
}

bool test_esimd_block_load_store(sycl::queue & q) {
    (void) q;
    TEST_BEGIN("ESIMD block load/store operations");
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

    fprintf(stderr, "=== ESIMD Basic Operations Tests ===\n");

#if !SYCL_ESIMD_AVAILABLE
    fprintf(stderr, "WARNING: ESIMD header not available, tests will be skipped.\n\n");
#endif

    // Create SYCL queue
    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v);
        fprintf(stderr, "Using GPU: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    } catch (const sycl::exception & e) {
        fprintf(stderr, "No GPU found, using default device\n");
        q = sycl::queue(sycl::default_selector_v);
    }

    fprintf(stderr, "ESIMD available: %s\n\n", SYCL_ESIMD_AVAILABLE ? "YES" : "NO");

    // Run tests
    bool all_passed = true;

    all_passed &= test_esimd_simd_vector_ops(q);
    all_passed &= test_esimd_simd_float32(q);
    all_passed &= test_esimd_nibble_unpack(q);
    all_passed &= test_esimd_horizontal_sum(q);
    all_passed &= test_esimd_block_load_store(q);

    // Summary
    fprintf(stderr, "\n=== Summary ===\n");
    fprintf(stderr, "Tests run: %d, Passed: %d, Skipped: %d, Failed: %d\n", g_tests_run, g_tests_passed,
            g_tests_skipped, g_tests_run - g_tests_passed - g_tests_skipped);

#if !SYCL_ESIMD_AVAILABLE
    fprintf(stderr, "\nNote: All tests were skipped because ESIMD is not available.\n");
    return 0;  // Success - graceful skip
#endif

    return all_passed ? 0 : 1;
}
