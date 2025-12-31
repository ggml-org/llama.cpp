// tests/test-dmmv-q4-0-coalesced.cpp
// Test that Q4_0 DMMV with coalesced layout produces same output as SoA
//
// This is a TDD test - written to fail until the coalesced DMMV kernel
// is properly integrated and produces correct results.
//
// The coalesced layout reorganizes SoA data into tile-based format for
// better cache line utilization during DMMV operations.
//
// Build: cmake --build build --target test-dmmv-q4-0-coalesced
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-dmmv-q4-0-coalesced

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sycl/sycl.hpp>
#include <vector>

// Q4_0 block structure (must match ggml-common.h)
#define QK4_0 32
#define QR4_0 2

// Coalesced tile configuration (must match common.hpp)
constexpr int MMVQ_COALESCED_TILE_BLOCKS     = 16;
constexpr int MMVQ_COALESCED_TILE_BYTES_Q4_0 = MMVQ_COALESCED_TILE_BLOCKS * 16;  // 256 bytes quants per tile

typedef struct {
    sycl::half d;
    uint8_t    qs[QK4_0 / 2];
} block_q4_0_test;

static_assert(sizeof(block_q4_0_test) == 18, "block_q4_0 size mismatch");

// =============================================================================
// CPU Reference Implementation
// =============================================================================

static void dequantize_block_q4_0_cpu(const block_q4_0_test * block, float * out) {
    const float d = float(block->d);
    for (int i = 0; i < QK4_0 / 2; i++) {
        const uint8_t byte = block->qs[i];
        const int     lo   = (byte & 0xF);
        const int     hi   = (byte >> 4);
        out[i]             = (lo - 8) * d;
        out[i + QK4_0 / 2] = (hi - 8) * d;
    }
}

static void dmmv_q4_0_cpu_reference(const block_q4_0_test * x, const float * y, float * dst, int ncols, int nrows) {
    const int blocks_per_row = ncols / QK4_0;

    for (int row = 0; row < nrows; row++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_0_test * block = &x[row * blocks_per_row + b];
            float                   dequant[QK4_0];
            dequantize_block_q4_0_cpu(block, dequant);

            for (int i = 0; i < QK4_0; i++) {
                sum += dequant[i] * y[b * QK4_0 + i];
            }
        }
        dst[row] = sum;
    }
}

// =============================================================================
// SoA Layout Conversion (AoS -> SoA)
// =============================================================================

static void convert_aos_to_soa(const block_q4_0_test * aos_data, uint8_t * soa_data, int total_blocks) {
    // SoA layout: all quants first, then all scales
    const size_t qs_bytes = total_blocks * (QK4_0 / 2);

    // Copy quants
    for (int b = 0; b < total_blocks; b++) {
        for (int i = 0; i < QK4_0 / 2; i++) {
            soa_data[b * (QK4_0 / 2) + i] = aos_data[b].qs[i];
        }
    }

    // Copy scales
    sycl::half * scales = reinterpret_cast<sycl::half *>(soa_data + qs_bytes);
    for (int b = 0; b < total_blocks; b++) {
        scales[b] = aos_data[b].d;
    }
}

// =============================================================================
// Coalesced Layout Conversion (SoA -> Coalesced)
// =============================================================================

static void convert_soa_to_coalesced(const uint8_t * soa_data, uint8_t * coalesced_data, int ncols, int nrows) {
    const int    blocks_per_row = ncols / QK4_0;
    const int    total_blocks   = nrows * blocks_per_row;
    const size_t qs_bytes       = total_blocks * (QK4_0 / 2);

    constexpr int TILE_BLOCKS     = MMVQ_COALESCED_TILE_BLOCKS;  // 16
    constexpr int BYTES_PER_BLOCK = QK4_0 / 2;                   // 16 bytes quants per Q4_0 block
    constexpr int WORDS_PER_BLOCK = BYTES_PER_BLOCK / 4;         // 4 words per block

    const int tiles_per_row = blocks_per_row / TILE_BLOCKS;

    // Coalesced row stride: quants + scales per row
    const int coalesced_row_stride = blocks_per_row * BYTES_PER_BLOCK + blocks_per_row * sizeof(sycl::half);

    // SoA pointers
    const uint8_t *    soa_qs     = soa_data;
    const sycl::half * soa_scales = reinterpret_cast<const sycl::half *>(soa_data + qs_bytes);

    for (int row = 0; row < nrows; row++) {
        uint8_t * row_out = coalesced_data + row * coalesced_row_stride;

        for (int tile = 0; tile < tiles_per_row; tile++) {
            uint8_t * tile_out = row_out + tile * MMVQ_COALESCED_TILE_BYTES_Q4_0;

            // Coalesced layout within tile:
            // Word w of block b is at: word_offset = w * (TILE_BLOCKS * 4) + b * 4
            // So all word-0s are together, then all word-1s, etc.

            for (int b = 0; b < TILE_BLOCKS; b++) {
                int             global_block = row * blocks_per_row + tile * TILE_BLOCKS + b;
                const uint8_t * src_block    = soa_qs + global_block * BYTES_PER_BLOCK;

                // Reorder words within tile
                for (int w = 0; w < WORDS_PER_BLOCK; w++) {
                    int dst_offset = w * (TILE_BLOCKS * 4) + b * 4;
                    memcpy(tile_out + dst_offset, src_block + w * 4, 4);
                }
            }
        }

        // Copy scales (not coalesced, remain block-sequential)
        sycl::half * row_scales = reinterpret_cast<sycl::half *>(row_out + blocks_per_row * BYTES_PER_BLOCK);
        for (int b = 0; b < blocks_per_row; b++) {
            row_scales[b] = soa_scales[row * blocks_per_row + b];
        }
    }
}

// =============================================================================
// Test: Compare SoA DMMV vs Coalesced DMMV
// =============================================================================

bool test_coalesced_vs_soa_dmmv(sycl::queue & q, int ncols, int nrows) {
    printf("\n=== Test: Coalesced vs SoA DMMV (%d x %d) ===\n", nrows, ncols);

    const int blocks_per_row = ncols / QK4_0;
    const int total_blocks   = nrows * blocks_per_row;

    // Check if dimensions are compatible with coalesced layout
    if (blocks_per_row % MMVQ_COALESCED_TILE_BLOCKS != 0) {
        printf("SKIP: blocks_per_row=%d not divisible by tile size %d\n", blocks_per_row, MMVQ_COALESCED_TILE_BLOCKS);
        return true;  // Skip, not a failure
    }

    // Generate random test data
    std::mt19937                          rng(42 + ncols + nrows);
    std::uniform_real_distribution<float> scale_dist(-0.1f, 0.1f);
    std::uniform_int_distribution<int>    byte_dist(0, 255);
    std::uniform_real_distribution<float> y_dist(-1.0f, 1.0f);

    std::vector<block_q4_0_test> h_aos(total_blocks);
    for (int b = 0; b < total_blocks; b++) {
        h_aos[b].d = sycl::half(scale_dist(rng));
        for (int i = 0; i < QK4_0 / 2; i++) {
            h_aos[b].qs[i] = static_cast<uint8_t>(byte_dist(rng));
        }
    }

    std::vector<float> h_y(ncols);
    for (int i = 0; i < ncols; i++) {
        h_y[i] = y_dist(rng);
    }

    // CPU reference output
    std::vector<float> cpu_output(nrows);
    dmmv_q4_0_cpu_reference(h_aos.data(), h_y.data(), cpu_output.data(), ncols, nrows);

    // Create SoA layout
    const size_t soa_qs_bytes    = total_blocks * (QK4_0 / 2);
    const size_t soa_d_bytes     = total_blocks * sizeof(sycl::half);
    const size_t soa_total_bytes = soa_qs_bytes + soa_d_bytes;

    std::vector<uint8_t> h_soa(soa_total_bytes);
    convert_aos_to_soa(h_aos.data(), h_soa.data(), total_blocks);

    // Create coalesced layout
    const int    coalesced_row_stride  = blocks_per_row * (QK4_0 / 2) + blocks_per_row * sizeof(sycl::half);
    const size_t coalesced_total_bytes = nrows * coalesced_row_stride;

    std::vector<uint8_t> h_coalesced(coalesced_total_bytes);
    convert_soa_to_coalesced(h_soa.data(), h_coalesced.data(), ncols, nrows);

    // TODO: Once the coalesced DMMV kernel is properly exposed via ggml API,
    // we would run both SoA and coalesced versions and compare.
    //
    // For now, this test fails to drive TDD - the kernel exists in dmmv.cpp
    // but needs proper integration testing.

    printf("  CPU reference computed: first 4 values = ");
    for (int i = 0; i < std::min(4, nrows); i++) {
        printf("%.4f ", cpu_output[i]);
    }
    printf("\n");

    printf("  SoA data size: %zu bytes\n", soa_total_bytes);
    printf("  Coalesced data size: %zu bytes\n", coalesced_total_bytes);
    printf("  Coalesced row stride: %d bytes\n", coalesced_row_stride);

    // This test should fail until proper coalesced DMMV is integrated
    fprintf(stderr, "ERROR: Coalesced DMMV kernel integration test not implemented\n");
    return false;
}

// =============================================================================
// Test: Coalesced Layout Data Integrity
// =============================================================================

bool test_coalesced_layout_integrity(int ncols, int nrows) {
    printf("\n=== Test: Coalesced Layout Data Integrity (%d x %d) ===\n", nrows, ncols);

    const int blocks_per_row = ncols / QK4_0;
    const int total_blocks   = nrows * blocks_per_row;

    if (blocks_per_row % MMVQ_COALESCED_TILE_BLOCKS != 0) {
        printf("SKIP: blocks_per_row=%d not divisible by tile size %d\n", blocks_per_row, MMVQ_COALESCED_TILE_BLOCKS);
        return true;
    }

    // Generate deterministic test data
    std::vector<block_q4_0_test> h_aos(total_blocks);
    for (int b = 0; b < total_blocks; b++) {
        h_aos[b].d = sycl::half(0.01f * (b % 100 + 1));
        for (int i = 0; i < QK4_0 / 2; i++) {
            h_aos[b].qs[i] = static_cast<uint8_t>((b + i) % 256);
        }
    }

    // Create SoA layout
    const size_t soa_qs_bytes    = total_blocks * (QK4_0 / 2);
    const size_t soa_d_bytes     = total_blocks * sizeof(sycl::half);
    const size_t soa_total_bytes = soa_qs_bytes + soa_d_bytes;

    std::vector<uint8_t> h_soa(soa_total_bytes);
    convert_aos_to_soa(h_aos.data(), h_soa.data(), total_blocks);

    // Create coalesced layout
    const int    coalesced_row_stride  = blocks_per_row * (QK4_0 / 2) + blocks_per_row * sizeof(sycl::half);
    const size_t coalesced_total_bytes = nrows * coalesced_row_stride;

    std::vector<uint8_t> h_coalesced(coalesced_total_bytes);
    convert_soa_to_coalesced(h_soa.data(), h_coalesced.data(), ncols, nrows);

    // Verify that all data is preserved (we can dequantize and get same values)
    bool passed = true;
    int  errors = 0;

    constexpr int TILE_BLOCKS     = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int BYTES_PER_BLOCK = QK4_0 / 2;
    constexpr int WORDS_PER_BLOCK = BYTES_PER_BLOCK / 4;

    for (int row = 0; row < nrows && errors < 10; row++) {
        const uint8_t *    row_data = h_coalesced.data() + row * coalesced_row_stride;
        const sycl::half * row_scales =
            reinterpret_cast<const sycl::half *>(row_data + blocks_per_row * BYTES_PER_BLOCK);

        int tiles_per_row = blocks_per_row / TILE_BLOCKS;

        for (int tile = 0; tile < tiles_per_row && errors < 10; tile++) {
            const uint8_t * tile_data = row_data + tile * MMVQ_COALESCED_TILE_BYTES_Q4_0;

            for (int b = 0; b < TILE_BLOCKS && errors < 10; b++) {
                int global_block = row * blocks_per_row + tile * TILE_BLOCKS + b;

                // Extract quants from coalesced layout
                uint8_t extracted_qs[BYTES_PER_BLOCK];
                for (int w = 0; w < WORDS_PER_BLOCK; w++) {
                    int src_offset = w * (TILE_BLOCKS * 4) + b * 4;
                    memcpy(extracted_qs + w * 4, tile_data + src_offset, 4);
                }

                // Compare with original AoS data
                for (int i = 0; i < BYTES_PER_BLOCK; i++) {
                    if (extracted_qs[i] != h_aos[global_block].qs[i]) {
                        fprintf(stderr,
                            "  FAIL: row=%d tile=%d block=%d byte=%d: "
                            "expected=0x%02x got=0x%02x\n",
                            row, tile, b, i, h_aos[global_block].qs[i], extracted_qs[i]);
                        passed = false;
                        errors++;
                    }
                }

                // Check scale
                sycl::half expected_scale = h_aos[global_block].d;
                sycl::half actual_scale   = row_scales[tile * TILE_BLOCKS + b];
                if (std::fabs(float(expected_scale) - float(actual_scale)) > 1e-6f) {
                    fprintf(stderr, "  FAIL: row=%d block=%d: scale expected=%.4f got=%.4f\n", row,
                           global_block % blocks_per_row, float(expected_scale), float(actual_scale));
                    passed = false;
                    errors++;
                }
            }
        }
    }

    printf("  Layout integrity: %s (errors=%d)\n", passed ? "PASSED" : "FAILED", errors);
    return passed;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("=== Q4_0 DMMV Coalesced Layout Tests ===\n");
    printf("Testing coalesced memory layout for DMMV kernel\n");
    printf("Tile configuration: %d blocks per tile, %d bytes per tile\n\n", MMVQ_COALESCED_TILE_BLOCKS,
           MMVQ_COALESCED_TILE_BYTES_Q4_0);

    // Test layout integrity first (CPU-only, no GPU needed)
    printf("=== Part 1: Layout Integrity Tests (CPU) ===\n");

    bool layout_ok = true;
    layout_ok &= test_coalesced_layout_integrity(512, 16);   // 512/32 = 16 blocks per row
    layout_ok &= test_coalesced_layout_integrity(1024, 32);  // 1024/32 = 32 blocks per row
    layout_ok &= test_coalesced_layout_integrity(4096, 64);  // 4096/32 = 128 blocks per row (64 rows)

    if (!layout_ok) {
        printf("\nLayout integrity tests FAILED\n");
        return 1;
    }
    printf("\nLayout integrity tests PASSED\n");

    // GPU tests
    printf("\n=== Part 2: GPU Kernel Tests ===\n");

    sycl::device gpu_dev;
    try {
        gpu_dev = sycl::device(sycl::gpu_selector_v);
    } catch (const sycl::exception& e) {
        fprintf(stderr, "ERROR: SYCL exception: %s\n", e.what());
        return 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "ERROR: Exception: %s\n", e.what());
        return 1;
    }

    printf("Device: %s\n", gpu_dev.get_info<sycl::info::device::name>().c_str());

    sycl::queue q(gpu_dev, sycl::property::queue::in_order());

    int passed = 0;
    int failed = 0;

    // Test various matrix sizes
    std::vector<std::pair<int, int>> test_cases = {
        { 512,  16    }, // Small
        { 1024, 64    }, // Medium
        { 4096, 256   }, // Large
        { 4096, 4096  }, // Model-sized
        { 4096, 14336 }, // Mistral FFN dimension
    };

    for (const auto & tc : test_cases) {
        if (test_coalesced_vs_soa_dmmv(q, tc.first, tc.second)) {
            passed++;
        } else {
            failed++;
        }
    }

    printf("\n=== Summary ===\n");
    printf("Layout tests: PASSED\n");
    printf("GPU tests: %d passed, %d failed\n", passed, failed);

    if (failed > 0) {
        printf("\nFAIL: Coalesced DMMV kernel not implemented or not producing correct results\n");
        printf("This test is designed to fail until the coalesced kernel integration is complete.\n");
        return 1;
    }

    printf("\nAll tests PASSED\n");
    return 0;
}
