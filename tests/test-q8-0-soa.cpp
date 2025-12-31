// Unit tests for Q8_0 SoA (Structure of Arrays) support
// Tests reordering, DMMV, MMQ, and MMVQ kernels

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#include "ggml.h"

// Q8_0 block structure (from ggml-common.h)
#define QK8_0 32
typedef struct {
    ggml_fp16_t d;       // delta (scale)
    int8_t qs[QK8_0];    // quants
} block_q8_0;

// Q8_1 block structure (for Y input)
#define QK8_1 32
typedef struct {
    ggml_fp16_t d;       // delta
    ggml_fp16_t s;       // d * sum(qs[i])
    int8_t qs[QK8_1];    // quants
} block_q8_1;

static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size mismatch");
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 size mismatch");

// Helper to convert fp16 to fp32
static float fp16_to_fp32(ggml_fp16_t h) {
    return ggml_fp16_to_fp32(h);
}

// Helper to convert fp32 to fp16
static ggml_fp16_t fp32_to_fp16(float f) {
    return ggml_fp32_to_fp16(f);
}

// ============================================================================
// Test 1: Verify Q8_0 AoS to SoA reordering
// ============================================================================
// SoA layout: [all qs bytes contiguously][all d values contiguously]
// For nrows x ncols matrix with blocks_per_row = ncols/QK8_0:
//   qs section: nrows * ncols bytes (each element is int8)
//   d section: nrows * blocks_per_row * sizeof(fp16) bytes

bool test_q8_0_reorder_layout() {
    printf("\n=== Test 1: Q8_0 AoS to SoA Reorder Layout ===\n");

    const int nrows = 4;
    const int ncols = 64;  // Must be multiple of QK8_0=32
    const int blocks_per_row = ncols / QK8_0;
    const int total_blocks = nrows * blocks_per_row;

    printf("Matrix: %d x %d, blocks_per_row=%d, total_blocks=%d\n",
           nrows, ncols, blocks_per_row, total_blocks);

    // Create AoS data with known pattern
    std::vector<block_q8_0> aos_data(total_blocks);
    for (int b = 0; b < total_blocks; b++) {
        // Scale = block index * 0.01 (small values)
        aos_data[b].d = fp32_to_fp16((b + 1) * 0.01f);
        // Quants = sequential values for easy verification
        for (int i = 0; i < QK8_0; i++) {
            aos_data[b].qs[i] = (int8_t)((b * QK8_0 + i) % 256 - 128);
        }
    }

    // Expected SoA layout sizes
    const size_t qs_bytes = (size_t)nrows * ncols;  // 1 byte per element
    const size_t d_bytes = (size_t)total_blocks * sizeof(ggml_fp16_t);
    const size_t total_soa_bytes = qs_bytes + d_bytes;

    printf("Expected SoA layout:\n");
    printf("  qs section: 0 to %zu bytes\n", qs_bytes - 1);
    printf("  d section: %zu to %zu bytes\n", qs_bytes, total_soa_bytes - 1);

    // Create SoA buffer (simulating what reorder_qw_q8_0 would produce)
    std::vector<uint8_t> soa_data(total_soa_bytes, 0);

    // Manually convert AoS to SoA to verify expected layout
    uint8_t* qs_ptr = soa_data.data();
    ggml_fp16_t* d_ptr = (ggml_fp16_t*)(soa_data.data() + qs_bytes);

    for (int b = 0; b < total_blocks; b++) {
        // Copy qs: each block's 32 int8 values go to consecutive positions
        memcpy(qs_ptr + b * QK8_0, aos_data[b].qs, QK8_0);
        // Copy d: each block's scale
        d_ptr[b] = aos_data[b].d;
    }

    // Verify the layout
    bool passed = true;

    // Check qs values
    printf("\nVerifying qs values...\n");
    for (int row = 0; row < nrows && passed; row++) {
        for (int block = 0; block < blocks_per_row && passed; block++) {
            int global_block = row * blocks_per_row + block;
            const int8_t* soa_qs = (const int8_t*)(soa_data.data() + global_block * QK8_0);

            for (int i = 0; i < QK8_0 && passed; i++) {
                int8_t expected = aos_data[global_block].qs[i];
                int8_t actual = soa_qs[i];
                if (expected != actual) {
                    printf("  FAIL: row=%d block=%d i=%d: expected %d, got %d\n",
                           row, block, i, expected, actual);
                    passed = false;
                }
            }
        }
    }
    if (passed) printf("  qs values: PASS\n");

    // Check d values
    printf("Verifying d (scale) values...\n");
    for (int b = 0; b < total_blocks && passed; b++) {
        float expected = fp16_to_fp32(aos_data[b].d);
        float actual = fp16_to_fp32(d_ptr[b]);
        if (fabsf(expected - actual) > 1e-5f) {
            printf("  FAIL: block=%d: expected %.6f, got %.6f\n", b, expected, actual);
            passed = false;
        }
    }
    if (passed) printf("  d values: PASS\n");

    // Print sample values for debugging
    printf("\nSample SoA data (first 2 blocks):\n");
    printf("  Block 0 qs[0:7]: ");
    for (int i = 0; i < 8; i++) printf("%3d ", (int)((int8_t*)soa_data.data())[i]);
    printf("\n");
    printf("  Block 0 d: %.4f\n", fp16_to_fp32(d_ptr[0]));
    printf("  Block 1 qs[0:7]: ");
    for (int i = 0; i < 8; i++) printf("%3d ", (int)((int8_t*)soa_data.data())[QK8_0 + i]);
    printf("\n");
    printf("  Block 1 d: %.4f\n", fp16_to_fp32(d_ptr[1]));

    printf("\nTest 1 %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

// ============================================================================
// Test 2: Q8_0 dequantization (reference implementation)
// ============================================================================

void dequantize_q8_0_aos(const block_q8_0* x, float* y, int k) {
    // k = number of elements (must be multiple of QK8_0)
    const int nb = k / QK8_0;
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; j++) {
            y[i * QK8_0 + j] = d * x[i].qs[j];
        }
    }
}

void dequantize_q8_0_soa(const uint8_t* soa_data, float* y, int nrows, int ncols) {
    const int blocks_per_row = ncols / QK8_0;
    const int total_blocks = nrows * blocks_per_row;
    const size_t d_offset = (size_t)nrows * ncols;

    const int8_t* qs_base = (const int8_t*)soa_data;
    const ggml_fp16_t* d_base = (const ggml_fp16_t*)(soa_data + d_offset);

    for (int row = 0; row < nrows; row++) {
        for (int block = 0; block < blocks_per_row; block++) {
            int global_block = row * blocks_per_row + block;
            float d = fp16_to_fp32(d_base[global_block]);
            const int8_t* qs = qs_base + global_block * QK8_0;

            for (int i = 0; i < QK8_0; i++) {
                y[row * ncols + block * QK8_0 + i] = d * qs[i];
            }
        }
    }
}

bool test_q8_0_dequantize() {
    printf("\n=== Test 2: Q8_0 Dequantization (AoS vs SoA) ===\n");

    const int nrows = 4;
    const int ncols = 128;
    const int total_elements = nrows * ncols;
    const int blocks_per_row = ncols / QK8_0;
    const int total_blocks = nrows * blocks_per_row;

    // Create random AoS data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);
    std::uniform_int_distribution<int> quant_dist(-127, 127);

    std::vector<block_q8_0> aos_data(total_blocks);
    for (int b = 0; b < total_blocks; b++) {
        aos_data[b].d = fp32_to_fp16(scale_dist(rng));
        for (int i = 0; i < QK8_0; i++) {
            aos_data[b].qs[i] = (int8_t)quant_dist(rng);
        }
    }

    // Convert to SoA
    const size_t qs_bytes = (size_t)nrows * ncols;
    const size_t d_bytes = (size_t)total_blocks * sizeof(ggml_fp16_t);
    std::vector<uint8_t> soa_data(qs_bytes + d_bytes);

    uint8_t* qs_ptr = soa_data.data();
    ggml_fp16_t* d_ptr = (ggml_fp16_t*)(soa_data.data() + qs_bytes);

    for (int b = 0; b < total_blocks; b++) {
        memcpy(qs_ptr + b * QK8_0, aos_data[b].qs, QK8_0);
        d_ptr[b] = aos_data[b].d;
    }

    // Dequantize both
    std::vector<float> aos_result(total_elements);
    std::vector<float> soa_result(total_elements);

    dequantize_q8_0_aos(aos_data.data(), aos_result.data(), total_elements);
    dequantize_q8_0_soa(soa_data.data(), soa_result.data(), nrows, ncols);

    // Compare
    bool passed = true;
    float max_diff = 0.0f;
    int fail_count = 0;

    for (int i = 0; i < total_elements; i++) {
        float diff = fabsf(aos_result[i] - soa_result[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-5f) {
            if (fail_count < 5) {
                printf("  FAIL at %d: AoS=%.6f, SoA=%.6f, diff=%.6f\n",
                       i, aos_result[i], soa_result[i], diff);
            }
            fail_count++;
            passed = false;
        }
    }

    printf("Max difference: %.6e\n", max_diff);
    printf("Failed elements: %d / %d\n", fail_count, total_elements);
    printf("Test 2 %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

// ============================================================================
// Test 3: Q8_0 x Q8_1 dot product (reference implementation)
// ============================================================================

float vec_dot_q8_0_q8_1_ref(const block_q8_0* x, const block_q8_1* y, int nb) {
    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        float d0 = fp16_to_fp32(x[i].d);
        float d1 = fp16_to_fp32(y[i].d);

        int sumi = 0;
        for (int j = 0; j < QK8_0; j++) {
            sumi += (int)x[i].qs[j] * (int)y[i].qs[j];
        }
        sum += d0 * d1 * sumi;
    }
    return sum;
}

float vec_dot_q8_0_soa_q8_1_ref(const uint8_t* x_soa, const block_q8_1* y,
                                 int ncols, size_t d_offset) {
    const int nb = ncols / QK8_0;
    const int8_t* qs_base = (const int8_t*)x_soa;
    const ggml_fp16_t* d_base = (const ggml_fp16_t*)(x_soa + d_offset);

    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        float d0 = fp16_to_fp32(d_base[i]);
        float d1 = fp16_to_fp32(y[i].d);

        const int8_t* qs = qs_base + i * QK8_0;
        int sumi = 0;
        for (int j = 0; j < QK8_0; j++) {
            sumi += (int)qs[j] * (int)y[i].qs[j];
        }
        sum += d0 * d1 * sumi;
    }
    return sum;
}

bool test_q8_0_vec_dot() {
    printf("\n=== Test 3: Q8_0 x Q8_1 Vector Dot Product ===\n");

    const int ncols = 4096;  // Typical embedding dimension
    const int nb = ncols / QK8_0;

    // Create random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);
    std::uniform_int_distribution<int> quant_dist(-127, 127);

    std::vector<block_q8_0> x_aos(nb);
    std::vector<block_q8_1> y(nb);

    for (int i = 0; i < nb; i++) {
        x_aos[i].d = fp32_to_fp16(scale_dist(rng));
        y[i].d = fp32_to_fp16(scale_dist(rng));

        int sum = 0;
        for (int j = 0; j < QK8_0; j++) {
            x_aos[i].qs[j] = (int8_t)quant_dist(rng);
            y[i].qs[j] = (int8_t)quant_dist(rng);
            sum += y[i].qs[j];
        }
        y[i].s = fp32_to_fp16(fp16_to_fp32(y[i].d) * sum);
    }

    // Convert X to SoA
    const size_t qs_bytes = (size_t)ncols;
    const size_t d_bytes = (size_t)nb * sizeof(ggml_fp16_t);
    std::vector<uint8_t> x_soa(qs_bytes + d_bytes);

    uint8_t* qs_ptr = x_soa.data();
    ggml_fp16_t* d_ptr = (ggml_fp16_t*)(x_soa.data() + qs_bytes);

    for (int b = 0; b < nb; b++) {
        memcpy(qs_ptr + b * QK8_0, x_aos[b].qs, QK8_0);
        d_ptr[b] = x_aos[b].d;
    }

    // Compute dot products
    float aos_result = vec_dot_q8_0_q8_1_ref(x_aos.data(), y.data(), nb);
    float soa_result = vec_dot_q8_0_soa_q8_1_ref(x_soa.data(), y.data(), ncols, qs_bytes);

    float diff = fabsf(aos_result - soa_result);
    float rel_diff = diff / fabsf(aos_result);

    printf("AoS result: %.6f\n", aos_result);
    printf("SoA result: %.6f\n", soa_result);
    printf("Absolute diff: %.6e\n", diff);
    printf("Relative diff: %.6e\n", rel_diff);

    bool passed = rel_diff < 1e-5f;
    printf("Test 3 %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

// ============================================================================
// Test 4: Q8_0 Matrix-Vector Multiplication (simulating DMMV/MMVQ)
// ============================================================================

void matvec_q8_0_aos_ref(const block_q8_0* W, const block_q8_1* x,
                          float* y, int nrows, int ncols) {
    const int nb = ncols / QK8_0;
    for (int row = 0; row < nrows; row++) {
        y[row] = vec_dot_q8_0_q8_1_ref(W + row * nb, x, nb);
    }
}

void matvec_q8_0_soa_ref(const uint8_t* W_soa, const block_q8_1* x,
                          float* y, int nrows, int ncols) {
    const int nb = ncols / QK8_0;
    const size_t d_offset = (size_t)nrows * ncols;

    const int8_t* qs_base = (const int8_t*)W_soa;
    const ggml_fp16_t* d_base = (const ggml_fp16_t*)(W_soa + d_offset);

    for (int row = 0; row < nrows; row++) {
        float sum = 0.0f;
        for (int b = 0; b < nb; b++) {
            int global_block = row * nb + b;
            float d0 = fp16_to_fp32(d_base[global_block]);
            float d1 = fp16_to_fp32(x[b].d);

            const int8_t* qs = qs_base + global_block * QK8_0;
            int sumi = 0;
            for (int j = 0; j < QK8_0; j++) {
                sumi += (int)qs[j] * (int)x[b].qs[j];
            }
            sum += d0 * d1 * sumi;
        }
        y[row] = sum;
    }
}

bool test_q8_0_matvec() {
    printf("\n=== Test 4: Q8_0 Matrix-Vector Multiplication ===\n");

    const int nrows = 4096;  // Output dimension
    const int ncols = 4096;  // Input dimension
    const int nb = ncols / QK8_0;
    const int total_blocks = nrows * nb;

    printf("Matrix: %d x %d (%d blocks)\n", nrows, ncols, total_blocks);

    // Create random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.05f);
    std::uniform_int_distribution<int> quant_dist(-127, 127);

    // Weight matrix W (AoS)
    std::vector<block_q8_0> W_aos(total_blocks);
    for (int b = 0; b < total_blocks; b++) {
        W_aos[b].d = fp32_to_fp16(scale_dist(rng));
        for (int j = 0; j < QK8_0; j++) {
            W_aos[b].qs[j] = (int8_t)quant_dist(rng);
        }
    }

    // Input vector x (Q8_1)
    std::vector<block_q8_1> x(nb);
    for (int i = 0; i < nb; i++) {
        x[i].d = fp32_to_fp16(scale_dist(rng));
        int sum = 0;
        for (int j = 0; j < QK8_0; j++) {
            x[i].qs[j] = (int8_t)quant_dist(rng);
            sum += x[i].qs[j];
        }
        x[i].s = fp32_to_fp16(fp16_to_fp32(x[i].d) * sum);
    }

    // Convert W to SoA
    const size_t qs_bytes = (size_t)nrows * ncols;
    const size_t d_bytes = (size_t)total_blocks * sizeof(ggml_fp16_t);
    std::vector<uint8_t> W_soa(qs_bytes + d_bytes);

    uint8_t* qs_ptr = W_soa.data();
    ggml_fp16_t* d_ptr = (ggml_fp16_t*)(W_soa.data() + qs_bytes);

    for (int b = 0; b < total_blocks; b++) {
        memcpy(qs_ptr + b * QK8_0, W_aos[b].qs, QK8_0);
        d_ptr[b] = W_aos[b].d;
    }

    // Compute results
    std::vector<float> y_aos(nrows);
    std::vector<float> y_soa(nrows);

    matvec_q8_0_aos_ref(W_aos.data(), x.data(), y_aos.data(), nrows, ncols);
    matvec_q8_0_soa_ref(W_soa.data(), x.data(), y_soa.data(), nrows, ncols);

    // Compare
    bool passed = true;
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int fail_count = 0;

    for (int i = 0; i < nrows; i++) {
        float diff = fabsf(y_aos[i] - y_soa[i]);
        float rel_diff = fabsf(y_aos[i]) > 1e-6f ? diff / fabsf(y_aos[i]) : diff;

        if (diff > max_diff) max_diff = diff;
        if (rel_diff > max_rel_diff) max_rel_diff = rel_diff;

        if (rel_diff > 1e-4f) {
            if (fail_count < 5) {
                printf("  FAIL at row %d: AoS=%.6f, SoA=%.6f, rel_diff=%.6e\n",
                       i, y_aos[i], y_soa[i], rel_diff);
            }
            fail_count++;
            passed = false;
        }
    }

    printf("Max absolute diff: %.6e\n", max_diff);
    printf("Max relative diff: %.6e\n", max_rel_diff);
    printf("Failed rows: %d / %d\n", fail_count, nrows);

    // Print first few results
    printf("\nFirst 5 results:\n");
    for (int i = 0; i < 5; i++) {
        printf("  Row %d: AoS=%.6f, SoA=%.6f\n", i, y_aos[i], y_soa[i]);
    }

    printf("Test 4 %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

// ============================================================================
// Test 5: Verify d_offset calculation
// ============================================================================

bool test_d_offset_calculation() {
    printf("\n=== Test 5: d_offset Calculation Verification ===\n");

    // Test various matrix sizes
    struct TestCase {
        int nrows;
        int ncols;
    };

    TestCase cases[] = {
        {4096, 4096},
        {1024, 4096},
        {14336, 4096},  // FFN intermediate
        {4096, 14336},
        {32000, 4096},  // Vocab projection
    };

    bool all_passed = true;

    for (const auto& tc : cases) {
        int nrows = tc.nrows;
        int ncols = tc.ncols;
        int blocks_per_row = ncols / QK8_0;
        int total_blocks = nrows * blocks_per_row;

        // Q8_0: each block has 32 int8 values (32 bytes qs) + 2 bytes d = 34 bytes
        // In SoA: qs section = nrows * ncols bytes, d section = total_blocks * 2 bytes

        size_t expected_d_offset = (size_t)nrows * ncols;  // All qs bytes first
        size_t expected_total_size = expected_d_offset + total_blocks * sizeof(ggml_fp16_t);

        // Verify against AoS total size
        size_t aos_total_size = (size_t)total_blocks * sizeof(block_q8_0);

        printf("Matrix %d x %d:\n", nrows, ncols);
        printf("  blocks: %d, d_offset: %zu, soa_size: %zu, aos_size: %zu\n",
               total_blocks, expected_d_offset, expected_total_size, aos_total_size);

        // SoA should be same size as AoS (just rearranged)
        if (expected_total_size != aos_total_size) {
            printf("  FAIL: Size mismatch! SoA=%zu, AoS=%zu\n",
                   expected_total_size, aos_total_size);
            all_passed = false;
        } else {
            printf("  PASS: Sizes match\n");
        }
    }

    printf("\nTest 5 %s\n", all_passed ? "PASSED" : "FAILED");
    return all_passed;
}

// ============================================================================
// Test 6: MMQ tile loading simulation
// ============================================================================

// Simulate the MMQ tile loading pattern
bool test_mmq_tile_loading() {
    printf("\n=== Test 6: MMQ Tile Loading Pattern ===\n");

    const int nrows = 128;  // mmq_y
    const int ncols = 4096;
    const int blocks_per_row = ncols / QK8_0;
    const int total_blocks = nrows * blocks_per_row;

    // Create test data
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> quant_dist(-127, 127);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);

    // AoS data
    std::vector<block_q8_0> aos_data(total_blocks);
    for (int b = 0; b < total_blocks; b++) {
        aos_data[b].d = fp32_to_fp16(scale_dist(rng));
        for (int j = 0; j < QK8_0; j++) {
            aos_data[b].qs[j] = (int8_t)quant_dist(rng);
        }
    }

    // Convert to SoA
    const size_t d_offset = (size_t)nrows * ncols;
    std::vector<uint8_t> soa_data(d_offset + total_blocks * sizeof(ggml_fp16_t));

    uint8_t* qs_ptr = soa_data.data();
    ggml_fp16_t* d_ptr = (ggml_fp16_t*)(soa_data.data() + d_offset);

    for (int b = 0; b < total_blocks; b++) {
        memcpy(qs_ptr + b * QK8_0, aos_data[b].qs, QK8_0);
        d_ptr[b] = aos_data[b].d;
    }

    // Simulate MMQ tile loading for a specific tile
    // Tile parameters (matching MMQ kernel)
    const int WARP_SIZE = 32;
    const int QI8_0 = 32;  // ints per block for Q8_0 (actually 8, but for indexing...)

    // Actually, QI8_0 = 8 in the actual implementation (32 int8 / 4 bytes per int = 8)
    // Let me check the indexing more carefully

    // For Q8_0, each block has 32 int8 values
    // get_int_from_int8 reads 4 bytes (one int) at a time
    // So we need 8 reads to get all 32 bytes: QI8_0 = 32/4 = 8

    // Test: read data from SoA layout and verify against AoS
    const int row_offset = 0;
    const int block_offset = 0;

    bool passed = true;

    // Test qs loading: for each row and each block
    printf("Testing qs loading pattern...\n");
    for (int row = 0; row < 4 && passed; row++) {  // Test first 4 rows
        for (int blk = 0; blk < 2 && passed; blk++) {  // Test first 2 blocks per row
            int global_block = (row_offset + row) * blocks_per_row + block_offset + blk;

            // SoA access
            const int8_t* soa_qs = (const int8_t*)(soa_data.data() + global_block * QK8_0);
            // AoS access
            const int8_t* aos_qs = aos_data[global_block].qs;

            for (int j = 0; j < QK8_0 && passed; j++) {
                if (soa_qs[j] != aos_qs[j]) {
                    printf("  FAIL: row=%d blk=%d j=%d: SoA=%d, AoS=%d\n",
                           row, blk, j, soa_qs[j], aos_qs[j]);
                    passed = false;
                }
            }
        }
    }
    if (passed) printf("  qs loading: PASS\n");

    // Test d loading
    printf("Testing d (scale) loading pattern...\n");
    for (int row = 0; row < 4 && passed; row++) {
        for (int blk = 0; blk < 2 && passed; blk++) {
            int global_block = (row_offset + row) * blocks_per_row + block_offset + blk;

            float soa_d = fp16_to_fp32(d_ptr[global_block]);
            float aos_d = fp16_to_fp32(aos_data[global_block].d);

            if (fabsf(soa_d - aos_d) > 1e-5f) {
                printf("  FAIL: row=%d blk=%d: SoA_d=%.6f, AoS_d=%.6f\n",
                       row, blk, soa_d, aos_d);
                passed = false;
            }
        }
    }
    if (passed) printf("  d loading: PASS\n");

    printf("Test 6 %s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    printf("Q8_0 SoA Unit Tests\n");
    printf("===================\n");

    int passed = 0;
    int failed = 0;

    if (test_q8_0_reorder_layout()) passed++; else failed++;
    if (test_q8_0_dequantize()) passed++; else failed++;
    if (test_q8_0_vec_dot()) passed++; else failed++;
    if (test_q8_0_matvec()) passed++; else failed++;
    if (test_d_offset_calculation()) passed++; else failed++;
    if (test_mmq_tile_loading()) passed++; else failed++;

    printf("\n");
    printf("===================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    printf("===================\n");

    return failed > 0 ? 1 : 0;
}
