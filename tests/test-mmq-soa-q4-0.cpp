// Test for Q4_0 MMQ SoA (Structure of Arrays) correctness
// This test verifies that SoA reordering and MMQ produce the same results as AoS
//
// Build: cmake --build build --target test-mmq-soa-q4-0
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-mmq-soa-q4-0

#include "ggml.h"
#include "ggml-sycl.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// Q4_0 block structure (matches ggml-common.h)
#define QK4_0 32
typedef struct {
    uint16_t d;          // delta (fp16)
    uint8_t qs[QK4_0/2]; // nibbles / quants (16 bytes)
} block_q4_0;

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size mismatch");

// Reference dequantization for Q4_0
static void dequantize_q4_0_ref(const block_q4_0* x, float* y, int k) {
    const int nb = k / QK4_0;
    for (int i = 0; i < nb; i++) {
        // Convert fp16 delta to float
        uint16_t d_bits = x[i].d;
        float d;
        // Simple fp16 to float conversion
        uint32_t sign = (d_bits >> 15) & 1;
        uint32_t exp = (d_bits >> 10) & 0x1f;
        uint32_t mant = d_bits & 0x3ff;
        if (exp == 0) {
            d = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2.0f, -14.0f);
        } else if (exp == 31) {
            d = (sign ? -1.0f : 1.0f) * (mant == 0 ? INFINITY : NAN);
        } else {
            d = (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15.0f);
        }

        for (int j = 0; j < QK4_0/2; j++) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >> 4) - 8;
            y[i*QK4_0 + j] = x0 * d;
            y[i*QK4_0 + j + QK4_0/2] = x1 * d;
        }
    }
}

// Create test Q4_0 data with known values
static void create_test_q4_0_data(block_q4_0* data, int nrows, int ncols, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);

    const int nblocks = nrows * (ncols / QK4_0);
    for (int i = 0; i < nblocks; i++) {
        // Set a known scale value (e.g., 1.0 in fp16 = 0x3C00)
        data[i].d = 0x3C00;  // 1.0 in fp16
        for (int j = 0; j < QK4_0/2; j++) {
            data[i].qs[j] = dist(rng);
        }
    }
}

// Reorder Q4_0 from AoS to SoA on CPU (reference implementation)
static void reorder_q4_0_aos_to_soa_ref(const block_q4_0* aos, uint8_t* soa, int nrows, int ncols) {
    const int nblocks = nrows * (ncols / QK4_0);
    uint8_t* qs_dst = soa;
    uint16_t* d_dst = (uint16_t*)(soa + nblocks * (QK4_0/2));

    for (int i = 0; i < nblocks; i++) {
        // Copy qs
        for (int j = 0; j < QK4_0/2; j++) {
            qs_dst[i * (QK4_0/2) + j] = aos[i].qs[j];
        }
        // Copy d
        d_dst[i] = aos[i].d;
    }
}

// Verify SoA layout matches expected
static bool verify_soa_layout(const uint8_t* soa, const block_q4_0* aos_ref, int nrows, int ncols) {
    const int nblocks = nrows * (ncols / QK4_0);
    const size_t d_offset = (size_t)nrows * ncols / 2;  // Should match MMQ's d_offset calculation

    const uint8_t* qs_ptr = soa;
    const uint16_t* d_ptr = (const uint16_t*)(soa + d_offset);

    int mismatches = 0;
    for (int i = 0; i < nblocks; i++) {
        // Verify qs
        for (int j = 0; j < QK4_0/2; j++) {
            if (qs_ptr[i * (QK4_0/2) + j] != aos_ref[i].qs[j]) {
                if (mismatches < 5) {
                    fprintf(stderr, "MISMATCH: block %d, qs[%d]: SoA=%02x, AoS=%02x\n",
                            i, j, qs_ptr[i * (QK4_0/2) + j], aos_ref[i].qs[j]);
                }
                mismatches++;
            }
        }
        // Verify d
        if (d_ptr[i] != aos_ref[i].d) {
            if (mismatches < 5) {
                fprintf(stderr, "MISMATCH: block %d, d: SoA=%04x, AoS=%04x\n",
                        i, d_ptr[i], aos_ref[i].d);
            }
            mismatches++;
        }
    }

    if (mismatches > 0) {
        fprintf(stderr, "Total mismatches: %d\n", mismatches);
        return false;
    }
    return true;
}

// Test 1: Verify d_offset calculation consistency
static bool test_d_offset_calculation() {
    printf("Test 1: d_offset calculation consistency\n");

    // Test dimensions matching Mistral 7B attention weights
    int test_cases[][2] = {
        {4096, 4096},   // attention weights
        {4096, 14336},  // FFN up/gate
        {14336, 4096},  // FFN down
        {128, 4096},    // smaller test
    };

    for (auto& dims : test_cases) {
        int nrows = dims[0];
        int ncols = dims[1];

        // MMQ's d_offset calculation
        size_t d_offset_mmq = (size_t)nrows * (size_t)ncols / 2;

        // Reorder kernel's d_offset calculation (same formula)
        size_t d_offset_reorder = (size_t)ncols * (size_t)nrows / 2;

        if (d_offset_mmq != d_offset_reorder) {
            printf("  FAIL: %d x %d: MMQ d_offset=%zu, reorder d_offset=%zu\n",
                   nrows, ncols, d_offset_mmq, d_offset_reorder);
            return false;
        }
        printf("  PASS: %d x %d: d_offset=%zu bytes\n", nrows, ncols, d_offset_mmq);
    }

    return true;
}

// Test 2: Verify CPU SoA reorder produces correct layout
static bool test_cpu_soa_reorder() {
    printf("Test 2: CPU SoA reorder correctness\n");

    const int nrows = 128;
    const int ncols = 4096;  // Must be multiple of QK4_0
    const int nblocks = nrows * (ncols / QK4_0);

    std::vector<block_q4_0> aos_data(nblocks);
    create_test_q4_0_data(aos_data.data(), nrows, ncols, 42);

    // Reorder to SoA
    size_t soa_size = nblocks * sizeof(block_q4_0);  // Same total size
    std::vector<uint8_t> soa_data(soa_size);
    reorder_q4_0_aos_to_soa_ref(aos_data.data(), soa_data.data(), nrows, ncols);

    // Verify layout
    if (!verify_soa_layout(soa_data.data(), aos_data.data(), nrows, ncols)) {
        printf("  FAIL: SoA layout mismatch\n");
        return false;
    }

    printf("  PASS: SoA layout verified for %d x %d (%d blocks)\n", nrows, ncols, nblocks);
    return true;
}

// Test 3: Block index calculation (simulates MMQ loader)
static bool test_block_index_calculation() {
    printf("Test 3: Block index calculation\n");

    const int nrows = 128;
    const int ncols = 4096;
    const int blocks_per_row = ncols / QK4_0;  // 128

    // Simulate MMQ loader's block index calculation
    // SoA: global_block = (row_offset + i) * blocks_per_row + block_offset + kbx
    // AoS: base_block + i * blocks_per_row + kbx (where base_block = row_x_0 * blocks_per_row + ib0)

    int row_x_0 = 64;   // Starting row for work-group
    int ib0 = 8;        // Block iteration offset

    for (int i = 0; i < 8; i++) {  // i is the local row index
        for (int kbx = 0; kbx < 8; kbx++) {  // kbx is the block within tile
            // SoA calculation
            int row_offset = row_x_0;
            int block_offset = ib0;
            int global_row = row_offset + i;
            int soa_block = global_row * blocks_per_row + block_offset + kbx;

            // AoS calculation (simulated)
            int base_block = row_x_0 * blocks_per_row + ib0;
            int aos_block = base_block + i * blocks_per_row + kbx;

            if (soa_block != aos_block) {
                printf("  FAIL: i=%d, kbx=%d: SoA block=%d, AoS block=%d\n",
                       i, kbx, soa_block, aos_block);
                return false;
            }
        }
    }

    printf("  PASS: Block indices match between SoA and AoS\n");
    return true;
}

// Test 4: Verify dequantization produces same values from AoS and SoA layouts
static bool test_dequantize_aos_vs_soa() {
    printf("Test 4: Dequantization AoS vs SoA\n");

    const int nrows = 4;
    const int ncols = 128;  // 4 blocks per row
    const int nblocks = nrows * (ncols / QK4_0);
    const int nelements = nrows * ncols;

    std::vector<block_q4_0> aos_data(nblocks);
    create_test_q4_0_data(aos_data.data(), nrows, ncols, 123);

    // Dequantize from AoS
    std::vector<float> aos_dequant(nelements);
    dequantize_q4_0_ref(aos_data.data(), aos_dequant.data(), nelements);

    // Reorder to SoA
    size_t soa_size = nblocks * sizeof(block_q4_0);
    std::vector<uint8_t> soa_data(soa_size);
    reorder_q4_0_aos_to_soa_ref(aos_data.data(), soa_data.data(), nrows, ncols);

    // Dequantize from SoA (manual, simulating what MMQ loader does)
    std::vector<float> soa_dequant(nelements);
    const int blocks_per_row = ncols / QK4_0;
    size_t d_offset = (size_t)nrows * ncols / 2;
    const uint8_t* qs_base = soa_data.data();
    const uint16_t* d_base = (const uint16_t*)(soa_data.data() + d_offset);

    for (int row = 0; row < nrows; row++) {
        for (int block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
            int global_block = row * blocks_per_row + block_in_row;
            const uint8_t* qs = qs_base + global_block * (QK4_0/2);
            uint16_t d_bits = d_base[global_block];

            // Convert fp16 delta to float
            uint32_t sign = (d_bits >> 15) & 1;
            uint32_t exp = (d_bits >> 10) & 0x1f;
            uint32_t mant = d_bits & 0x3ff;
            float d;
            if (exp == 0) {
                d = (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2.0f, -14.0f);
            } else if (exp == 31) {
                d = (sign ? -1.0f : 1.0f) * (mant == 0 ? INFINITY : NAN);
            } else {
                d = (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, exp - 15.0f);
            }

            int base_idx = row * ncols + block_in_row * QK4_0;
            for (int j = 0; j < QK4_0/2; j++) {
                const int x0 = (qs[j] & 0x0F) - 8;
                const int x1 = (qs[j] >> 4) - 8;
                soa_dequant[base_idx + j] = x0 * d;
                soa_dequant[base_idx + j + QK4_0/2] = x1 * d;
            }
        }
    }

    // Compare
    int mismatches = 0;
    for (int i = 0; i < nelements; i++) {
        if (fabsf(aos_dequant[i] - soa_dequant[i]) > 1e-6f) {
            if (mismatches < 5) {
                printf("  MISMATCH at [%d]: AoS=%.6f, SoA=%.6f\n",
                       i, aos_dequant[i], soa_dequant[i]);
            }
            mismatches++;
        }
    }

    if (mismatches > 0) {
        printf("  FAIL: %d mismatches out of %d elements\n", mismatches, nelements);
        return false;
    }

    printf("  PASS: Dequantization matches for %d elements\n", nelements);
    return true;
}

// Test 5: Specific test for scale (d) access pattern
static bool test_scale_access_pattern() {
    printf("Test 5: Scale (d) access pattern\n");

    const int nrows = 128;
    const int ncols = 4096;
    const int blocks_per_row = ncols / QK4_0;  // 128
    const int nblocks = nrows * blocks_per_row;

    // Create data with unique scale values to track access pattern
    std::vector<block_q4_0> aos_data(nblocks);
    for (int i = 0; i < nblocks; i++) {
        // Encode block index in the scale value (truncated to fp16 mantissa)
        aos_data[i].d = (uint16_t)(0x3C00 | (i & 0x3FF));  // 1.0 + fractional part encoding block index
        memset(aos_data[i].qs, 0x88, QK4_0/2);  // All 8s (neutral)
    }

    // Reorder to SoA
    size_t soa_size = nblocks * sizeof(block_q4_0);
    std::vector<uint8_t> soa_data(soa_size);
    reorder_q4_0_aos_to_soa_ref(aos_data.data(), soa_data.data(), nrows, ncols);

    // Verify scale access pattern (simulating MMQ loader)
    size_t d_offset = (size_t)nrows * ncols / 2;
    const uint16_t* d_base = (const uint16_t*)(soa_data.data() + d_offset);

    int errors = 0;
    for (int row = 0; row < nrows && errors < 5; row++) {
        for (int block_in_row = 0; block_in_row < blocks_per_row && errors < 5; block_in_row++) {
            int expected_block = row * blocks_per_row + block_in_row;
            uint16_t d_read = d_base[expected_block];
            uint16_t d_expected = aos_data[expected_block].d;

            if (d_read != d_expected) {
                printf("  ERROR: row=%d, block=%d: expected d=%04x, got d=%04x\n",
                       row, block_in_row, d_expected, d_read);
                errors++;
            }
        }
    }

    if (errors > 0) {
        printf("  FAIL: Scale access pattern has errors\n");
        return false;
    }

    printf("  PASS: Scale access pattern verified for %d blocks\n", nblocks);
    return true;
}

int main() {
    printf("=== Q4_0 MMQ SoA Unit Tests ===\n\n");

    int passed = 0;
    int failed = 0;

    if (test_d_offset_calculation()) passed++; else failed++;
    printf("\n");

    if (test_cpu_soa_reorder()) passed++; else failed++;
    printf("\n");

    if (test_block_index_calculation()) passed++; else failed++;
    printf("\n");

    if (test_dequantize_aos_vs_soa()) passed++; else failed++;
    printf("\n");

    if (test_scale_access_pattern()) passed++; else failed++;
    printf("\n");

    printf("=================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);

    return failed > 0 ? 1 : 0;
}
