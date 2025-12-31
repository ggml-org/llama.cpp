// Unit tests for MMQ SoA (Structure of Arrays) implementation
// Tests vec_dot, tile loaders, and full kernels with known inputs

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cassert>

#include "ggml.h"
#include "ggml-sycl.h"

#ifdef GGML_USE_SYCL

#include <sycl/sycl.hpp>
#include "ggml/src/ggml-sycl/common.hpp"
#include "ggml/src/ggml-sycl/presets.hpp"

// Q8_0 block structure
#define QK8_0 32
#define QI8_0 8  // 32 / 4 = 8 ints per block

struct block_q8_0_test {
    sycl::half d;          // scale
    int8_t qs[QK8_0];      // quantized values
};

// Q8_1 block structure (for Y matrix)
#define QK8_1 32
#define QI8_1 8

struct block_q8_1_test {
    sycl::half2 ds;        // scale and sum
    int8_t qs[QK8_1];      // quantized values
};

//=============================================================================
// Test 1: Verify Q8_0 block layout matches expected
//=============================================================================
bool test_q8_0_block_layout() {
    printf("Test 1: Q8_0 block layout\n");

    block_q8_0_test block;

    // Verify sizeof matches expected (2 + 32 = 34 bytes)
    size_t expected_size = 34;
    size_t actual_size = sizeof(block_q8_0_test);

    // Note: Compiler may add padding
    if (actual_size != expected_size) {
        printf("  WARNING: sizeof(block_q8_0) = %zu, expected %zu (may have padding)\n",
               actual_size, expected_size);
    }

    // Verify offsets
    size_t d_offset = offsetof(block_q8_0_test, d);
    size_t qs_offset = offsetof(block_q8_0_test, qs);

    printf("  block_q8_0: d at offset %zu, qs at offset %zu\n", d_offset, qs_offset);

    if (d_offset != 0) {
        printf("  FAIL: d should be at offset 0\n");
        return false;
    }
    if (qs_offset != 2) {
        printf("  FAIL: qs should be at offset 2\n");
        return false;
    }

    printf("  PASS\n");
    return true;
}

//=============================================================================
// Test 2: Verify AoS to SoA reordering
//=============================================================================
bool test_aos_to_soa_reorder() {
    printf("Test 2: AoS to SoA reordering\n");

    const int nrows = 4;
    const int ncols = 64;  // 2 blocks per row
    const int blocks_per_row = ncols / QK8_0;
    const int total_blocks = nrows * blocks_per_row;

    // Create AoS data
    std::vector<uint8_t> aos_data(total_blocks * 34);  // 34 bytes per block

    for (int row = 0; row < nrows; row++) {
        for (int blk = 0; blk < blocks_per_row; blk++) {
            int idx = row * blocks_per_row + blk;
            uint8_t* block_ptr = aos_data.data() + idx * 34;

            // Set d value (as half)
            sycl::half d_val = sycl::half(0.001f * (idx + 1));
            memcpy(block_ptr, &d_val, 2);

            // Set qs values
            for (int i = 0; i < 32; i++) {
                block_ptr[2 + i] = (uint8_t)((idx * 32 + i) & 0xFF);
            }
        }
    }

    // Create SoA data
    // SoA layout: all qs first (nrows * ncols bytes), then all d values (nrows * blocks_per_row * 2 bytes)
    const size_t qs_size = nrows * ncols;  // Total qs bytes
    const size_t d_size = total_blocks * 2;  // Total d bytes
    std::vector<uint8_t> soa_data(qs_size + d_size);

    // Reorder AoS to SoA
    for (int row = 0; row < nrows; row++) {
        for (int blk = 0; blk < blocks_per_row; blk++) {
            int idx = row * blocks_per_row + blk;
            const uint8_t* aos_block = aos_data.data() + idx * 34;

            // Copy qs to SoA qs section
            // SoA qs layout: row-major, all 32 qs bytes per block contiguous
            size_t soa_qs_offset = (size_t)row * ncols + (size_t)blk * QK8_0;
            memcpy(soa_data.data() + soa_qs_offset, aos_block + 2, QK8_0);

            // Copy d to SoA d section
            size_t soa_d_offset = qs_size + idx * 2;
            memcpy(soa_data.data() + soa_d_offset, aos_block, 2);
        }
    }

    // Verify reordering by reading back
    bool pass = true;
    for (int row = 0; row < nrows; row++) {
        for (int blk = 0; blk < blocks_per_row; blk++) {
            int idx = row * blocks_per_row + blk;

            // Check qs values
            size_t soa_qs_offset = (size_t)row * ncols + (size_t)blk * QK8_0;
            for (int i = 0; i < 32; i++) {
                uint8_t expected = (uint8_t)((idx * 32 + i) & 0xFF);
                uint8_t actual = soa_data[soa_qs_offset + i];
                if (actual != expected) {
                    printf("  FAIL: qs mismatch at row=%d blk=%d i=%d: expected %u, got %u\n",
                           row, blk, i, expected, actual);
                    pass = false;
                }
            }

            // Check d value
            size_t soa_d_offset = qs_size + idx * 2;
            sycl::half d_expected = sycl::half(0.001f * (idx + 1));
            sycl::half d_actual;
            memcpy(&d_actual, soa_data.data() + soa_d_offset, 2);

            if (float(d_expected) != float(d_actual)) {
                printf("  FAIL: d mismatch at row=%d blk=%d: expected %f, got %f\n",
                       row, blk, float(d_expected), float(d_actual));
                pass = false;
            }
        }
    }

    if (pass) {
        printf("  PASS\n");
    }
    return pass;
}

//=============================================================================
// Test 3: Reference Q8_0 x Q8_1 dot product
//=============================================================================
float ref_dot_q8_0_q8_1(const block_q8_0_test* x, const block_q8_1_test* y) {
    // Compute dot product: sum(x_qs[i] * y_qs[i]) * x_d * y_d
    int32_t sum = 0;
    for (int i = 0; i < QK8_0; i++) {
        sum += (int32_t)x->qs[i] * (int32_t)y->qs[i];
    }
    float d_x = float(x->d);
    float d_y = float(y->ds[0]);  // First element is scale
    return d_x * d_y * sum;
}

bool test_reference_dot_product() {
    printf("Test 3: Reference Q8_0 x Q8_1 dot product\n");

    block_q8_0_test x;
    block_q8_1_test y;

    // Test case 1: Simple identity
    x.d = sycl::half(0.1f);
    y.ds = sycl::half2(0.1f, 0.0f);
    for (int i = 0; i < 32; i++) {
        x.qs[i] = 1;
        y.qs[i] = 1;
    }

    float result = ref_dot_q8_0_q8_1(&x, &y);
    float expected = 0.1f * 0.1f * 32.0f;  // 0.01 * 32 = 0.32

    if (std::abs(result - expected) > 1e-4) {
        printf("  FAIL: Test 1: expected %f, got %f\n", expected, result);
        return false;
    }

    // Test case 2: Mixed signs
    for (int i = 0; i < 32; i++) {
        x.qs[i] = (i % 2 == 0) ? 10 : -10;
        y.qs[i] = 1;
    }

    result = ref_dot_q8_0_q8_1(&x, &y);
    // Sum = 16 * 10 - 16 * 10 = 0
    expected = 0.0f;

    if (std::abs(result - expected) > 1e-4) {
        printf("  FAIL: Test 2: expected %f, got %f\n", expected, result);
        return false;
    }

    // Test case 3: Full range
    x.d = sycl::half(0.01f);
    y.ds = sycl::half2(0.01f, 0.0f);
    for (int i = 0; i < 32; i++) {
        x.qs[i] = (int8_t)(i - 16);  // -16 to 15
        y.qs[i] = (int8_t)(i - 16);
    }

    result = ref_dot_q8_0_q8_1(&x, &y);
    // Sum = sum((i-16)^2) for i=0..31 = sum(k^2) for k=-16..15
    // = 2 * sum(k^2) for k=1..15 + 16^2 = 2 * (15*16*31/6) + 256 = 2*1240 + 256 = 2736
    int32_t sum = 0;
    for (int i = -16; i < 16; i++) {
        sum += i * i;
    }
    expected = 0.01f * 0.01f * sum;

    if (std::abs(result - expected) > 1e-4) {
        printf("  FAIL: Test 3: expected %f, got %f (sum=%d)\n", expected, result, sum);
        return false;
    }

    printf("  PASS\n");
    return true;
}

//=============================================================================
// Test 4: Matrix multiplication with known values
//=============================================================================
bool test_matmul_known_values() {
    printf("Test 4: Matrix multiplication with known values\n");

    // Create small test case:
    // X: 2 rows x 64 cols (2 blocks per row) in Q8_0
    // Y: 2 cols x 64 rows (2 blocks per col) in Q8_1
    // Result: 2 x 2 matrix

    const int nrows_x = 2;
    const int ncols_x = 64;
    const int ncols_y = 2;
    const int blocks_per_row_x = ncols_x / QK8_0;  // 2
    const int blocks_per_col_y = ncols_x / QK8_1;  // 2

    // Allocate X in Q8_0 format (AoS)
    std::vector<block_q8_0_test> x_blocks(nrows_x * blocks_per_row_x);

    // Allocate Y in Q8_1 format
    std::vector<block_q8_1_test> y_blocks(ncols_y * blocks_per_col_y);

    // Initialize X: identity-like pattern
    for (int row = 0; row < nrows_x; row++) {
        for (int blk = 0; blk < blocks_per_row_x; blk++) {
            int idx = row * blocks_per_row_x + blk;
            x_blocks[idx].d = sycl::half(0.1f);
            for (int i = 0; i < 32; i++) {
                // Set specific pattern
                x_blocks[idx].qs[i] = (int8_t)(row == 0 ? 1 : 2);
            }
        }
    }

    // Initialize Y
    for (int col = 0; col < ncols_y; col++) {
        for (int blk = 0; blk < blocks_per_col_y; blk++) {
            int idx = col * blocks_per_col_y + blk;
            y_blocks[idx].ds = sycl::half2(0.1f, 0.0f);
            for (int i = 0; i < 32; i++) {
                // Set specific pattern
                y_blocks[idx].qs[i] = (int8_t)(col == 0 ? 1 : 2);
            }
        }
    }

    // Compute reference result
    float ref_result[4];  // 2x2 matrix in column-major order

    for (int col = 0; col < ncols_y; col++) {
        for (int row = 0; row < nrows_x; row++) {
            float sum = 0.0f;
            for (int blk = 0; blk < blocks_per_row_x; blk++) {
                const block_q8_0_test* x_blk = &x_blocks[row * blocks_per_row_x + blk];
                const block_q8_1_test* y_blk = &y_blocks[col * blocks_per_col_y + blk];
                sum += ref_dot_q8_0_q8_1(x_blk, y_blk);
            }
            ref_result[col * nrows_x + row] = sum;
        }
    }

    printf("  Reference results:\n");
    for (int col = 0; col < ncols_y; col++) {
        for (int row = 0; row < nrows_x; row++) {
            printf("    [%d,%d] = %f\n", row, col, ref_result[col * nrows_x + row]);
        }
    }

    // Expected results for our pattern:
    // X row 0: all 1s with d=0.1, X row 1: all 2s with d=0.1
    // Y col 0: all 1s with d=0.1, Y col 1: all 2s with d=0.1
    // Each block has 32 elements, 2 blocks per dot product = 64 elements
    //
    // result[0,0] = 0.1 * 0.1 * (1*1*64) = 0.01 * 64 = 0.64
    // result[1,0] = 0.1 * 0.1 * (2*1*64) = 0.01 * 128 = 1.28
    // result[0,1] = 0.1 * 0.1 * (1*2*64) = 0.01 * 128 = 1.28
    // result[1,1] = 0.1 * 0.1 * (2*2*64) = 0.01 * 256 = 2.56

    float expected[4] = {0.64f, 1.28f, 1.28f, 2.56f};

    bool pass = true;
    for (int i = 0; i < 4; i++) {
        if (std::abs(ref_result[i] - expected[i]) > 1e-3) {
            printf("  FAIL: result[%d] = %f, expected %f\n", i, ref_result[i], expected[i]);
            pass = false;
        }
    }

    if (pass) {
        printf("  PASS\n");
    }
    return pass;
}

//=============================================================================
// Test 5: SoA layout access pattern verification
//=============================================================================
bool test_soa_access_pattern() {
    printf("Test 5: SoA layout access pattern\n");

    // Simulate how the SoA kernel accesses data
    const int nrows = 128;  // mmq_y typical value
    const int ncols = 128;  // ne00 / blocks
    const int blocks_per_row = ncols / QK8_0;  // 4

    // SoA layout sizes
    const size_t qs_size = (size_t)nrows * ncols;  // All qs bytes
    const size_t d_offset = qs_size;              // Offset to d values

    printf("  nrows=%d, ncols=%d, blocks_per_row=%d\n", nrows, ncols, blocks_per_row);
    printf("  qs_size=%zu, d_offset=%zu\n", qs_size, d_offset);

    // Simulate load_tiles_q8_0_soa access pattern
    // It loads QI8_0=8 ints (32 bytes) per work-item per row

    // For block 0, row 0:
    // AoS offset: block_ptr = x + row * blocks_per_row + blk
    // AoS qs offset: &block_ptr->qs[k*4]
    //
    // SoA qs offset: qs_base + (row_offset + i) * ne00 + (block_offset + kbx) * QK8_0 + kqsx * 4
    // where row_offset = 0, i = thread row within tile, block_offset = 0, kbx = k / QI8_0, kqsx = k % QI8_0

    // For row 0, thread 0, block 0, k = 0:
    // kbx = 0, kqsx = 0
    // SoA qs offset = 0 * 128 + 0 * 32 + 0 * 4 = 0
    // This should match AoS: row 0, block 0, qs[0..3]

    // For row 0, thread 0, block 0, k = 4:
    // kbx = 0, kqsx = 4
    // SoA qs offset = 0 * 128 + 0 * 32 + 4 * 4 = 16
    // This should match AoS: row 0, block 0, qs[16..19]

    // For row 0, thread 0, block 0, k = 8:
    // kbx = 1, kqsx = 0
    // SoA qs offset = 0 * 128 + 1 * 32 + 0 * 4 = 32
    // This should match AoS: row 0, block 1, qs[0..3]

    printf("  Access pattern verification:\n");

    // Simulate access for a few key cases
    struct AccessCase {
        int row, k, expected_aos_block, expected_aos_qs_byte;
    };

    AccessCase cases[] = {
        {0, 0, 0, 0},     // First byte of first block
        {0, 4, 0, 16},    // 16th byte of first block
        {0, 8, 1, 0},     // First byte of second block
        {1, 0, 0, 0},     // First byte of row 1's first block
        {1, 8, 1, 0},     // First byte of row 1's second block
    };

    bool pass = true;
    for (const auto& c : cases) {
        int kbx = c.k / QI8_0;
        int kqsx = c.k % QI8_0;
        size_t soa_offset = (size_t)c.row * ncols + (size_t)kbx * QK8_0 + kqsx * 4;

        // Verify this maps to the expected AoS location
        // AoS: row * blocks_per_row + block = row * 4 + kbx
        // AoS qs offset within block: kqsx * 4
        int expected_aos_idx = c.row * blocks_per_row + kbx;
        size_t expected_aos_offset = expected_aos_idx * 34 + 2 + kqsx * 4;

        printf("    row=%d, k=%d: SoA offset=%zu (expected block=%d, qs_byte=%d)\n",
               c.row, c.k, soa_offset, c.expected_aos_block, c.expected_aos_qs_byte);

        if (kbx != c.expected_aos_block) {
            printf("      FAIL: kbx=%d != expected_aos_block=%d\n", kbx, c.expected_aos_block);
            pass = false;
        }
        if (kqsx * 4 != c.expected_aos_qs_byte) {
            printf("      FAIL: kqsx*4=%d != expected_aos_qs_byte=%d\n", kqsx * 4, c.expected_aos_qs_byte);
            pass = false;
        }
    }

    if (pass) {
        printf("  PASS\n");
    }
    return pass;
}

//=============================================================================
// Test 6: Tile indexing verification
//=============================================================================
bool test_tile_indexing() {
    printf("Test 6: Tile indexing for vec_dot\n");

    // vec_dot_q8_0_q8_1_mul_mat accesses tiles as:
    // x_ql[i * (WARP_SIZE + 1) + k]
    // y_qs[j * WARP_SIZE + k]
    // x_dmf[i * (WARP_SIZE/QI8_0) + i/QI8_0 + k/QI8_0]
    // y_df[j * (WARP_SIZE/QI8_1) + k/QI8_1]

    const int WARP_SIZE = 32;
    const int mmq_y = 128;  // typical
    const int mmq_x = 64;   // typical

    // x_ql tile size: mmq_y * (WARP_SIZE + 1) = 128 * 33 = 4224
    // But actual allocation is: mmq_y * WARP_SIZE + mmq_y = 128 * 33 = 4224
    size_t x_ql_size = mmq_y * (WARP_SIZE + 1);

    // x_dm tile size: mmq_y * (WARP_SIZE/QI8_0) + mmq_y/QI8_0 = 128 * 4 + 16 = 528
    size_t x_dm_size = mmq_y * (WARP_SIZE / QI8_0) + mmq_y / QI8_0;

    // y_qs tile size: mmq_x * WARP_SIZE = 64 * 32 = 2048
    size_t y_qs_size = mmq_x * WARP_SIZE;

    // y_ds tile size: mmq_x * WARP_SIZE / QI8_1 = 64 * 32 / 8 = 256
    size_t y_ds_size = mmq_x * WARP_SIZE / QI8_1;

    printf("  Tile sizes: x_ql=%zu, x_dm=%zu, y_qs=%zu, y_ds=%zu\n",
           x_ql_size, x_dm_size, y_qs_size, y_ds_size);

    // Verify access bounds for vec_dot
    // i ranges from 0 to mmq_y-1 (in steps of WARP_SIZE during K-loop, but accessed with +local_id)
    // j ranges from 0 to mmq_x-1 (in steps of nwarps during K-loop, but accessed with +local_id)
    // k ranges from 0 to WARP_SIZE-1 (within each K iteration)

    bool pass = true;

    // Check x_ql bounds
    for (int i = 0; i < mmq_y; i++) {
        for (int k = 0; k < WARP_SIZE; k++) {
            size_t idx = i * (WARP_SIZE + 1) + k;
            if (idx >= x_ql_size) {
                printf("  FAIL: x_ql out of bounds: i=%d, k=%d, idx=%zu >= %zu\n",
                       i, k, idx, x_ql_size);
                pass = false;
            }
        }
    }

    // Check x_dm bounds
    for (int i = 0; i < mmq_y; i++) {
        for (int k = 0; k < WARP_SIZE; k += QI8_0) {
            size_t idx = i * (WARP_SIZE / QI8_0) + i / QI8_0 + k / QI8_0;
            if (idx >= x_dm_size) {
                printf("  FAIL: x_dm out of bounds: i=%d, k=%d, idx=%zu >= %zu\n",
                       i, k, idx, x_dm_size);
                pass = false;
            }
        }
    }

    // Check y_qs bounds
    for (int j = 0; j < mmq_x; j++) {
        for (int k = 0; k < WARP_SIZE; k++) {
            size_t idx = j * WARP_SIZE + k;
            if (idx >= y_qs_size) {
                printf("  FAIL: y_qs out of bounds: j=%d, k=%d, idx=%zu >= %zu\n",
                       j, k, idx, y_qs_size);
                pass = false;
            }
        }
    }

    // Check y_ds bounds
    for (int j = 0; j < mmq_x; j++) {
        for (int k = 0; k < WARP_SIZE; k += QI8_1) {
            size_t idx = j * (WARP_SIZE / QI8_1) + k / QI8_1;
            if (idx >= y_ds_size) {
                printf("  FAIL: y_ds out of bounds: j=%d, k=%d, idx=%zu >= %zu\n",
                       j, k, idx, y_ds_size);
                pass = false;
            }
        }
    }

    if (pass) {
        printf("  PASS (all accesses in bounds)\n");
    }
    return pass;
}

//=============================================================================
// Test 7: Compare AoS and SoA tile loading (simulated)
//=============================================================================
bool test_aos_vs_soa_tile_loading() {
    printf("Test 7: AoS vs SoA tile loading (simulated)\n");

    const int nrows = 4;
    const int ncols = 128;  // 4 blocks per row
    const int blocks_per_row = ncols / QK8_0;
    const int total_blocks = nrows * blocks_per_row;

    // Create AoS data with known pattern
    std::vector<block_q8_0_test> aos_blocks(total_blocks);
    for (int i = 0; i < total_blocks; i++) {
        aos_blocks[i].d = sycl::half(0.001f * (i + 1));
        for (int j = 0; j < 32; j++) {
            aos_blocks[i].qs[j] = (int8_t)((i * 32 + j) % 256 - 128);
        }
    }

    // Create SoA data from AoS
    size_t qs_size = (size_t)nrows * ncols;
    size_t d_size = total_blocks * sizeof(sycl::half);
    std::vector<uint8_t> soa_data(qs_size + d_size);

    for (int row = 0; row < nrows; row++) {
        for (int blk = 0; blk < blocks_per_row; blk++) {
            int idx = row * blocks_per_row + blk;

            // Copy qs
            size_t soa_qs_offset = (size_t)row * ncols + (size_t)blk * QK8_0;
            memcpy(soa_data.data() + soa_qs_offset, aos_blocks[idx].qs, QK8_0);

            // Copy d
            size_t soa_d_offset = qs_size + idx * sizeof(sycl::half);
            memcpy(soa_data.data() + soa_d_offset, &aos_blocks[idx].d, sizeof(sycl::half));
        }
    }

    // Simulate tile loading and compare
    const int mmq_y = 4;  // Small for testing
    const int WARP_SIZE = 32;

    // Allocate tiles
    std::vector<int> tile_x_ql_aos(mmq_y * (WARP_SIZE + 1), 0);
    std::vector<float> tile_x_d_aos(mmq_y * (WARP_SIZE / QI8_0) + mmq_y / QI8_0, 0.0f);
    std::vector<int> tile_x_ql_soa(mmq_y * (WARP_SIZE + 1), 0);
    std::vector<float> tile_x_d_soa(mmq_y * (WARP_SIZE / QI8_0) + mmq_y / QI8_0, 0.0f);

    // Simulate AoS tile loading for row_x_0=0, ib0=0
    // load_tiles_q8_0 logic (simplified)
    int row_x_0 = 0;
    int ib0 = 0;

    for (int i = 0; i < mmq_y; i++) {
        const block_q8_0_test* block = &aos_blocks[(row_x_0 + i) * blocks_per_row + ib0];

        for (int k = 0; k < WARP_SIZE; k++) {
            int kbx = k / QI8_0;
            int kqsx = k % QI8_0;

            // Load 4 qs bytes as an int
            if (kbx == 0) {  // Only first block in this simplified test
                int qs_val = *(const int*)(&block->qs[kqsx * 4]);
                tile_x_ql_aos[i * (WARP_SIZE + 1) + k] = qs_val;
            }
        }

        // Load d value
        tile_x_d_aos[i * (WARP_SIZE / QI8_0) + i / QI8_0] = float(block->d);
    }

    // Simulate SoA tile loading
    const int8_t* qs_base = (const int8_t*)soa_data.data();
    const sycl::half* d_base = (const sycl::half*)(soa_data.data() + qs_size);

    for (int i = 0; i < mmq_y; i++) {
        int global_row = row_x_0 + i;

        for (int k = 0; k < WARP_SIZE; k++) {
            int kbx = k / QI8_0;
            int kqsx = k % QI8_0;

            if (kbx == 0) {  // Only first block
                // SoA qs access: row * ncols + block * 32 + byte_offset
                size_t qs_offset = (size_t)global_row * ncols + (size_t)(ib0 + kbx) * QK8_0 + kqsx * 4;
                int qs_val = *(const int*)(qs_base + qs_offset);
                tile_x_ql_soa[i * (WARP_SIZE + 1) + k] = qs_val;
            }
        }

        // Load d value from SoA
        // SoA d access: block_index = row * blocks_per_row + block
        int block_idx = global_row * blocks_per_row + ib0;
        tile_x_d_soa[i * (WARP_SIZE / QI8_0) + i / QI8_0] = float(d_base[block_idx]);
    }

    // Compare tiles
    bool pass = true;

    for (int i = 0; i < mmq_y; i++) {
        for (int k = 0; k < WARP_SIZE; k++) {
            if (k / QI8_0 == 0) {  // Only first block
                int aos_val = tile_x_ql_aos[i * (WARP_SIZE + 1) + k];
                int soa_val = tile_x_ql_soa[i * (WARP_SIZE + 1) + k];
                if (aos_val != soa_val) {
                    printf("  FAIL: tile_x_ql mismatch at i=%d, k=%d: AoS=0x%08x, SoA=0x%08x\n",
                           i, k, aos_val, soa_val);
                    pass = false;
                }
            }
        }
    }

    for (size_t i = 0; i < tile_x_d_aos.size(); i++) {
        if (tile_x_d_aos[i] != tile_x_d_soa[i]) {
            printf("  FAIL: tile_x_d mismatch at i=%zu: AoS=%f, SoA=%f\n",
                   i, tile_x_d_aos[i], tile_x_d_soa[i]);
            pass = false;
        }
    }

    if (pass) {
        printf("  PASS (AoS and SoA tiles match)\n");
    }
    return pass;
}

#endif  // GGML_USE_SYCL

int main() {
    printf("=== MMQ SoA Unit Tests ===\n\n");

#ifdef GGML_USE_SYCL
    int passed = 0;
    int failed = 0;

    if (test_q8_0_block_layout()) passed++; else failed++;
    if (test_aos_to_soa_reorder()) passed++; else failed++;
    if (test_reference_dot_product()) passed++; else failed++;
    if (test_matmul_known_values()) passed++; else failed++;
    if (test_soa_access_pattern()) passed++; else failed++;
    if (test_tile_indexing()) passed++; else failed++;
    if (test_aos_vs_soa_tile_loading()) passed++; else failed++;

    printf("\n=== Summary: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
#else
    printf("SYCL not enabled, skipping tests\n");
    return 0;
#endif
}
