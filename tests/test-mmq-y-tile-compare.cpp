// Unit test to compare Y-tile loading between AoS and SoA kernels
// Tests that the SoA kernel correctly reads Y data quantized by quantize_and_reorder_q8_1_soa
//
// Y SoA Layout (per column in Y matrix, which is a "row" in quantizer terms):
//   - qs bytes at: col * y_col_stride + block * QK8_1 + elem
//   - ds (half2) at: col * y_col_stride + nrows_y_unpadded + block * sizeof(half2)
//   - y_col_stride = (nrows_y / QK8_1) * sizeof(block_q8_1) = blocks_per_col * 36
//
// Y AoS Layout:
//   - block_q8_1 array at: y[col * blocks_per_col + block]
//   - Each block_q8_1 = {int8_t qs[32]; half2 ds}

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdint>

// Constants from llama.cpp
#define QK8_1 32
#define QI8_1 8  // number of int32 in a Q8_1 block (32/4)

struct block_q8_1 {
    int8_t qs[QK8_1];
    uint16_t ds[2];  // half2 as two uint16_t (d, sum)
};
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 must be 36 bytes");

// Simulate half precision
float half_to_float(uint16_t h) {
    // Simple conversion for testing (assumes no denormals/special values matter)
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Denormal
        float m = mant / 1024.0f;
        return (sign ? -1.0f : 1.0f) * m * powf(2.0f, -14.0f);
    } else if (exp == 31) {
        return sign ? -INFINITY : INFINITY;
    }

    float m = 1.0f + mant / 1024.0f;
    return (sign ? -1.0f : 1.0f) * m * powf(2.0f, (int)exp - 15);
}

uint16_t float_to_half(float f) {
    if (f == 0.0f) return 0;
    if (std::isinf(f)) return f > 0 ? 0x7c00 : 0xfc00;

    uint32_t* fi = (uint32_t*)&f;
    uint32_t sign = (*fi >> 31) & 0x1;
    int32_t exp = ((*fi >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (*fi >> 13) & 0x3ff;

    if (exp <= 0) return sign << 15;
    if (exp >= 31) return (sign << 15) | 0x7c00;

    return (sign << 15) | (exp << 10) | mant;
}

// CPU simulation of Q8_1 quantization for a single block
void quantize_block_q8_1(const float* x, int8_t* qs, float* d_out, float* sum_out) {
    float max_abs = 0.0f;
    float sum = 0.0f;

    for (int i = 0; i < QK8_1; i++) {
        max_abs = fmaxf(max_abs, fabsf(x[i]));
        sum += x[i];
    }

    float d = max_abs / 127.0f;
    *d_out = d;
    *sum_out = sum;

    if (d == 0.0f) {
        memset(qs, 0, QK8_1);
        return;
    }

    float id = 1.0f / d;
    for (int i = 0; i < QK8_1; i++) {
        qs[i] = (int8_t)roundf(x[i] * id);
    }
}

// Create Y data in AoS format (array of block_q8_1)
void create_y_aos(const float* src, block_q8_1* dst, int nrows_y, int ncols_y) {
    int blocks_per_col = nrows_y / QK8_1;

    for (int col = 0; col < ncols_y; col++) {
        for (int block = 0; block < blocks_per_col; block++) {
            const float* src_block = src + col * nrows_y + block * QK8_1;
            block_q8_1* dst_block = &dst[col * blocks_per_col + block];

            float d, sum;
            quantize_block_q8_1(src_block, dst_block->qs, &d, &sum);
            dst_block->ds[0] = float_to_half(d);
            dst_block->ds[1] = float_to_half(sum);
        }
    }
}

// Create Y data in SoA format
// Layout per column: [qs bytes for all blocks][ds values for all blocks]
// Column stride = blocks_per_col * sizeof(block_q8_1) = blocks_per_col * 36
void create_y_soa(const float* src, char* dst, int nrows_y, int nrows_y_unpadded, int ncols_y) {
    int blocks_per_col = nrows_y / QK8_1;
    int y_col_stride = blocks_per_col * sizeof(block_q8_1);

    printf("  SoA layout: blocks_per_col=%d, y_col_stride=%d, nrows_y_unpadded=%d\n",
           blocks_per_col, y_col_stride, nrows_y_unpadded);

    for (int col = 0; col < ncols_y; col++) {
        char* col_base = dst + col * y_col_stride;
        int8_t* qs_base = (int8_t*)col_base;
        // ds offset = nrows_y_unpadded (this is what kernel uses)
        uint16_t* ds_base = (uint16_t*)(col_base + nrows_y_unpadded);

        for (int block = 0; block < blocks_per_col; block++) {
            // Use actual data rows, not padded
            int src_row_base = block * QK8_1;
            if (src_row_base >= nrows_y_unpadded) {
                // Padding block - zero it
                memset(qs_base + block * QK8_1, 0, QK8_1);
                ds_base[block * 2 + 0] = 0;
                ds_base[block * 2 + 1] = 0;
                continue;
            }

            const float* src_block = src + col * nrows_y_unpadded + src_row_base;
            int8_t* dst_qs = qs_base + block * QK8_1;

            float d, sum;
            // Handle partial block at end
            int elems = std::min(QK8_1, nrows_y_unpadded - src_row_base);
            float block_data[QK8_1] = {0};
            memcpy(block_data, src_block, elems * sizeof(float));

            quantize_block_q8_1(block_data, dst_qs, &d, &sum);
            ds_base[block * 2 + 0] = float_to_half(d);
            ds_base[block * 2 + 1] = float_to_half(sum);
        }
    }
}

// Simulate AoS kernel Y-tile loading (from mul_mat_q8_0 lines 2750-2783)
void load_y_tile_aos(const block_q8_1* y, int* tile_y_qs, float* tile_y_d,
                     int col_y_0, int ib0, int blocks_per_col_y,
                     int mmq_x, int WARP_SIZE) {
    // For simplicity, simulate single thread loading entire tile
    // In real kernel: multiple threads load in parallel

    for (int i = 0; i < mmq_x; i++) {
        int col_y_eff = col_y_0 + i;

        for (int kqs = 0; kqs < WARP_SIZE; kqs++) {
            int kbxd = kqs / QI8_1;  // which block within tile

            const block_q8_1* by0 = &y[col_y_eff * blocks_per_col_y + ib0 + kbxd];

            // Load 4 int8 values as one int32
            int qi = kqs % QI8_1;
            int32_t val = 0;
            memcpy(&val, &by0->qs[qi * 4], 4);

            tile_y_qs[i * WARP_SIZE + kqs] = val;
        }

        // Load d values (scale factors)
        int blocks_per_tile = WARP_SIZE / QI8_1;  // 16/8=2 or 32/8=4
        for (int kby = 0; kby < blocks_per_tile; kby++) {
            const block_q8_1* by0 = &y[col_y_eff * blocks_per_col_y + ib0 + kby];
            tile_y_d[i * blocks_per_tile + kby] = half_to_float(by0->ds[0]);
        }
    }
}

// Simulate SoA kernel Y-tile loading (from mul_mat_q8_0_soa lines 2885-2928)
void load_y_tile_soa(const char* vy, int* tile_y_qs, float* tile_y_d,
                     int col_y_0, int ib0, int nrows_y, int nrows_y_unpadded,
                     int mmq_x, int WARP_SIZE) {
    int blocks_per_col_y = nrows_y / QK8_1;
    int y_col_stride = blocks_per_col_y * sizeof(block_q8_1);

    for (int i = 0; i < mmq_x; i++) {
        int col_y_eff = col_y_0 + i;

        for (int kqs = 0; kqs < WARP_SIZE; kqs++) {
            int kbxd = kqs / QI8_1;
            int block_idx = ib0 + kbxd;

            // Y SoA: qs at col_base + block * QK8_1 + elem
            const int8_t* y_col_qs = (const int8_t*)vy + col_y_eff * y_col_stride;
            const int8_t* y_block_qs = y_col_qs + block_idx * QK8_1;

            int qi = kqs % QI8_1;
            int32_t val = 0;
            memcpy(&val, &y_block_qs[qi * 4], 4);

            tile_y_qs[i * WARP_SIZE + kqs] = val;
        }

        // Load d values from SoA ds section
        int blocks_per_tile = WARP_SIZE / QI8_1;
        for (int kby = 0; kby < blocks_per_tile; kby++) {
            int block_idx = ib0 + kby;

            // Y SoA: ds at col_base + nrows_y_unpadded + block * sizeof(half2)
            const char* y_col_base = vy + col_y_eff * y_col_stride;
            const uint16_t* y_col_ds = (const uint16_t*)(y_col_base + nrows_y_unpadded);

            // ds[0] is d, ds[1] is sum - we want d
            tile_y_d[i * blocks_per_tile + kby] = half_to_float(y_col_ds[block_idx * 2]);
        }
    }
}

// Test with specific padding scenario
int run_test(int nrows_y_unpadded, int nrows_y, int ncols_y, int WARP_SIZE, int mmq_x) {
    printf("\n--- Test: nrows_y_unpadded=%d, nrows_y=%d ---\n", nrows_y_unpadded, nrows_y);

    if (nrows_y_unpadded % QK8_1 != 0) {
        printf("  SKIP: nrows_y_unpadded must be a multiple of QK8_1 for Q8_1 SoA layout\n");
        return 0;
    }

    int blocks_per_col = nrows_y / QK8_1;
    int blocks_per_tile = WARP_SIZE / QI8_1;

    printf("  blocks_per_col=%d, blocks_per_tile=%d\n", blocks_per_col, blocks_per_tile);

    // Create source float data
    std::vector<float> src_y(nrows_y_unpadded * ncols_y);
    for (int i = 0; i < (int)src_y.size(); i++) {
        src_y[i] = (float)(i % 256 - 128) / 10.0f;
    }

    // Allocate quantized buffers
    std::vector<block_q8_1> y_aos(ncols_y * blocks_per_col);
    int y_soa_size = ncols_y * blocks_per_col * sizeof(block_q8_1);
    std::vector<char> y_soa(y_soa_size, 0);

    // Quantize
    create_y_aos(src_y.data(), y_aos.data(), nrows_y, ncols_y);
    create_y_soa(src_y.data(), y_soa.data(), nrows_y, nrows_y_unpadded, ncols_y);

    // Allocate tiles
    int tile_qs_size = mmq_x * WARP_SIZE;
    int tile_d_size = mmq_x * blocks_per_tile;

    std::vector<int> tile_aos_qs(tile_qs_size);
    std::vector<float> tile_aos_d(tile_d_size);
    std::vector<int> tile_soa_qs(tile_qs_size);
    std::vector<float> tile_soa_d(tile_d_size);

    int total_qs_errors = 0;
    int total_d_errors = 0;

    // Test first tile position
    int col_y_0 = 0;
    int ib0 = 0;

    memset(tile_aos_qs.data(), 0, tile_qs_size * sizeof(int));
    memset(tile_aos_d.data(), 0, tile_d_size * sizeof(float));
    memset(tile_soa_qs.data(), 0, tile_qs_size * sizeof(int));
    memset(tile_soa_d.data(), 0, tile_d_size * sizeof(float));

    load_y_tile_aos(y_aos.data(), tile_aos_qs.data(), tile_aos_d.data(),
                   col_y_0, ib0, blocks_per_col, mmq_x, WARP_SIZE);

    load_y_tile_soa(y_soa.data(), tile_soa_qs.data(), tile_soa_d.data(),
                   col_y_0, ib0, nrows_y, nrows_y_unpadded, mmq_x, WARP_SIZE);

    // Compare qs
    for (int i = 0; i < tile_qs_size; i++) {
        if (tile_aos_qs[i] != tile_soa_qs[i]) {
            if (total_qs_errors < 3) {
                printf("  qs MISMATCH at [%d]: AoS=0x%08x SoA=0x%08x\n",
                       i, tile_aos_qs[i], tile_soa_qs[i]);
            }
            total_qs_errors++;
        }
    }
    printf("  qs comparison: %d errors out of %d\n", total_qs_errors, tile_qs_size);

    // Compare d values
    for (int i = 0; i < tile_d_size; i++) {
        float diff = fabsf(tile_aos_d[i] - tile_soa_d[i]);
        if (diff > 1e-5f) {
            if (total_d_errors < 3) {
                printf("  d MISMATCH at [%d]: AoS=%f SoA=%f diff=%f\n",
                       i, tile_aos_d[i], tile_soa_d[i], diff);
            }
            total_d_errors++;
        }
    }
    printf("  d comparison: %d errors out of %d\n", total_d_errors, tile_d_size);

    return total_qs_errors + total_d_errors;
}

int main() {
    printf("=== Y-Tile Loading Comparison Test ===\n\n");

    int total_errors = 0;

    // Test 1: No padding (nrows_y_unpadded == nrows_y)
    total_errors += run_test(128, 128, 4, 32, 4);

    // Test 2: With padding - unpadded is NOT multiple of QK8_1
    // nrows_y_unpadded=100, padded nrows_y=128 (next multiple of 32)
    // This exposes the kx vs nrows_y_unpadded mismatch!
    total_errors += run_test(100, 128, 4, 32, 4);

    // Test 3: Different padding scenario
    // nrows_y_unpadded=50, padded nrows_y=64
    total_errors += run_test(50, 64, 4, 32, 4);

    printf("\n=== FINAL SUMMARY ===\n");
    printf("Total errors across all tests: %d\n", total_errors);

    if (total_errors == 0) {
        printf("\n=== RESULT: PASS ===\n");
        return 0;
    } else {
        printf("\n=== RESULT: FAIL ===\n");
        return 1;
    }
}

int main_old() {
    printf("=== Y-Tile Loading Comparison Test ===\n\n");

    // Test parameters matching real usage
    const int nrows_y_unpadded = 128;  // Actual data rows (like K dimension)
    const int nrows_y = 128;           // Padded to multiple of QK8_1
    const int ncols_y = 4;             // Number of Y columns (batch size)
    const int WARP_SIZE = 32;          // Must match GGML_SYCL_WARP_SIZE
    const int mmq_x = 4;               // Tile width

    int blocks_per_col = nrows_y / QK8_1;
    int blocks_per_tile = WARP_SIZE / QI8_1;

    printf("Test parameters:\n");
    printf("  nrows_y_unpadded=%d, nrows_y=%d, ncols_y=%d\n", nrows_y_unpadded, nrows_y, ncols_y);
    printf("  WARP_SIZE=%d, mmq_x=%d\n", WARP_SIZE, mmq_x);
    printf("  blocks_per_col=%d, blocks_per_tile=%d\n\n", blocks_per_col, blocks_per_tile);

    // Create source float data
    std::vector<float> src_y(nrows_y_unpadded * ncols_y);
    for (int i = 0; i < (int)src_y.size(); i++) {
        src_y[i] = (float)(i % 256 - 128) / 10.0f;  // Range roughly [-12.8, 12.7]
    }

    // Allocate quantized buffers
    std::vector<block_q8_1> y_aos(ncols_y * blocks_per_col);
    int y_soa_size = ncols_y * blocks_per_col * sizeof(block_q8_1);
    std::vector<char> y_soa(y_soa_size, 0);

    // Quantize
    printf("Quantizing Y data...\n");
    create_y_aos(src_y.data(), y_aos.data(), nrows_y, ncols_y);
    create_y_soa(src_y.data(), y_soa.data(), nrows_y, nrows_y_unpadded, ncols_y);

    // Allocate tiles
    int tile_qs_size = mmq_x * WARP_SIZE;
    int tile_d_size = mmq_x * blocks_per_tile;

    std::vector<int> tile_aos_qs(tile_qs_size);
    std::vector<float> tile_aos_d(tile_d_size);
    std::vector<int> tile_soa_qs(tile_qs_size);
    std::vector<float> tile_soa_d(tile_d_size);

    int total_qs_errors = 0;
    int total_d_errors = 0;

    // Test multiple tile positions
    for (int col_y_0 = 0; col_y_0 < ncols_y; col_y_0 += mmq_x) {
        for (int ib0 = 0; ib0 < blocks_per_col; ib0 += blocks_per_tile) {
            printf("\nTesting col_y_0=%d, ib0=%d:\n", col_y_0, ib0);

            // Clear tiles
            memset(tile_aos_qs.data(), 0, tile_qs_size * sizeof(int));
            memset(tile_aos_d.data(), 0, tile_d_size * sizeof(float));
            memset(tile_soa_qs.data(), 0, tile_qs_size * sizeof(int));
            memset(tile_soa_d.data(), 0, tile_d_size * sizeof(float));

            // Load tiles
            load_y_tile_aos(y_aos.data(), tile_aos_qs.data(), tile_aos_d.data(),
                           col_y_0, ib0, blocks_per_col, mmq_x, WARP_SIZE);

            load_y_tile_soa(y_soa.data(), tile_soa_qs.data(), tile_soa_d.data(),
                           col_y_0, ib0, nrows_y, nrows_y_unpadded, mmq_x, WARP_SIZE);

            // Compare qs
            int qs_errors = 0;
            for (int i = 0; i < tile_qs_size; i++) {
                if (tile_aos_qs[i] != tile_soa_qs[i]) {
                    if (qs_errors < 5) {
                        printf("  qs MISMATCH at [%d]: AoS=0x%08x SoA=0x%08x\n",
                               i, tile_aos_qs[i], tile_soa_qs[i]);
                    }
                    qs_errors++;
                }
            }
            printf("  qs comparison: %d errors out of %d\n", qs_errors, tile_qs_size);
            total_qs_errors += qs_errors;

            // Compare d values
            int d_errors = 0;
            for (int i = 0; i < tile_d_size; i++) {
                float diff = fabsf(tile_aos_d[i] - tile_soa_d[i]);
                if (diff > 1e-5f) {
                    if (d_errors < 5) {
                        printf("  d MISMATCH at [%d]: AoS=%f SoA=%f diff=%f\n",
                               i, tile_aos_d[i], tile_soa_d[i], diff);
                    }
                    d_errors++;
                }
            }
            printf("  d comparison: %d errors out of %d\n", d_errors, tile_d_size);
            total_d_errors += d_errors;
        }
    }

    printf("\n=== SUMMARY ===\n");
    printf("Total qs errors: %d\n", total_qs_errors);
    printf("Total d errors: %d\n", total_d_errors);

    if (total_qs_errors == 0 && total_d_errors == 0) {
        printf("\n=== RESULT: PASS ===\n");
        printf("Y-tile loading matches between AoS and SoA!\n");
        return 0;
    } else {
        printf("\n=== RESULT: FAIL ===\n");
        printf("Y-tile loading MISMATCH between AoS and SoA!\n");
        return 1;
    }
}
