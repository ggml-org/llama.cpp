// Step-by-step Q6_K SoA kernel debug test
// Traces through load_tiles and vec_dot to find where NaN appears

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml-quants.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// Q6_K constants (from ggml-common.h)
#define QK_K 256
#define QI6_K (QK_K / 8)   // 32
#define QR6_K 2

// MMQ tile constants
#define MMQ_TILE_NE_K 32
#define WARP_SIZE 32

// Helper: get int from uint8 array (matches SYCL kernel)
static int get_int_from_uint8(const uint8_t* x, int i) {
    return x[4*i] | (x[4*i+1] << 8) | (x[4*i+2] << 16) | (x[4*i+3] << 24);
}

// Helper: get int from int8 array
static int get_int_from_int8(const int8_t* x, int i) {
    const uint8_t* xu = (const uint8_t*)x;
    return xu[4*i] | (xu[4*i+1] << 8) | (xu[4*i+2] << 16) | (xu[4*i+3] << 24);
}

// Helper: vectorized subtract-saturate (mimics dpct::vectorized_binary with sub_sat)
static int vec_sub_sat_char4(int a, int b) {
    int8_t a0 = (a >> 0) & 0xFF;
    int8_t a1 = (a >> 8) & 0xFF;
    int8_t a2 = (a >> 16) & 0xFF;
    int8_t a3 = (a >> 24) & 0xFF;
    int8_t b0 = (b >> 0) & 0xFF;
    int8_t b1 = (b >> 8) & 0xFF;
    int8_t b2 = (b >> 16) & 0xFF;
    int8_t b3 = (b >> 24) & 0xFF;

    // Saturating subtract
    auto sat_sub = [](int8_t x, int8_t y) -> int8_t {
        int result = (int)x - (int)y;
        if (result < -128) return -128;
        if (result > 127) return 127;
        return (int8_t)result;
    };

    uint8_t r0 = (uint8_t)sat_sub(a0, b0);
    uint8_t r1 = (uint8_t)sat_sub(a1, b1);
    uint8_t r2 = (uint8_t)sat_sub(a2, b2);
    uint8_t r3 = (uint8_t)sat_sub(a3, b3);

    return r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
}

// Simulate load_tiles_q6_K_soa for a single row/block
void debug_load_tiles_q6_K_soa(
    const uint8_t* qs_base,      // SoA data start
    size_t qh_offset,
    size_t scales_offset,
    size_t d_offset,
    int nblocks,                 // total blocks
    int blocks_per_row,
    int row,                     // which row to load
    int block_in_row,            // which block in that row
    // Output tiles (simplified - just for one row/block)
    int* x_ql,                   // [2 * MMQ_TILE_NE_K + 1]
    float* x_dm,                 // [1]
    int* x_sc                    // [MMQ_TILE_NE_K/8]
) {
    const uint8_t* qh_base = qs_base + qh_offset;
    const int8_t* scales_base_ptr = (const int8_t*)(qs_base + scales_offset);
    const uint16_t* d_base = (const uint16_t*)(qs_base + d_offset);  // ggml_fp16_t

    int global_block = row * blocks_per_row + block_in_row;

    printf("\n=== load_tiles_q6_K_soa debug ===\n");
    printf("row=%d, block_in_row=%d, global_block=%d\n", row, block_in_row, global_block);
    printf("nblocks=%d, blocks_per_row=%d\n", nblocks, blocks_per_row);
    printf("qh_offset=%zu, scales_offset=%zu, d_offset=%zu\n", qh_offset, scales_offset, d_offset);

    // Check global_block is valid
    if (global_block >= nblocks) {
        printf("ERROR: global_block %d >= nblocks %d!\n", global_block, nblocks);
        return;
    }

    // Load d value
    uint16_t d_raw = d_base[global_block];
    float d_float = ggml_fp16_to_fp32(d_raw);
    *x_dm = d_float;
    printf("d_base[%d] = 0x%04x = %.6f\n", global_block, d_raw, d_float);

    if (std::isnan(d_float) || std::isinf(d_float)) {
        printf("*** d value is NaN or Inf! ***\n");
    }

    // Load scales (16 bytes per block)
    const int8_t* sc_ptr = scales_base_ptr + global_block * (QK_K/16);
    printf("scales for block %d (first 8): ", global_block);
    for (int i = 0; i < 8; i++) {
        printf("%d ", sc_ptr[i]);
    }
    printf("\n");

    // Load ql and qh for k=0 (first thread's work)
    const int k = 0;
    const int kbx = k / QI6_K;  // 0
    const int kqsx = k % QI6_K; // 0

    // ql pointer for this block
    const uint8_t* ql_ptr = qs_base + global_block * (QK_K/2);  // 128 bytes per block
    // qh pointer for this block
    const uint8_t* qh_ptr = qh_base + global_block * (QK_K/4);  // 64 bytes per block

    printf("ql_ptr offset from base: %zu\n", (size_t)(ql_ptr - qs_base));
    printf("qh_ptr offset from base: %zu\n", (size_t)(qh_ptr - qs_base));

    // Read first few bytes of ql
    printf("ql bytes (first 16): ");
    for (int i = 0; i < 16; i++) {
        printf("%02x ", ql_ptr[i]);
    }
    printf("\n");

    // Read first few bytes of qh
    printf("qh bytes (first 16): ");
    for (int i = 0; i < 16; i++) {
        printf("%02x ", qh_ptr[i]);
    }
    printf("\n");

    // Compute ql and qh for k=0
    const int ky = QR6_K * kqsx;  // 0

    int ql_int = get_int_from_uint8(ql_ptr, kqsx);  // kqsx=0
    int ql0 = (ql_int >> 0) & 0x0F0F0F0F;
    int ql1 = (ql_int >> 4) & 0x0F0F0F0F;

    printf("ql_int = 0x%08x, ql0 = 0x%08x, ql1 = 0x%08x\n", ql_int, ql0, ql1);

    int qh_int = get_int_from_uint8(qh_ptr, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
    int qh0 = ((qh_int >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
    int qh1 = (qh_int >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) & 0x30303030;

    printf("qh_int = 0x%08x, qh0 = 0x%08x, qh1 = 0x%08x\n", qh_int, qh0, qh1);

    // Combine and subtract bias
    int x_ql0 = vec_sub_sat_char4(ql0 | qh0, 0x20202020);
    int x_ql1 = vec_sub_sat_char4(ql1 | qh1, 0x20202020);

    printf("x_ql0 (after sub 32) = 0x%08x\n", x_ql0);
    printf("x_ql1 (after sub 32) = 0x%08x\n", x_ql1);

    // Store in x_ql
    const int kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
    const int kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

    printf("kq0=%d, kq1=%d\n", kq0, kq1);

    x_ql[kq0] = x_ql0;
    x_ql[kq1] = x_ql1;

    // Extract individual q values for verification
    int8_t q_vals[8];
    q_vals[0] = (x_ql0 >> 0) & 0xFF;
    q_vals[1] = (x_ql0 >> 8) & 0xFF;
    q_vals[2] = (x_ql0 >> 16) & 0xFF;
    q_vals[3] = (x_ql0 >> 24) & 0xFF;
    q_vals[4] = (x_ql1 >> 0) & 0xFF;
    q_vals[5] = (x_ql1 >> 8) & 0xFF;
    q_vals[6] = (x_ql1 >> 16) & 0xFF;
    q_vals[7] = (x_ql1 >> 24) & 0xFF;

    printf("Extracted q values (first 8): ");
    for (int i = 0; i < 8; i++) {
        printf("%d ", q_vals[i]);
    }
    printf("\n");

    // Load a scale value for verification
    int sc_int = get_int_from_int8(sc_ptr, 0);
    x_sc[0] = sc_int;
    printf("sc_int[0] = 0x%08x\n", sc_int);

    // Extract individual scale values
    int8_t sc_vals[4];
    sc_vals[0] = (sc_int >> 0) & 0xFF;
    sc_vals[1] = (sc_int >> 8) & 0xFF;
    sc_vals[2] = (sc_int >> 16) & 0xFF;
    sc_vals[3] = (sc_int >> 24) & 0xFF;
    printf("Scale values (first 4): %d %d %d %d\n", sc_vals[0], sc_vals[1], sc_vals[2], sc_vals[3]);
}

// Simplified vec_dot for Q6_K (just for one element)
float debug_vec_dot_q6k(
    int x_ql,           // packed 4 x int8 quants
    int8_t scale,       // scale for this group
    float d,            // d value
    const int8_t* y_qs, // Y quant values (4 values)
    float y_d           // Y scale
) {
    printf("\n=== vec_dot debug ===\n");
    printf("x_ql = 0x%08x, scale = %d, d = %.6f\n", x_ql, scale, d);
    printf("y_d = %.6f\n", y_d);

    // Extract x quants
    int8_t x0 = (x_ql >> 0) & 0xFF;
    int8_t x1 = (x_ql >> 8) & 0xFF;
    int8_t x2 = (x_ql >> 16) & 0xFF;
    int8_t x3 = (x_ql >> 24) & 0xFF;

    printf("x quants: %d %d %d %d\n", x0, x1, x2, x3);
    printf("y quants: %d %d %d %d\n", y_qs[0], y_qs[1], y_qs[2], y_qs[3]);

    // Integer dot product
    int sum = x0 * y_qs[0] + x1 * y_qs[1] + x2 * y_qs[2] + x3 * y_qs[3];
    printf("int dot product sum = %d\n", sum);

    // Apply scales
    float result = d * scale * y_d * sum;
    printf("result = d * scale * y_d * sum = %.6f * %d * %.6f * %d = %.6f\n",
           d, scale, y_d, sum, result);

    if (std::isnan(result)) {
        printf("*** RESULT IS NaN! ***\n");
        printf("Checking components:\n");
        printf("  d = %.6f (isnan=%d, isinf=%d)\n", d, std::isnan(d), std::isinf(d));
        printf("  scale = %d\n", scale);
        printf("  y_d = %.6f (isnan=%d, isinf=%d)\n", y_d, std::isnan(y_d), std::isinf(y_d));
        printf("  sum = %d\n", sum);
    }

    return result;
}

int main() {
    printf("Q6_K SoA Step-by-Step Debug Test\n");
    printf("=================================\n\n");

    // Small test case: 2 rows, 1 block per row
    const int n_rows = 2;
    const int n_cols = QK_K;  // 256, so 1 block per row
    const int blocks_per_row = n_cols / QK_K;  // 1
    const int nblocks = n_rows * blocks_per_row;  // 2

    printf("Test setup: %d rows, %d cols, %d blocks_per_row, %d total blocks\n",
           n_rows, n_cols, blocks_per_row, nblocks);

    // Create original AoS Q6_K data
    std::vector<block_q6_K> aos_data(nblocks);

    // Initialize with known pattern
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int b = 0; b < nblocks; b++) {
        // Set d to a small but valid value
        float d_val = 0.0001f + 0.00001f * b;
        aos_data[b].d = ggml_fp32_to_fp16(d_val);

        // Set scales to small values
        for (int i = 0; i < QK_K/16; i++) {
            aos_data[b].scales[i] = (int8_t)(1 + (i % 7));
        }

        // Set ql (low 4 bits)
        for (int i = 0; i < QK_K/2; i++) {
            aos_data[b].ql[i] = (uint8_t)(rng() & 0xFF);
        }

        // Set qh (high 2 bits)
        for (int i = 0; i < QK_K/4; i++) {
            aos_data[b].qh[i] = (uint8_t)(rng() & 0xFF);
        }
    }

    printf("\nOriginal AoS d values:\n");
    for (int b = 0; b < nblocks; b++) {
        float d = ggml_fp16_to_fp32(aos_data[b].d);
        printf("  block[%d].d = 0x%04x = %.6f\n", b, aos_data[b].d, d);
    }

    // Create SoA layout
    // Layout: ql (nblocks*128) | qh (nblocks*64) | scales (nblocks*16) | d (nblocks*2)
    const size_t ql_size = nblocks * (QK_K/2);      // 128 bytes per block
    const size_t qh_size = nblocks * (QK_K/4);      // 64 bytes per block
    const size_t scales_size = nblocks * (QK_K/16); // 16 bytes per block
    const size_t d_size = nblocks * sizeof(ggml_fp16_t);  // 2 bytes per block

    const size_t qh_offset = ql_size;
    const size_t scales_offset = qh_offset + qh_size;
    const size_t d_offset = scales_offset + scales_size;
    const size_t total_size = d_offset + d_size;

    printf("\nSoA layout:\n");
    printf("  ql:     [0, %zu)\n", ql_size);
    printf("  qh:     [%zu, %zu)\n", qh_offset, qh_offset + qh_size);
    printf("  scales: [%zu, %zu)\n", scales_offset, scales_offset + scales_size);
    printf("  d:      [%zu, %zu)\n", d_offset, d_offset + d_size);
    printf("  total:  %zu bytes\n", total_size);

    std::vector<uint8_t> soa_data(total_size);

    // Convert AoS to SoA
    for (int b = 0; b < nblocks; b++) {
        // Copy ql
        memcpy(&soa_data[b * (QK_K/2)], aos_data[b].ql, QK_K/2);
        // Copy qh
        memcpy(&soa_data[qh_offset + b * (QK_K/4)], aos_data[b].qh, QK_K/4);
        // Copy scales
        memcpy(&soa_data[scales_offset + b * (QK_K/16)], aos_data[b].scales, QK_K/16);
        // Copy d
        memcpy(&soa_data[d_offset + b * sizeof(ggml_fp16_t)], &aos_data[b].d, sizeof(ggml_fp16_t));
    }

    // Verify d values in SoA
    printf("\nSoA d values:\n");
    const uint16_t* soa_d = (const uint16_t*)&soa_data[d_offset];
    for (int b = 0; b < nblocks; b++) {
        float d = ggml_fp16_to_fp32(soa_d[b]);
        printf("  soa_d[%d] = 0x%04x = %.6f\n", b, soa_d[b], d);
    }

    // Now run the load_tiles debug for block 0
    int x_ql[2 * MMQ_TILE_NE_K + 1] = {0};
    float x_dm = 0.0f;
    int x_sc[MMQ_TILE_NE_K/8] = {0};

    debug_load_tiles_q6_K_soa(
        soa_data.data(),
        qh_offset,
        scales_offset,
        d_offset,
        nblocks,
        blocks_per_row,
        0,  // row 0
        0,  // block 0
        x_ql,
        &x_dm,
        x_sc
    );

    // Create fake Y data for vec_dot test
    int8_t y_qs[4] = {10, -5, 3, -8};
    float y_d = 0.05f;

    // Extract scale for first group
    int8_t scale0 = (x_sc[0] >> 0) & 0xFF;

    // Run vec_dot debug
    float dot_result = debug_vec_dot_q6k(x_ql[0], scale0, x_dm, y_qs, y_d);

    printf("\n=== Final Result ===\n");
    printf("dot_result = %.6f\n", dot_result);

    if (std::isnan(dot_result)) {
        printf("FAIL: Result is NaN\n");
        return 1;
    } else if (std::isinf(dot_result)) {
        printf("FAIL: Result is Inf\n");
        return 1;
    } else {
        printf("PASS: Result is valid\n");
        return 0;
    }
}
