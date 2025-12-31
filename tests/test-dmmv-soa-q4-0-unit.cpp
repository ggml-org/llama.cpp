// Unit test for DMMV SoA Q4_0 kernel
// Tests the kernel with known input values to verify correctness

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <sycl/sycl.hpp>

// Constants from llama.cpp
#define QK4_0 32
#define QR4_0 2
#define WARP_SIZE 32

// The dequantize function we're testing
static void dequantize_q4_0_reorder(const void *d_ptr, const int64_t ib, const void *qs,
                                    const int iqs, float &vx, float &vy) {
    const float d = (float)*((const sycl::half*)d_ptr + ib);
    const int vui = *((const uint8_t *)qs + iqs);

    vx = vui & 0xF;
    vy = vui >> 4;

    vx = (vx - 8.0f) * d;
    vy = (vy - 8.0f) * d;
}

// Simple reference DMMV implementation (single-threaded, matches kernel logic)
void dmmv_q4_0_soa_reference(
    const void *vx,      // SoA formatted Q4_0 data
    const float *y,      // Input vector
    float *dst,          // Output vector
    int ncols,           // Number of columns
    int nrows,           // Number of rows
    int64_t d_offset)    // Offset to scale values
{
    const char *d_ptr = (const char*)vx + d_offset;

    for (int row = 0; row < nrows; row++) {
        float sum = 0.0f;

        for (int col = 0; col < ncols; col += 2) {
            int ib = (row * ncols + col) / QK4_0;
            int iqs = (col % QK4_0) / QR4_0;
            int iybs = col - (col % QK4_0);

            float vx_val, vy_val;
            int qs_offset = ib * QK4_0 / 2 + iqs;
            dequantize_q4_0_reorder(d_ptr, ib, vx, qs_offset, vx_val, vy_val);

            // Low nibble corresponds to iqs, high nibble to iqs + 16
            sum += vx_val * y[iybs + iqs];
            sum += vy_val * y[iybs + iqs + QK4_0/2];
        }

        dst[row] = sum;
    }
}

// Convert AoS Q4_0 to SoA format
void aos_to_soa_q4_0(const uint8_t* aos, uint8_t* soa, int nblocks, int64_t d_offset) {
    // AoS block_q4_0: [d:2 bytes][qs:16 bytes] = 18 bytes
    // SoA: all qs first (16 bytes per block), then all d (2 bytes per block)

    for (int ib = 0; ib < nblocks; ib++) {
        // Copy qs
        const uint8_t* aos_block = aos + ib * 18;
        memcpy(soa + ib * 16, aos_block + 2, 16);  // Skip the 2-byte d, copy 16-byte qs

        // Copy d
        sycl::half* d_dst = (sycl::half*)(soa + d_offset) + ib;
        memcpy(d_dst, aos_block, 2);  // d is at start of AoS block
    }
}

// Compute expected output using AoS directly (ground truth)
void dmmv_q4_0_aos_reference(
    const uint8_t *aos,  // AoS formatted Q4_0 data
    const float *y,      // Input vector
    float *dst,          // Output vector
    int ncols,           // Number of columns
    int nrows)           // Number of rows
{
    struct block_q4_0 {
        sycl::half d;
        uint8_t qs[16];
    };

    for (int row = 0; row < nrows; row++) {
        float sum = 0.0f;

        for (int col = 0; col < ncols; col += 2) {
            int ib = (row * ncols + col) / QK4_0;
            int iqs = (col % QK4_0) / QR4_0;
            int iybs = col - (col % QK4_0);

            const block_q4_0* x = (const block_q4_0*)aos + ib;
            float d = (float)x->d;
            int vui = x->qs[iqs];

            float vx_val = ((vui & 0xF) - 8.0f) * d;
            float vy_val = ((vui >> 4) - 8.0f) * d;

            sum += vx_val * y[iybs + iqs];
            sum += vy_val * y[iybs + iqs + QK4_0/2];
        }

        dst[row] = sum;
    }
}

int main() {
    printf("=== DMMV SoA Q4_0 Unit Test ===\n\n");

    // Test parameters: 2 rows, 64 columns (2 blocks per row, 4 blocks total)
    const int nrows = 2;
    const int ncols = 64;
    const int nblocks = nrows * ncols / QK4_0;  // 4 blocks
    const int aos_size = nblocks * 18;  // 18 bytes per AoS block
    const int soa_size = nblocks * 16 + nblocks * 2;  // qs + d
    const int64_t d_offset = ncols * nrows / 2;  // 64

    printf("Test config: nrows=%d ncols=%d nblocks=%d\n", nrows, ncols, nblocks);
    printf("d_offset=%lld (should be %d)\n\n", (long long)d_offset, ncols * nrows / 2);

    // Allocate buffers
    std::vector<uint8_t> aos_data(aos_size);
    std::vector<uint8_t> soa_data(soa_size);
    std::vector<float> y(ncols);
    std::vector<float> dst_aos(nrows);
    std::vector<float> dst_soa(nrows);

    // Initialize with known values
    // For simplicity: all weights = 1.0 (d=0.125, nibbles=8+1=9 for low, 8+1=9 for high)
    // This gives dequantized value of (9-8)*0.125 = 0.125 for all elements
    for (int ib = 0; ib < nblocks; ib++) {
        uint8_t* block = aos_data.data() + ib * 18;

        // d = 0.125 as fp16
        sycl::half d_val = sycl::half(0.125f);
        memcpy(block, &d_val, 2);

        // qs: all nibbles = 9 (gives value 1 after dequant)
        // byte = (high_nibble << 4) | low_nibble = (9 << 4) | 9 = 0x99
        for (int j = 0; j < 16; j++) {
            block[2 + j] = 0x99;
        }
    }

    // y = 1.0 for all elements
    for (int i = 0; i < ncols; i++) {
        y[i] = 1.0f;
    }

    // Convert to SoA
    aos_to_soa_q4_0(aos_data.data(), soa_data.data(), nblocks, d_offset);

    // Debug: print layouts
    printf("AoS first block (18 bytes): ");
    for (int i = 0; i < 18; i++) printf("%02x ", aos_data[i]);
    printf("\n");

    printf("SoA first 32 bytes (qs for 2 blocks): ");
    for (int i = 0; i < 32; i++) printf("%02x ", soa_data[i]);
    printf("\n");

    printf("SoA d values at offset %lld: ", (long long)d_offset);
    for (int i = 0; i < 8; i++) printf("%02x ", soa_data[d_offset + i]);
    printf("\n");

    sycl::half d0, d1;
    memcpy(&d0, &soa_data[d_offset], 2);
    memcpy(&d1, &soa_data[d_offset + 2], 2);
    printf("d[0]=%.6f d[1]=%.6f\n\n", (float)d0, (float)d1);

    // Compute reference outputs
    printf("Computing AoS reference...\n");
    dmmv_q4_0_aos_reference(aos_data.data(), y.data(), dst_aos.data(), ncols, nrows);

    printf("Computing SoA reference...\n");
    dmmv_q4_0_soa_reference(soa_data.data(), y.data(), dst_soa.data(), ncols, nrows, d_offset);

    // Expected: each row has 64 elements, all dequantized to 0.125
    // Sum = 64 * 0.125 * 1.0 = 8.0
    printf("\n=== Results ===\n");
    printf("Expected per row: 64 * 0.125 * 1.0 = 8.0\n\n");

    bool passed = true;
    for (int row = 0; row < nrows; row++) {
        printf("Row %d: AoS=%.6f  SoA=%.6f  Expected=8.0\n",
               row, dst_aos[row], dst_soa[row]);

        if (fabs(dst_aos[row] - 8.0f) > 0.01f) {
            printf("  ERROR: AoS output wrong!\n");
            passed = false;
        }
        if (fabs(dst_soa[row] - 8.0f) > 0.01f) {
            printf("  ERROR: SoA output wrong!\n");
            passed = false;
        }
        if (fabs(dst_aos[row] - dst_soa[row]) > 0.01f) {
            printf("  ERROR: AoS and SoA don't match!\n");
            passed = false;
        }
    }

    printf("\n=== %s ===\n", passed ? "PASSED" : "FAILED");

    // More detailed trace for debugging
    if (!passed) {
        printf("\n=== Detailed trace for row 0 ===\n");
        const char *d_ptr = (const char*)soa_data.data() + d_offset;

        float sum = 0.0f;
        for (int col = 0; col < ncols; col += 2) {
            int ib = col / QK4_0;  // For row 0
            int iqs = (col % QK4_0) / QR4_0;
            int iybs = col - (col % QK4_0);
            int qs_offset = ib * 16 + iqs;

            float d = (float)*((const sycl::half*)d_ptr + ib);
            int vui = soa_data[qs_offset];
            float vx_val = ((vui & 0xF) - 8.0f) * d;
            float vy_val = ((vui >> 4) - 8.0f) * d;

            printf("col=%2d: ib=%d iqs=%d qs_offset=%d vui=0x%02x d=%.4f vx=%.4f vy=%.4f y[%d]=%.2f y[%d]=%.2f\n",
                   col, ib, iqs, qs_offset, vui, d, vx_val, vy_val,
                   iybs + iqs, y[iybs + iqs],
                   iybs + iqs + 16, y[iybs + iqs + 16]);

            sum += vx_val * y[iybs + iqs];
            sum += vy_val * y[iybs + iqs + 16];
        }
        printf("Final sum: %.6f\n", sum);
    }

    return passed ? 0 : 1;
}
